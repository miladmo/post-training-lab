"""Small, reproducible LoRA experiments on fictional context-checking tasks."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
from itertools import zip_longest
import json
import math
from pathlib import Path
import time

from tasks import audit_dataset, build_dataset, reward, score_completion, write_dataset
from prompts import prompt_manifest, prompt_messages
from summaries import aggregate_results


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def select_rows(rows, limit):
    """Interleave roles so short validation runs still cover all three roles."""
    groups = [[row for row in rows if row["role"] == role]
              for role in ("HR", "Sales", "Science")]
    ordered = [row for triplet in zip_longest(*groups) for row in triplet if row is not None]
    return ordered[:limit] if limit else ordered


def setup(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    torch.set_num_threads(args.threads)
    set_seed(args.seed)
    if args.snapshot:
        source = Path(args.snapshot).resolve()
        expected_repo = "models--" + args.model.replace("/", "--")
        if (source.name != args.revision or source.parent.name != "snapshots"
                or source.parent.parent.name != expected_repo):
            raise ValueError("--snapshot must be the cache snapshot of the specified model and commit.")
    else:
        from huggingface_hub import snapshot_download
        source = Path(snapshot_download(args.model, revision=args.revision,
                      allow_patterns=["*.json", "*.safetensors", "*.jinja", "merges.txt", "vocab.json"]))
    contents = {}
    for path in sorted(source.iterdir()):
        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            contents[path.name] = digest.hexdigest()
    args.model_files_sha256 = contents
    kwargs = {"trust_remote_code": False, "local_files_only": bool(args.snapshot)}
    if not args.snapshot:
        kwargs["revision"] = args.revision
    tokenizer = AutoTokenizer.from_pretrained(source, **kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        source, dtype=torch.float32, attn_implementation="eager", **kwargs)
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable.")
        model.to("cuda")
    return model, tokenizer


def render(tokenizer, messages, style="original"):
    return tokenizer.apply_chat_template(
        prompt_messages(messages, style), tokenize=False,
        add_generation_prompt=True, enable_thinking=False)


def common_metadata(args):
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model, "revision": args.revision,
        "seed": args.seed, "device": args.device, "threads": args.threads,
        "dtype": "float32", "thinking": False,
        "model_files_sha256": args.model_files_sha256,
        "prompt": prompt_manifest(args.prompt_style),
        "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                          for name in ("run.py", "tasks.py", "prompts.py", "summaries.py")},
        "versions": {name: importlib.metadata.version(name)
                     for name in ("torch", "transformers", "trl", "peft", "datasets",
                                  "accelerate", "tokenizers", "safetensors")},
    }


def fresh_directory(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def train(args):
    preparation_started = time.monotonic()
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from trl import DPOConfig, DPOTrainer, GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer

    splits = build_dataset()
    audit_dataset(splits)
    rows = splits["train"]
    output = fresh_directory(args.output)
    model_loading_started = time.monotonic()
    model, tokenizer = setup(args)
    model_loading_seconds = time.monotonic() - model_loading_started
    model = get_peft_model(model, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0, target_modules=["q_proj", "v_proj"],
        bias="none", task_type="CAUSAL_LM"))
    before = {name: param.detach().cpu().clone()
              for name, param in model.named_parameters() if param.requires_grad}
    data = []
    for row in rows:
        prompt = render(tokenizer, row["prompt"], args.prompt_style)
        chosen = row["chosen"][0]["content"] + tokenizer.eos_token
        rejected = row["rejected"][0]["content"] + tokenizer.eos_token
        # Truncating required evidence or a target would change the task.
        for answer in (chosen, rejected):
            if len(tokenizer(prompt + answer, add_special_tokens=False)["input_ids"]) > args.max_length:
                raise ValueError(f"{row['id']} exceeds --max-length; no targets were truncated.")
        if args.method == "sft":
            data.append({"prompt": prompt, "completion": chosen})
        elif args.method == "dpo":
            data.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
        else:
            data.append({"prompt": prompt_messages(row["prompt"], args.prompt_style),
                         "expected": row["expected"]})
    options = dict(
        output_dir=str(output / "checkpoints"), max_steps=args.steps,
        per_device_train_batch_size=1, gradient_accumulation_steps=2 if args.method == "grpo" else 1,
        learning_rate=args.learning_rate, seed=args.seed, data_seed=args.seed,
        use_cpu=args.device == "cpu", bf16=False, fp16=False,
        optim="adamw_torch", dataloader_num_workers=0, dataloader_pin_memory=False,
        report_to="none", push_to_hub=False, save_strategy="no",
        logging_steps=max(1, min(10, args.steps // 4)),
        gradient_checkpointing=False,
    )
    dataset = Dataset.from_list(data)
    trainer_initialization_started = time.monotonic()
    if args.method == "sft":
        config = SFTConfig(**options, max_length=args.max_length,
                           completion_only_loss=True, packing=False)
        trainer = SFTTrainer(model=model, args=config, train_dataset=dataset,
                             processing_class=tokenizer)
    elif args.method == "dpo":
        # TRL freezes a copy of this fresh, zero-output adapter as its reference.
        # Both therefore start at the instruction base, independently of SFT.
        config = DPOConfig(**options, max_length=args.max_length, beta=0.1,
                           loss_type="sigmoid", precompute_ref_log_probs=True)
        class TimedDPOTrainer(DPOTrainer):
            def compute_ref_log_probs(self, reference_model, inputs):
                if args.device == "cuda":
                    torch.cuda.synchronize()
                reference_started = time.monotonic()
                result = super().compute_ref_log_probs(reference_model, inputs)
                if args.device == "cuda":
                    torch.cuda.synchronize()
                self.reference_precompute_seconds = (
                    getattr(self, "reference_precompute_seconds", 0.0)
                    + time.monotonic() - reference_started)
                return result

        trainer = TimedDPOTrainer(model=model, args=config, train_dataset=dataset,
                                  processing_class=tokenizer)
    else:
        config = GRPOConfig(**options, num_generations=2, max_completion_length=96,
                            use_vllm=False, beta=0.0, loss_type="dapo",
                            chat_template_kwargs={"enable_thinking": False})
        trainer = GRPOTrainer(model=model, args=config, train_dataset=dataset,
                              processing_class=tokenizer, reward_funcs=reward)
    started = time.monotonic()
    trainer_initialization_seconds = started - trainer_initialization_started
    preparation_seconds = started - preparation_started
    result = trainer.train()
    seconds = time.monotonic() - started
    delta_squared = 0.0
    for name, param in model.named_parameters():
        if name in before:
            current = param.detach().cpu()
            if not torch.isfinite(current).all():
                raise ValueError("Training produced non-finite adapter parameters.")
            delta_squared += (current - before[name]).double().square().sum().item()
    trainer.save_model(str(output / "adapter"))
    tokenizer.save_pretrained(output / "adapter")
    metadata = common_metadata(args) | {
        "method": args.method, "steps": args.steps, "max_length": args.max_length,
        "learning_rate": args.learning_rate, "train_count": len(rows),
        "training_data_sha256": fingerprint(rows), "train_seconds": seconds,
        "preparation_seconds": preparation_seconds,
        "model_loading_seconds": model_loading_seconds,
        "trainer_initialization_seconds": trainer_initialization_seconds,
        "reference_precompute_seconds": getattr(trainer, "reference_precompute_seconds", 0.0),
        "total_seconds": time.monotonic() - preparation_started,
        "completed_steps": trainer.state.global_step,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "adapter_delta_l2": delta_squared ** 0.5,
        "metrics": result.metrics, "log_history": trainer.state.log_history,
    }
    save(output / "training.json", metadata)
    print(json.dumps({key: metadata[key] for key in
                     ("method", "steps", "train_seconds", "adapter_delta_l2")}), flush=True)


def evaluate(args):
    import torch
    from peft import PeftModel

    splits = build_dataset()
    audit_dataset(splits)
    rows = select_rows(splits[args.split], args.limit)
    adapter_meta = None
    if args.adapter:
        adapter_path = Path(args.adapter)
        adapter_meta = json.loads((adapter_path.parent / "training.json").read_text())
        for key in ("model", "revision"):
            if adapter_meta[key] != getattr(args, key):
                raise ValueError(f"Adapter {key} differs from evaluation base.")
        training_prompt = adapter_meta.get("prompt", prompt_manifest("original"))
        if training_prompt != prompt_manifest(args.prompt_style):
            raise ValueError("Adapter and evaluation prompt protocols differ.")
    output = fresh_directory(args.output)
    model, tokenizer = setup(args)
    if args.adapter:
        if adapter_meta["model_files_sha256"] != args.model_files_sha256:
            raise ValueError("Actual model files differ from the adapter's training base.")
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    model.eval()
    predictions = []
    started = time.monotonic()
    for row in rows:
        prompt = render(tokenizer, row["prompt"], args.prompt_style)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        case_started = time.monotonic()
        with torch.inference_mode():
            generated = model.generate(
                **inputs, do_sample=False, max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        new_tokens = generated[0, inputs["input_ids"].shape[1]:]
        completion = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        record = {"id": row["id"], "role": row["role"], "family": row["family"],
                  "prompt_sha256": fingerprint(prompt), "expected": row["expected"],
                  "completion": completion, "metrics": score_completion(completion, row["expected"]),
                  "new_tokens": len(new_tokens), "seconds": time.monotonic() - case_started}
        predictions.append(record)
        print(json.dumps({"id": row["id"], "completion": completion,
                          "metrics": record["metrics"]}), flush=True)
        save(output / "predictions.json", predictions)
    summary = aggregate_results(rows, predictions)
    metadata = common_metadata(args) | {
        "condition": adapter_meta["method"] if adapter_meta else "base",
        "adapter_training_sha256": fingerprint(adapter_meta) if adapter_meta else None,
        "split": args.split, "count": len(rows), "data_sha256": fingerprint(rows),
        "generation": {"do_sample": False, "max_new_tokens": args.max_new_tokens},
        **summary, "seconds": time.monotonic() - started,
    }
    save(output / "evaluation.json", metadata)
    print(json.dumps(metadata), flush=True)


def compare(directories):
    loaded = []
    for directory in directories:
        path = Path(directory)
        meta = json.loads((path / "evaluation.json").read_text())
        predictions = json.loads((path / "predictions.json").read_text())
        signature = {key: meta[key] for key in (
            "model", "revision", "split", "count", "data_sha256", "generation",
            "thinking", "dtype", "device", "threads", "seed", "versions", "model_files_sha256")}
        signature["prompt"] = meta.get("prompt", prompt_manifest("original"))
        signature["source_sha256"] = meta.get("source_sha256")
        signature["cases"] = [(row["id"], row["prompt_sha256"], row["expected"])
                              for row in predictions]
        if len(predictions) != meta["count"]:
            raise ValueError("Incomplete evaluation.")
        if len({row["id"] for row in predictions}) != len(predictions):
            raise ValueError("Duplicate evaluation case IDs.")
        for row in predictions:
            if row["metrics"] != score_completion(row["completion"], row["expected"]):
                raise ValueError("Saved metrics differ from the actual completions.")
        actual = {key: sum(row["metrics"][key] for row in predictions) / len(predictions)
                  for key in predictions[0]["metrics"]}
        if actual != meta["metrics"]:
            raise ValueError("Aggregate metrics differ from case scores.")
        if "context_pairs" in meta or "roles" in meta:
            indexed = {row["id"]: row for row in build_dataset()[meta["split"]]}
            try:
                source_rows = [indexed[row["id"]] for row in predictions]
            except KeyError as error:
                raise ValueError("Unknown evaluation case.") from error
            summary = aggregate_results(source_rows, predictions)
            if any(meta[key] != summary[key] for key in ("metrics", "roles", "context_pairs")):
                raise ValueError("Role or context-pair summary differs from case scores.")
        if loaded and signature != loaded[0][0]:
            raise ValueError("Evaluation settings or cases differ; comparison rejected.")
        loaded.append((signature, meta))
    return [{key: meta[key] for key in ("condition", "count", "metrics", "seconds", "roles", "context_pairs")
             if key in meta}
            for _, meta in loaded]


def positive(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("Must be positive.")
    return value


def learning_rate(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("Learning rate must be finite and positive.")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--output", default="data")
    comparison = commands.add_parser("compare")
    comparison.add_argument("directories", nargs="+", type=Path)
    for name in ("train", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--model", default="Qwen/Qwen3-0.6B")
        command.add_argument("--revision", default="c1899de289a04d12100db370d81485cdf75e47ca")
        command.add_argument("--snapshot", help="Local snapshot of the specified model and revision")
        command.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
        command.add_argument("--threads", type=positive, default=4)
        command.add_argument("--seed", type=int, default=42)
        command.add_argument("--prompt-style", choices=("original", "two-shot"), default="original")
        command.add_argument("--output", required=True)
        if name == "train":
            command.add_argument("--method", choices=("sft", "dpo", "grpo"), default="sft")
            command.add_argument("--steps", type=positive, default=4)
            command.add_argument("--max-length", type=positive, default=512)
            command.add_argument("--learning-rate", type=learning_rate, default=2e-4)
        else:
            command.add_argument("--adapter")
            command.add_argument("--split", choices=("validation", "test"), default="validation")
            command.add_argument("--limit", type=positive)
            command.add_argument("--max-new-tokens", type=positive, default=96)
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps(write_dataset(args.output), indent=2))
    elif args.command == "compare":
        print(json.dumps(compare(args.directories), indent=2))
    elif args.command == "train":
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
