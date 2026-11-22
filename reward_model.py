"""Reward model training, reliability checks and best-of-N selection.

A learned reward model (RM) is trained on the rule-derived preference pairs. It
is then evaluated as a measurement instrument before it is used: pairwise
accuracy on held-out pairs, robustness to malformed variants of correct
answers, and agreement with the verifiable rule reward on sampled completions.
Best-of-N compares RM selection with a single sample, majority voting and the
rule-based oracle on identical candidates. See PROTOCOL_V4.md.
"""

from collections import Counter
import json
from pathlib import Path
import time

from run import (common_metadata, fingerprint, fresh_directory, load_adapter_metadata, render,
                 save, select_rows, setup)
from summaries import aggregate_results
from tasks import _loads, _valid_prediction, audit_dataset, build_dataset, graded_reward, score_completion

SELECTIONS = ("first_sample", "majority_vote", "reward_model", "oracle")


def corruptions(answer):
    """Deterministic malformed variants of a correct answer (all fail the strict schema)."""
    data = json.loads(answer)
    renamed = {("evidence" if key == "evidence_ids" else key): value for key, value in data.items()}
    return {
        "trailing_text": answer + " Done.",
        "markdown_fence": "```json\n" + answer + "\n```",
        "truncated": answer[: len(answer) // 2],
        "renamed_key": json.dumps(renamed, separators=(",", ":")),
    }


def rankdata(values):
    """Ranks starting at 1 with ties sharing their average rank."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(x, y):
    """Spearman correlation; None when either variable is constant."""
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("Spearman correlation needs two equally long sequences of length >= 2.")
    rx, ry = rankdata(x), rankdata(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    if sxx == 0 or syy == 0:
        return None
    return sxy / (sxx * syy) ** 0.5


def canonical_answer(text):
    """Canonical JSON of a valid decision object, else None (used for majority voting)."""
    try:
        value = _loads(text) if isinstance(text, str) else None
    except (TypeError, ValueError):
        return None
    return json.dumps(value, sort_keys=True) if _valid_prediction(value) else None


def select(candidates):
    """Indices chosen by each selection rule; ties go to the earliest candidate."""
    def argmax(key):
        return max(range(len(candidates)), key=lambda i: (candidates[i][key], -i))

    votes = Counter(c["canonical"] for c in candidates if c["canonical"] is not None)
    if votes:
        top = max(votes.values())
        winners = {answer for answer, count in votes.items() if count == top}
        majority = next(i for i, c in enumerate(candidates) if c["canonical"] in winners)
    else:
        majority = 0
    return {"first_sample": 0, "majority_vote": majority,
            "reward_model": argmax("rm_score"), "oracle": argmax("rule_reward")}


# ---------------------------------------------------------------------------
# model helpers


def score_texts(model, tokenizer, prompt, completions, max_length):
    """Scalar RM score for each completion, one sequence at a time (no padding)."""
    import torch

    scores = []
    for completion in completions:
        encoded = tokenizer(prompt + completion + tokenizer.eos_token, return_tensors="pt")
        if encoded["input_ids"].shape[1] > max_length:
            raise ValueError("A scored sequence exceeds --max-length; nothing was truncated.")
        encoded = encoded.to(model.device)
        with torch.inference_mode():
            scores.append(float(model(**encoded).logits[0, 0]))
    return scores


def load_reward_model(args):
    from peft import PeftModel

    meta = load_adapter_metadata(args.reward_adapter, args)
    if meta["method"] != "reward":
        raise ValueError("--reward-adapter must point to a reward model adapter.")
    model, tokenizer = setup(args, head="reward")
    if meta["model_files_sha256"] != args.model_files_sha256:
        raise ValueError("Actual model files differ from the reward model's training base.")
    model = PeftModel.from_pretrained(model, args.reward_adapter, is_trainable=False)
    model.eval()
    return model, tokenizer, meta


# ---------------------------------------------------------------------------
# commands


def train_reward(args):
    if args.init_adapter:
        raise ValueError("--init-adapter is not supported for --method reward.")
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from trl import RewardConfig, RewardTrainer

    started_all = time.monotonic()
    splits = build_dataset()
    audit_dataset(splits)
    rows = splits["train"]
    output = fresh_directory(args.output)
    model, tokenizer = setup(args, head="reward")
    model = get_peft_model(model, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0, target_modules=["q_proj", "v_proj"],
        bias="none", task_type="SEQ_CLS"))
    before = {name: param.detach().cpu().clone()
              for name, param in model.named_parameters() if param.requires_grad}
    data = []
    for row in rows:
        prompt = render(tokenizer, row["prompt"], args.prompt_style)
        chosen = row["chosen"][0]["content"] + tokenizer.eos_token
        rejected = row["rejected"][0]["content"] + tokenizer.eos_token
        for answer in (chosen, rejected):
            if len(tokenizer(prompt + answer)["input_ids"]) > args.max_length:
                raise ValueError(f"{row['id']} exceeds --max-length; no targets were truncated.")
        data.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
    config = RewardConfig(
        output_dir=str(output / "checkpoints"), max_steps=args.steps, per_device_train_batch_size=1,
        learning_rate=args.learning_rate, seed=args.seed, data_seed=args.seed,
        use_cpu=args.device == "cpu", bf16=False, fp16=False, optim="adamw_torch",
        dataloader_num_workers=0, dataloader_pin_memory=False, report_to="none",
        push_to_hub=False, save_strategy="no", logging_steps=max(1, min(10, args.steps // 4)),
        gradient_checkpointing=False, max_length=args.max_length)
    trainer = RewardTrainer(model=model, args=config, train_dataset=Dataset.from_list(data),
                            processing_class=tokenizer)
    started = time.monotonic()
    result = trainer.train()
    seconds = time.monotonic() - started
    delta_squared = 0.0
    for name, param in model.named_parameters():
        if name in before:
            current = param.detach().cpu()
            if not torch.isfinite(current).all():
                raise ValueError("Training produced non-finite parameters.")
            delta_squared += (current - before[name]).double().square().sum().item()
    trainer.save_model(str(output / "adapter"))
    tokenizer.save_pretrained(output / "adapter")
    metadata = common_metadata(args) | {
        "method": "reward", "condition": "reward_model", "steps": args.steps,
        "max_length": args.max_length, "learning_rate": args.learning_rate,
        "train_count": len(rows), "training_data_sha256": fingerprint(rows),
        "train_seconds": seconds, "total_seconds": time.monotonic() - started_all,
        "completed_steps": trainer.state.global_step,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "adapter_delta_l2": delta_squared ** 0.5,
        "metrics": result.metrics, "log_history": trainer.state.log_history,
    }
    save(output / "training.json", metadata)
    print(json.dumps({k: metadata[k] for k in ("condition", "steps", "train_seconds", "adapter_delta_l2")}),
          flush=True)


def evaluate_reward(args):
    splits = build_dataset()
    audit_dataset(splits)
    rows = select_rows(splits[args.split], args.limit)
    output = fresh_directory(args.output)
    model, tokenizer, meta = load_reward_model(args)
    records = []
    started = time.monotonic()
    for row in rows:
        prompt = render(tokenizer, row["prompt"], args.prompt_style)
        chosen = row["chosen"][0]["content"]
        rejected = row["rejected"][0]["content"]
        variants = corruptions(chosen)
        for name, text in variants.items():
            if score_completion(text, row["expected"])["schema_valid"]:
                raise ValueError(f"Corruption '{name}' unexpectedly passes the schema check.")
        texts = [chosen, rejected, *variants.values()]
        scores = score_texts(model, tokenizer, prompt, texts, args.max_length)
        record = {"id": row["id"], "role": row["role"], "chosen_score": scores[0],
                  "rejected_score": scores[1],
                  "corruption_scores": dict(zip(variants, scores[2:]))}
        records.append(record)
        print(json.dumps(record), flush=True)
    pairwise = [r["chosen_score"] > r["rejected_score"] for r in records]
    probes = {name: sum(r["chosen_score"] > r["corruption_scores"][name] for r in records) / len(records)
              for name in records[0]["corruption_scores"]}
    summary = {
        "pairwise_accuracy": sum(pairwise) / len(pairwise),
        "mean_margin": sum(r["chosen_score"] - r["rejected_score"] for r in records) / len(records),
        "prefers_valid_over_malformed": probes,
    }
    metadata = common_metadata(args) | {
        "reward_adapter_training_sha256": fingerprint(meta), "split": args.split,
        "count": len(rows), "data_sha256": fingerprint(rows), **summary,
        "seconds": time.monotonic() - started,
    }
    save(output / "reward_scores.json", records)
    save(output / "reward_eval.json", metadata)
    print(json.dumps(summary, indent=2), flush=True)


def best_of_n(args):
    import torch
    from peft import PeftModel
    from transformers import set_seed

    splits = build_dataset()
    audit_dataset(splits)
    rows = select_rows(splits[args.split], args.limit)
    policy_meta = load_adapter_metadata(args.adapter, args) if args.adapter else None
    output = fresh_directory(args.output)
    reward_model, rm_tokenizer, rm_meta = load_reward_model(args)
    policy, tokenizer = setup(args)
    if policy_meta:
        if policy_meta["model_files_sha256"] != args.model_files_sha256:
            raise ValueError("Actual model files differ from the policy adapter's training base.")
        policy = PeftModel.from_pretrained(policy, args.adapter, is_trainable=False)
    policy.eval()
    started = time.monotonic()
    cases = []
    for index, row in enumerate(rows):
        prompt = render(tokenizer, row["prompt"], args.prompt_style)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(policy.device)
        set_seed(args.seed + index)
        with torch.inference_mode():
            generated = policy.generate(
                **inputs, do_sample=True, temperature=args.temperature, top_p=1.0, top_k=0,
                num_return_sequences=args.n, max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        texts = [tokenizer.decode(seq[inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                 for seq in generated]
        rm_scores = score_texts(reward_model, rm_tokenizer, prompt, texts, args.max_length)
        rule = graded_reward(texts, [row["expected"]] * len(texts))
        candidates = [{"completion": text, "rm_score": rm, "rule_reward": rr,
                       "canonical": canonical_answer(text),
                       "metrics": score_completion(text, row["expected"])}
                      for text, rm, rr in zip(texts, rm_scores, rule)]
        chosen = select(candidates)
        cases.append({"id": row["id"], "role": row["role"], "family": row["family"],
                      "expected": row["expected"], "candidates": candidates, "selected": chosen})
        print(json.dumps({"id": row["id"], "selected": chosen,
                          "rule_rewards": [round(r, 3) for r in rule]}), flush=True)
        save(output / "candidates.json", cases)
    results = {}
    for name in SELECTIONS:
        predictions = [{"id": c["id"], "expected": c["expected"],
                        "completion": c["candidates"][c["selected"][name]]["completion"],
                        "metrics": c["candidates"][c["selected"][name]]["metrics"]} for c in cases]
        results[name] = aggregate_results(rows, predictions)
        results[name]["selected_invalid_rate"] = sum(
            p["metrics"]["schema_valid"] == 0 for p in predictions) / len(predictions)
    pooled_rm = [cand["rm_score"] for c in cases for cand in c["candidates"]]
    pooled_rule = [cand["rule_reward"] for c in cases for cand in c["candidates"]]
    within = [spearman([x["rm_score"] for x in c["candidates"]], [x["rule_reward"] for x in c["candidates"]])
              for c in cases]
    within = [value for value in within if value is not None]
    agreement = {
        "spearman_pooled": spearman(pooled_rm, pooled_rule),
        "spearman_within_case_mean": sum(within) / len(within) if within else None,
        "cases_with_varying_rule_reward": len(within),
        "candidate_valid_rate": sum(cand["metrics"]["schema_valid"] for c in cases
                                    for cand in c["candidates"]) / len(pooled_rm),
    }
    metadata = common_metadata(args) | {
        "policy": policy_meta.get("condition", policy_meta["method"]) if policy_meta else "base",
        "policy_adapter_training_sha256": fingerprint(policy_meta) if policy_meta else None,
        "reward_adapter_training_sha256": fingerprint(rm_meta),
        "split": args.split, "count": len(rows), "data_sha256": fingerprint(rows),
        "sampling": {"n": args.n, "temperature": args.temperature, "top_p": 1.0,
                     "max_new_tokens": args.max_new_tokens, "seed_per_case": "seed + case index"},
        "selections": results, "agreement": agreement, "seconds": time.monotonic() - started,
    }
    save(output / "best_of_n.json", metadata)
    print(json.dumps({name: {"task_success": r["metrics"]["task_success"],
                             "pairs": r["context_pairs"]["both_correct"],
                             "invalid": r["selected_invalid_rate"]}
                      for name, r in results.items()} | {"agreement": agreement}, indent=2), flush=True)


if __name__ == "__main__":
    raise SystemExit("Use run.py: train --method reward, reward-eval, best-of-n.")
