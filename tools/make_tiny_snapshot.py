"""Build a tiny, randomly initialised model snapshot for offline smoke tests.

The snapshot mimics the Hugging Face cache layout expected by ``--snapshot``:
``<root>/models--fictional--tiny-chat/snapshots/<revision>/``. The model is a
two-layer Llama with random weights and a byte-level BPE tokenizer trained on
the fictional task texts. It cannot solve the task; it only exercises the
training and evaluation code paths without network access.

    python tools/make_tiny_snapshot.py snapshots-test
"""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MODEL = "fictional/tiny-chat"
REVISION = "0" * 40
CHAT_TEMPLATE = (
    "{% for message in messages %}<|im_start|>{{ message['role'] }}\n"
    "{{ message['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


def build(root):
    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    from tasks import build_dataset

    target = Path(root) / ("models--" + MODEL.replace("/", "--")) / "snapshots" / REVISION
    target.mkdir(parents=True, exist_ok=False)
    texts = []
    for rows in build_dataset().values():
        for row in rows:
            texts += [message["content"] for message in row["prompt"]]
            texts += [row["chosen"][0]["content"], row["rejected"][0]["content"]]
    specials = ["<|im_start|>", "<|im_end|>", "<|pad|>"]
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(texts, trainers.BpeTrainer(
        vocab_size=600, special_tokens=specials,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, bos_token="<|im_start|>",
                                   eos_token="<|im_end|>", pad_token="<|pad|>")
    fast.chat_template = CHAT_TEMPLATE
    fast.save_pretrained(target)
    torch.manual_seed(0)
    config = LlamaConfig(vocab_size=len(fast), hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                         max_position_embeddings=2048, bos_token_id=fast.bos_token_id,
                         eos_token_id=fast.eos_token_id, pad_token_id=fast.pad_token_id,
                         tie_word_embeddings=True)
    LlamaForCausalLM(config).save_pretrained(target, safe_serialization=True)
    return {"model": MODEL, "revision": REVISION, "snapshot": str(target)}


if __name__ == "__main__":
    print(json.dumps(build(sys.argv[1] if len(sys.argv) > 1 else "snapshots-test"), indent=2))
