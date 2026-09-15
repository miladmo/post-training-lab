# Local CPU experiment — 15 September 2026

This run checks whether a downloaded instruction model can be adapted, saved,
reloaded and evaluated locally. It is a small execution experiment, not a
benchmark of professional work quality.

## Protocol

- Model: [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct),
  revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`. Downloaded weights matched
  the published SHA-256 value. The smaller model was used for this laptop run;
  the lab's Qwen3 configuration has not been trained with pretrained weights here.
- Windows laptop, CPU only, four Torch threads, FP32, Python 3.11.9.
- Independent rank-8 LoRA SFT and DPO adapters, each starting from the same
  instruction model. Four optimizer steps, batch size one, learning rate 0.0002,
  seed 42. The training split contains 48 cases; four steps do not cover all of it.
- Six fixed validation cases: one context-reversal pair each for HR, Sales and
  Science. Scenario families are disjoint from training. Greedy decoding,
  maximum 96 new tokens, identical prompts and model files across conditions.
- Outputs are scored as generated, without repairing JSON. All training labels
  and cases are synthetic and rule-derived. The final 24-case test was not run.

## Observations

| Condition | Valid JSON schema | Correct decision | Exact missing fields | Exact evidence | Entire task correct |
|---|---:|---:|---:|---:|---:|
| BASE | 0/6 | 0/6 | 0/6 | 0/6 | 0/6 |
| SFT | 0/6 | 0/6 | 0/6 | 0/6 | 0/6 |
| DPO | 0/6 | 0/6 | 0/6 | 0/6 | 0/6 |

All three conditions failed the required output schema. The observed failure
was input-like task JSON instead of the requested decision object. Four updates
did not resolve this failure; no task-quality improvement was observed.

| Training | Elapsed trainer time | Adapter parameter change (L2) |
|---|---:|---:|
| SFT | 8.33 s | 0.219406 |
| DPO | 9.72 s | 0.201870 |

Trainer time excludes model loading, DPO reference-log-probability precomputation,
and saving. Including preparation and saving, the observed SFT process took
18.69 seconds and DPO took 76.27 seconds. Both adapters were saved and
loaded in separate evaluation processes. The comparison verified case IDs,
prompts, targets, decoding settings, model-file hashes and software versions,
and recomputed scores from the stored completions.

Six cases, four training steps and one seed cannot establish that either method
improves general performance or is superior. Subsequent experiments need a
prespecified larger training budget, a stronger prompting baseline, repeated
seeds, full validation, and a final test after choices have been frozen.

## Reproduction and checks

Use the README's model and revision with `--prompt-style original`, `--steps 4`
for each training command and `--limit 6` for each evaluation command. The
current README's two-shot, 96-step commands instead reproduce the v2 protocol.
Local raw logs, training metadata, predictions and comparisons are in
`runs/local-smol-20260915/`; generated run files and adapters are ignored by Git.

Fourteen standard-library tests passed. A separate, randomly initialized
85,312-parameter Qwen3 model exercised SFT, DPO and GRPO for one step each.
SFT/DPO changed parameters and passed adapter-reload checks. GRPO produced
equal zero rewards and no parameter change. That random-model test checks
implementation paths only; no pretrained GRPO experiment was performed.
