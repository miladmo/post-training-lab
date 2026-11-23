# Post-training lab

A small CPU lab for adapting an instruction model with LoRA SFT and DPO.
The task: check a fictional work handoff against its supplied requirements and
return a JSON decision with missing fields and supporting evidence IDs.

## Experiment

The v2 pilot uses SmolLM2-135M-Instruct in FP32 with four CPU threads. SFT and DPO
each start independently from the same pinned instruction model: 96 optimizer
steps, batch size 1, initial learning rate 0.0002, seed 42, LoRA rank 8 / alpha 16
on `q_proj` and `v_proj`. Both methods and the base receive the same two fixed
training examples through the native chat template. Generation is greedy.

There are 96 synthetic cases: 48 training, 24 validation and 24 final test cases.
Each split covers HR, Sales and Science with disjoint scenario families.
Context-reversal pairs hold evidence constant while changing the requirements.
Preferences are derived from rules; all scenarios and data are fictional.

Evaluation checks output schema, decision, missing fields and evidence IDs.
Entire-task success requires all checks; pair success requires both tasks to
succeed. Results include all 24 validation cases, 12 pairs and role summaries.
The final test stays unused in this pilot. One seed and a shared task generator
do not establish professional quality or broad generalization.

## Results

Completed locally on CPU on 15 September 2026. These are validation
results from the fixed v2 comparison, not final-test performance.

| Condition | Entire tasks correct | Complete context pairs |
|---|---:|---:|
| BASE | 0/24 | 0/12 |
| SFT | 6/24 | 0/12 |
| DPO | 1/24 | 0/12 |

The [result bundle](results/cpu-pilot-v2.json) includes all 72 predictions,
scores, settings and hashes.

See the [v2 protocol](PROTOCOL_V2.md), [v2 report](EXPERIMENT_V2.md) and
[original four-step experiment](LOCAL_EXPERIMENT.md).

## Run locally

Use PowerShell in this directory with Python 3.11. Initial downloads require
internet access; no paid inference service is used. The suite contains 49 tests;
the two tiny-model smoke tests are skipped when torch, TRL and PEFT are not installed.

```powershell
py -3.11 -m venv .venv
$labPython = ".\.venv\Scripts\python.exe"
& $labPython -m pip install -r requirements.txt
& $labPython -m unittest discover -s tests -v
$pilotArgs = @("--model", "HuggingFaceTB/SmolLM2-135M-Instruct", "--revision", "12fd25f77366fa6b3b4b768ec3050bf629380bac", "--prompt-style", "two-shot")

& $labPython run.py prepare --output data
& $labPython run.py evaluate @pilotArgs --split validation --output runs/v2-base-val24
& $labPython run.py train @pilotArgs --method sft --steps 96 --output runs/v2-sft-96
& $labPython run.py evaluate @pilotArgs --split validation --adapter runs/v2-sft-96/adapter --output runs/v2-sft-val24
& $labPython run.py train @pilotArgs --method dpo --steps 96 --output runs/v2-dpo-96
& $labPython run.py evaluate @pilotArgs --split validation --adapter runs/v2-dpo-96/adapter --output runs/v2-dpo-val24
& $labPython run.py compare runs/v2-base-val24 runs/v2-sft-val24 runs/v2-dpo-val24
```

Use fresh output directories for another run. Logs record predictions, adapter
changes, timing, source/model/data hashes and versions. `compare` checks matching
cases, prompts and evaluation settings and recomputes the saved scores.
Generated data, adapters and run outputs are ignored by Git.

The CLI also offers the larger `Qwen/Qwen3-0.6B` as its pinned default; its
pretrained configuration has not been run locally. The commands above explicitly
select the smaller model used for the CPU pilot.

## SFT followed by GRPO (v3, prespecified)

The [v3 protocol](PROTOCOL_V3.md) adds the sequence used by current open
post-training recipes: supervised fine-tuning first, then reinforcement learning
with verifiable rewards. GRPO continues the SFT adapter (`--init-adapter`)
with eight sampled completions per prompt, the DAPO loss with clip-higher and
masking of truncated completions, and no reference model.

The training reward is graded: format validity, decision, and F1 scores for the
missing fields and evidence IDs, so near misses still carry signal. Evaluation
stays exact. Training logs record the reward per component, the share of groups
with zero reward variance (which produce no gradient) and the share of
truncated completions.

**Result (23 November 2026, [report](EXPERIMENT_V3_V4.md)).** SFT+GRPO solved
7/24 validation tasks versus 6/24 for SFT, but this is not an improvement:
GRPO shifted the policy to answering `ready` in 23 of 24 cases and decision
accuracy fell from 19/24 to 13/24. Training diagnostics show the optimization
worked (1 % zero-variance groups); the reward's balance across cases did not.
The re-run SFT reproduced all 24 v2 completions exactly.

## Reward model and best-of-N (v4, prespecified)

The [v4 protocol](PROTOCOL_V4.md) asks whether a small learned reward model
can stand in for the verifiable rule reward. The reward model is trained on
the 48 preference pairs (`train --method reward`) and checked like a
measurement instrument before use (`reward-eval`): pairwise accuracy on
held-out pairs and robustness probes with four malformed variants of correct
answers. `best-of-n` then samples eight answers per case from the SFT policy
and compares four selection rules on identical candidates: a single sample,
majority voting, the reward model and a rule-based oracle. It also reports how
well reward model scores agree with the rule reward.

**Result (23 November 2026, [report](EXPERIMENT_V3_V4.md)).** The reward model
ranked 17/24 held-out pairs correctly but scored malformed variants of correct
answers higher than the answers themselves in almost all cases. In best-of-8 it
selected invalid answers in 11/24 cases and solved 1/24, worse than a single
sample (3/24). Majority voting solved 7/24 and the first complete context pair;
the oracle shows correct candidates existed for 12/24 cases.

## KL anchor against decision collapse (v5, prespecified)

The [v5 protocol](PROTOCOL_V5.md) follows up on the v3 collapse. Both GRPO arms
merge the SFT adapter into the base and train a fresh adapter, so that the KL
reference is the SFT policy (`--init-mode merge`); they differ only in the KL
coefficient (`--beta 0` versus `--beta 0.04`). Training logs the share of
sampled `ready` answers and the KL value, and evaluations report all metrics
split by the expected decision. Collapse is defined in advance as at least 20
of 24 greedy answers with the same decision.

**Status: running** in the workflow `pilot-v5.yml`; results will be added here.

Research concept: Milad Morad. Implementation developed with AI assistance.

[MIT License](LICENSE). Model weights are downloaded separately and retain
their upstream licenses.

Sources: [SmolLM2 model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct),
[GRPO](https://arxiv.org/abs/2402.03300), [DAPO](https://arxiv.org/abs/2503.14476),
[Dr. GRPO](https://arxiv.org/abs/2503.20783), [Tülu 3](https://arxiv.org/abs/2411.15124),
[reward model overoptimization](https://proceedings.mlr.press/v202/gao23h.html),
[self-consistency](https://arxiv.org/abs/2203.11171),
[TRL 1.13.0](https://github.com/huggingface/trl/releases/tag/v1.13.0),
[PEFT LoRA](https://huggingface.co/docs/peft/developer_guides/lora).
