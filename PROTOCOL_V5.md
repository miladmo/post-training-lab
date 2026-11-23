# Prespecified CPU pilot protocol, v5: a KL anchor against decision collapse

Written on 23 November 2026, after the v3 and v4 results and before any v5
model run. It contains no v5 results.

## Motivation

In v3, GRPO started from the SFT adapter without a KL penalty (beta 0). The
greedy policy then answered `ready` in 23 of 24 validation cases, and decision
accuracy on `needs_review` cases fell from 10/12 to 1/12. A post-hoc look at
sampled candidates (see [EXPERIMENT_V3_V4.md](EXPERIMENT_V3_V4.md)) showed that
within each prompt the graded reward favours the correct decision in both
classes, which weakens reward asymmetry as the explanation.

v5 tests a different explanation: without an anchor, the policy drifts away
from the SFT policy along a direction the reward does not penalize strongly
enough. The original GRPO setup regularized the policy with a KL penalty
towards a reference policy, with coefficient 0.04 (Shao et al., 2024).

## Design

All arms start from one SFT adapter, retrained under the v3 settings (96 steps,
batch 1, lr 0.0002, seed 42).

| Arm | Initialization | KL coefficient beta | Reference policy |
|---|---|---|---|
| SFT | SFT adapter | not applicable | not applicable |
| GRPO, beta 0 | SFT merged into the base, fresh LoRA | 0 | none |
| GRPO, beta 0.04 | SFT merged into the base, fresh LoRA | 0.04 | the merged SFT model |

Merging is required so that the reference policy is the SFT model. With a
LoRA model, TRL computes reference log-probabilities with the adapter
disabled; without merging that reference would be the original instruction
model. Both GRPO arms use the same parametrization, so beta is the only
difference between them. BASE is evaluated again for completeness.

All other GRPO settings are those of v3: graded reward, 8 generations per
prompt, 2 prompts per step, temperature 1.0, maximum 96 completion tokens,
`dapo` loss with clip range 0.2 / 0.28, masking of truncated completions, 96
steps, lr 0.00005, seed 42, same 48 training prompts.

New diagnostics, logged in both arms: the share of sampled completions that
validly answer `ready` (a reward function with weight 0, logged but not
optimized) and the KL value reported by TRL. Evaluations now also report all
metrics split by the expected decision.

## Outcomes

Primary, on the 24 validation cases with greedy decoding:

1. Decision accuracy separately for `needs_review` and `ready` cases.
2. Collapse, defined in advance as at least 20 of 24 greedy answers with the
   same decision.

Secondary: entire-task success, complete context pairs, the remaining exact
metrics, the share of `ready` answers during training and the KL trajectory.

## Expectations stated before the run

If unanchored drift causes the collapse, the beta 0 arm collapses again under
the merged parametrization, and the beta 0.04 arm keeps decision accuracy on
`needs_review` cases closer to SFT (10/12) and does not collapse. If both arms
collapse, the KL anchor at this strength is not sufficient and drift alone does
not explain the result. If neither collapses, the collapse in v3 depended on
continuing the SFT adapter rather than training a fresh one, which would be
reported as a finding in its own right.

With 24 cases and one seed, results are descriptive. The final test split stays
unused. No setting is tuned on validation results.

## Execution

The runs execute in the workflow `.github/workflows/pilot-v5.yml` on
GitHub-hosted CPU runners: one job for BASE and SFT, then the two GRPO arms as
parallel jobs, then a job that compares all conditions and commits the JSON
outputs to the branch `results/pilot-v5`.

Equivalent local commands (PowerShell):

```powershell
$labPython = ".\.venv\Scripts\python.exe"
$pilotArgs = @("--model", "HuggingFaceTB/SmolLM2-135M-Instruct", "--revision", "12fd25f77366fa6b3b4b768ec3050bf629380bac", "--prompt-style", "two-shot")

& $labPython run.py evaluate @pilotArgs --split validation --output runs/v5-base-val24
& $labPython run.py train @pilotArgs --method sft --steps 96 --output runs/v5-sft-96
& $labPython run.py evaluate @pilotArgs --split validation --adapter runs/v5-sft-96/adapter --output runs/v5-sft-val24
& $labPython run.py train @pilotArgs --method grpo --init-adapter runs/v5-sft-96/adapter --init-mode merge --beta 0 --steps 96 --learning-rate 5e-5 --output runs/v5-grpo-b0-96
& $labPython run.py evaluate @pilotArgs --split validation --adapter runs/v5-grpo-b0-96/adapter --output runs/v5-grpo-b0-val24
& $labPython run.py train @pilotArgs --method grpo --init-adapter runs/v5-sft-96/adapter --init-mode merge --beta 0.04 --steps 96 --learning-rate 5e-5 --output runs/v5-grpo-b004-96
& $labPython run.py evaluate @pilotArgs --split validation --adapter runs/v5-grpo-b004-96/adapter --output runs/v5-grpo-b004-val24
& $labPython run.py compare runs/v5-base-val24 runs/v5-sft-val24 runs/v5-grpo-b0-val24 runs/v5-grpo-b004-val24
```

## Reference

Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., et al. (2024). *DeepSeekMath: Pushing the limits of mathematical reasoning in open language models* (arXiv:2402.03300). https://arxiv.org/abs/2402.03300
