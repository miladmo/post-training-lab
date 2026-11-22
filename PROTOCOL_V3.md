# Prespecified CPU pilot protocol, v3: SFT followed by GRPO

Written on 22 November 2026, before any v3 model run. This document fixes one
additional comparison. It contains no v3 results. The v2 protocol, report and
result bundle remain unchanged.

## Motivation

In v2, SFT solved 6 of 24 validation tasks and DPO 1 of 24. The optional GRPO
path started from the instruction model, sampled two completions per prompt and
used an exact-match reward. On the tiny random model every reward was zero, so
every group had zero reward variance and the adapter did not change.

Group relative policy optimization (GRPO; Shao et al., 2024) is the reinforcement learning method behind several current open recipes. These apply reinforcement learning with verifiable
rewards after a supervised stage instead of from scratch (Lambert et al., 2024;
DeepSeek-AI, 2025). v3 tests this sequence on the same narrow task. It asks
whether GRPO, started from the SFT adapter with a denser rule-based reward,
improves exact task success over SFT alone.

## Reward

The training reward has four equally weighted components, each in [0, 1]:

| Component | Definition |
|---|---|
| `format` | 1 if the completion is exactly one valid decision object, else 0 |
| `decision` | 1 if the decision is correct |
| `missing_f1` | F1 between predicted and expected missing fields (two empty sets count as 1) |
| `evidence_f1` | F1 between predicted and expected evidence IDs |

A completion that is not a valid decision object receives zero on all four, so
partial credit never rewards malformed output. Evaluation is unchanged and
remains exact: entire-task success requires all four exact checks.

## Prespecified conditions

| Setting | BASE | SFT | SFT+GRPO |
|---|---|---|---|
| Starting point | Instruction model | Instruction model, fresh LoRA | The SFT adapter from this protocol |
| Adaptation | None | As in v2: 96 steps, batch 1, lr 0.0002 | GRPO, 96 optimizer steps |
| Training cases | None | 48 training cases | Same 48 training prompts |
| LoRA | None | Rank 8, alpha 16, `q_proj` and `v_proj` | Continues the same adapter |
| Seed | 42 | 42 | 42 |
| Evaluation | 24 validation cases, greedy | Same | Same |

GRPO settings (TRL 1.13.0):

| Setting | Value | Reason |
|---|---|---|
| Generations per prompt | 8 | Enough samples for non-zero within-group variance |
| Prompts per optimizer step | 2 (16 completions) | CPU budget |
| Sampling temperature | 1.0 | Exploration during training only |
| Maximum completion length | 96 tokens | As in evaluation |
| Loss | `dapo`: token-level loss, clip range 0.2 / 0.28, truncated completions masked (Yu et al., 2025) | |
| Reward scaling | Within-group standard deviation | DAPO default |
| KL coefficient | 0, no reference model | DAPO, Dr. GRPO |
| Learning rate | 0.00005 | Lower than SFT for on-policy updates |

The alternative `dr_grpo` loss (Liu et al., 2025) is available through
`--grpo-loss dr_grpo` but is not part of this comparison.

## Outcomes and interpretation

Primary outcomes: entire-task success on the 24 validation cases and complete
context-reversal pairs (of 12), for SFT and SFT+GRPO. Secondary: the four exact
component metrics and role summaries.

Training diagnostics are recorded in `training.json` and reported regardless of
the outcome: mean reward and each component, the fraction of groups with zero
reward variance (these contribute no gradient) and the fraction of truncated
completions (masked from the loss).

With 24 cases and one seed, a difference of one or two tasks is not evidence of
an effect. Results are descriptive. The final test split stays unused. No
setting is tuned on the validation results; if the run fails or is changed,
this is reported as a deviation.

Because `compare` requires identical source files, BASE and SFT are evaluated
again with the v3 code rather than reusing v2 outputs. Agreement of the re-run
SFT with the v2 result (6/24) is a secondary reproducibility check.

## Commands

```powershell
$labPython = ".\.venv\Scripts\python.exe"
$pilotArgs = @("--model", "HuggingFaceTB/SmolLM2-135M-Instruct", "--revision", "12fd25f77366fa6b3b4b768ec3050bf629380bac", "--prompt-style", "two-shot")

& $labPython run.py evaluate @pilotArgs --split validation --output runs/v3-base-val24
& $labPython run.py train @pilotArgs --method sft --steps 96 --output runs/v3-sft-96
& $labPython run.py evaluate @pilotArgs --split validation --adapter runs/v3-sft-96/adapter --output runs/v3-sft-val24
& $labPython run.py train @pilotArgs --method grpo --init-adapter runs/v3-sft-96/adapter --steps 96 --learning-rate 5e-5 --output runs/v3-sft-grpo-96
& $labPython run.py evaluate @pilotArgs --split validation --adapter runs/v3-sft-grpo-96/adapter --output runs/v3-sft-grpo-val24
& $labPython run.py compare runs/v3-base-val24 runs/v3-sft-val24 runs/v3-sft-grpo-val24
```

Bash uses the same commands with `.venv/bin/python` and the arguments written
out instead of `@pilotArgs`.

## References

DeepSeek-AI. (2025). *DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning* (arXiv:2501.12948). https://arxiv.org/abs/2501.12948

Lambert, N., Morrison, J., Pyatkin, V., Huang, S., Ivison, H., Brahman, F., Miranda, L. J. V., et al. (2024). *Tülu 3: Pushing frontiers in open language model post-training* (arXiv:2411.15124). https://arxiv.org/abs/2411.15124

Liu, Z., Chen, C., Li, W., Qi, P., Pang, T., Du, C., Lee, W. S., & Lin, M. (2025). *Understanding R1-Zero-like training: A critical perspective* (arXiv:2503.20783). https://arxiv.org/abs/2503.20783

Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., et al. (2024). *DeepSeekMath: Pushing the limits of mathematical reasoning in open language models* (arXiv:2402.03300). https://arxiv.org/abs/2402.03300

Yu, Q., Zhang, Z., Zhu, R., Yuan, Y., Zuo, X., Yue, Y., et al. (2025). *DAPO: An open-source LLM reinforcement learning system at scale* (arXiv:2503.14476). https://arxiv.org/abs/2503.14476
