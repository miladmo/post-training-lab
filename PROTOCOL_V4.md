# Prespecified CPU pilot protocol, v4: reward model and best-of-N

Written on 22 November 2026, before any v4 model run. It contains no v4
results and leaves v2 and v3 unchanged.

## Question

Can a small learned reward model (RM), trained on the 48 rule-derived
preference pairs, stand in for the verifiable rule reward? v4 treats the RM
as a measurement instrument: it is first checked for reliability and
robustness, then used to select among sampled completions. Selection quality
is compared with selection rules that need no learned model.

## Reward model

| Setting | Value |
|---|---|
| Base | SmolLM2-135M-Instruct, same pinned revision |
| Head | Scalar sequence-classification head, trained |
| LoRA | Rank 8, alpha 16, `q_proj` and `v_proj`, fresh |
| Data | The 48 training preference pairs (both answers schema-valid) |
| Loss | Bradley-Terry pairwise loss (TRL `RewardTrainer`) |
| Steps, batch, learning rate | 96, 1 pair, 0.0001 |
| Seed | 42 |

## Reliability checks (24 validation cases)

1. **Pairwise accuracy:** share of validation pairs where the RM scores the
   correct answer above the rule-derived rejected answer, and the mean margin.
2. **Robustness probes:** share of cases where the correct answer scores above
   each of four deterministic malformed variants of itself: trailing text,
   Markdown code fence, truncation at half length, and a renamed key. All four
   fail the strict schema. The RM never saw malformed answers in training, so
   these probes test whether it rewards content while ignoring validity, a
   simple form of reward hacking risk.

## Best-of-N

The policy is the SFT adapter from v3 (`runs/v3-sft-96`). For each validation
case, eight completions are sampled (temperature 1.0, top-p 1.0, maximum 96
new tokens, seed 42 plus case index). On these identical candidates four rules
select one answer:

| Rule | Uses |
|---|---|
| `first_sample` | The first sample; equals a single sampled answer |
| `majority_vote` | Most frequent valid answer (self-consistency), first sample if none is valid |
| `reward_model` | Highest RM score |
| `oracle` | Highest graded rule reward; an upper bound that uses the answer key |

Ties go to the earliest candidate.

## Outcomes and interpretation

Primary: exact entire-task success (of 24) and complete context pairs (of 12)
per selection rule. The key comparison is `reward_model` against
`majority_vote`, the strongest rule without a learned component, and the gap
to `oracle`.

Secondary: share of invalid selected answers per rule; Spearman correlation
between RM score and graded rule reward, pooled over all candidates and as the
mean within cases whose rule rewards vary; share of valid candidates; the
reliability checks above.

With 24 cases and one seed the results are descriptive; differences of one or
two tasks are not evidence of an effect. The final test split stays unused.
No setting is tuned on validation results.

## Commands

Run after the v3 SFT adapter exists.

```powershell
$labPython = ".\.venv\Scripts\python.exe"
$pilotArgs = @("--model", "HuggingFaceTB/SmolLM2-135M-Instruct", "--revision", "12fd25f77366fa6b3b4b768ec3050bf629380bac", "--prompt-style", "two-shot")

& $labPython run.py train @pilotArgs --method reward --steps 96 --learning-rate 1e-4 --output runs/v4-rm-96
& $labPython run.py reward-eval @pilotArgs --reward-adapter runs/v4-rm-96/adapter --split validation --output runs/v4-rm-val24
& $labPython run.py best-of-n @pilotArgs --adapter runs/v3-sft-96/adapter --reward-adapter runs/v4-rm-96/adapter --n 8 --split validation --output runs/v4-bon8-val24
```

Outputs: `reward_eval.json` and `reward_scores.json` (per-case scores),
`best_of_n.json` (summary) and `candidates.json` (all 192 candidates with RM
score, rule reward and selections).

## References

Bradley, R. A., & Terry, M. E. (1952). Rank analysis of incomplete block designs: I. The method of paired comparisons. *Biometrika, 39*(3/4), 324–345. https://doi.org/10.2307/2334029

Gao, L., Schulman, J., & Hilton, J. (2023). Scaling laws for reward model overoptimization. In *Proceedings of the 40th International Conference on Machine Learning* (PMLR 202, pp. 10835–10866). https://proceedings.mlr.press/v202/gao23h.html

Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang, S., Chowdhery, A., & Zhou, D. (2023). Self-consistency improves chain of thought reasoning in language models. In *International Conference on Learning Representations*. https://arxiv.org/abs/2203.11171
