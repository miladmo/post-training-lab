# CPU pilot v3 and v4: SFT followed by GRPO, reward model and best-of-N

Completed 23 November 2026 under the prespecified [v3](PROTOCOL_V3.md) and
[v4](PROTOCOL_V4.md) protocols, in one GitHub Actions run
([run 45158013091](https://github.com/miladmo/post-training-lab/actions/runs/45158013091),
source commit `c95a341`). All outputs are in
[results/pilot-v3-v4/run-45158013091](results/pilot-v3-v4/run-45158013091).

## Execution

The protocols fixed all model, data and training settings but not the machine.
All runs used a GitHub-hosted Ubuntu runner, CPU only, FP32, four threads, seed
42, with the pinned packages (torch 2.11.0+cpu, transformers 5.17.0, TRL 1.13.0,
PEFT 0.20.0; full list in `environment.txt`). Every command was executed exactly
as listed in the protocols. No setting was changed after seeing results.

Training time: SFT 222 s, GRPO 4,808 s (96 steps of 16 completions), reward
model 333 s. Best-of-8 sampling and scoring took 469 s.

**Reproducibility check.** BASE reproduced the v2 metrics, and the re-trained
SFT adapter produced all 24 validation completions character for character
identical to the stored v2 SFT run (`results/cpu-pilot-v2.json`).

## v3 results: does GRPO after SFT help?

| Condition | Valid schema | Correct decision | Exact missing fields | Exact evidence | Entire task | Complete pairs |
|---|---:|---:|---:|---:|---:|---:|
| BASE | 20/24 | 11/24 | 1/24 | 12/24 | 0/24 | 0/12 |
| SFT | 24/24 | 19/24 | 12/24 | 14/24 | 6/24 | 0/12 |
| SFT+GRPO | 24/24 | 13/24 | 12/24 | 14/24 | 7/24 | 0/12 |

Split by the expected decision:

| Condition | Answered `ready` | Correct on `ready` cases | Correct on `needs_review` cases | Entire task `ready` | Entire task `needs_review` |
|---|---:|---:|---:|---:|---:|
| SFT | 11/24 | 9/12 | 10/12 | 6/12 | 0/12 |
| SFT+GRPO | 23/24 | 12/12 | 1/12 | 7/12 | 0/12 |

**Reading.** The one additional fully solved task is not an improvement. GRPO
moved the policy towards answering `ready` almost always (23 of 24 greedy
answers). Decision accuracy fell from 19 to 13 of 24, which is about the level
of a constant answer on this balanced split. The extra success is a `ready`
case that becomes trivially correct under that shift. No condition solved a
case that requires `needs_review` completely, and no condition solved any
context-reversal pair.

**Training diagnostics.** The optimization itself worked: only 1 % of groups
had zero reward variance, 0.8 % of completions were truncated, and the adapter
moved (L2 change 0.36 from the SFT starting point). The mean training reward
rose only slightly and noisily, from 0.57 at step 10 to 0.64 at step 96; the
decision component stayed near 0.46 on average. The graded reward did provide
signal, but the signal did not translate into better decisions.

A plausible contributor, not tested here, is an asymmetry in the reward: for a
`ready` case the correct answer has an empty `missing` list and earns full
credit easily, whereas a correct `needs_review` answer must name the exact
missing fields to earn the same. A follow-up would need to test this, for
example with class-balanced reward weights.

## v4 results: can a learned reward model replace the rule reward?

**Reliability checks** (24 validation cases):

| Check | Result |
|---|---:|
| Pairwise accuracy, correct answer above rule-derived rejected answer | 17/24 (71 %) |
| Mean score margin | 0.27 |
| Correct answer above the same answer with trailing text | 0/24 |
| Correct answer above the same answer in a Markdown code fence | 0/24 |
| Correct answer above the same answer truncated at half length | 1/24 |
| Correct answer above the same answer with a renamed key | 12/24 |

The reward model separated the held-out preference pairs moderately well but
ranked malformed variants of correct answers higher than the answers
themselves in almost every case. It never saw malformed answers in training,
so it learned nothing about validity.

**Best-of-8** with the SFT policy (192 candidates, temperature 1.0):

| Selection rule | Entire task | Complete pairs | Correct decision | Invalid selected |
|---|---:|---:|---:|---:|
| First sample | 3/24 | 0/12 | 8/24 | 7/24 |
| Majority vote | 7/24 | 1/12 | 12/24 | 0/24 |
| Reward model | 1/24 | 0/12 | 3/24 | 11/24 |
| Oracle (uses answer key) | 12/24 | 2/12 | 24/24 | 0/24 |

Agreement between reward model scores and the graded rule reward was absent:
Spearman correlation −0.001 over all candidates and 0.09 on average within
cases. 145 of 192 candidates (76 %) were valid.

**Reading.** Selecting by the reward model was worse than taking a single
sample, mainly because it picked invalid answers in 11 of 24 cases, exactly the
failure the robustness probes predicted. Majority voting, which needs no
learned model, matched greedy SFT (7 versus 6 of 24) and solved the first
complete context pair in this project. The oracle shows that the eight
candidates contained a fully correct answer for 12 of 24 cases, so selection,
not generation alone, limits performance here.

## Conclusions for this pilot

1. Reinforcement learning with a verifiable reward is only as good as the
   reward's balance across cases. Here it produced a policy that looks slightly
   better on the headline metric while being worse on the decision itself.
   Component and per-class metrics were necessary to see this.
2. A reward model with acceptable pairwise accuracy can still be unusable for
   selection. Robustness probes run before deployment predicted its failure in
   best-of-N.
3. Simple test-time aggregation (majority voting) was the most reliable
   improvement over a single sample in this setting.

## Limitations

One synthetic task family, 24 validation cases, one seed and a 135M-parameter
model on CPU. Differences of one or two tasks are within noise. The final test
split remains unused. Results do not generalize beyond this setup without
further runs.
