# CPU pilot v5: a KL anchor against decision collapse

Completed 23 November 2026 under the prespecified [v5 protocol](PROTOCOL_V5.md)
in one GitHub Actions run
([run 45178622396](https://github.com/miladmo/post-training-lab/actions/runs/45178622396),
source commit `b58848d`). All outputs are in
[results/pilot-v5/run-45178622396](results/pilot-v5/run-45178622396).

## Execution

GitHub-hosted Ubuntu runners, CPU only, FP32, seed 42, pinned packages as in
v3 and v4. The SFT job ran first; the two GRPO arms then ran in parallel jobs on
separate runners. All commands matched the protocol. Training time: GRPO with
beta 0 2,370 s, with beta 0.04 2,880 s.

**Reproducibility.** BASE and SFT produced all 24 validation completions
character for character identical to v3, and therefore to v2 as well.

## Results

| Condition | Answers `ready` | Correct decision, `needs_review` cases | Correct decision, `ready` cases | Entire task | Complete pairs | Collapse (≥ 20/24 same decision) |
|---|---:|---:|---:|---:|---:|---|
| BASE | 0/24 (20 `needs_review`, 4 invalid) | 11/12 | 0/12 | 0/24 | 0/12 | yes, towards `needs_review` |
| SFT | 11/24 | 10/12 | 9/12 | 6/24 | 0/12 | no |
| GRPO, beta 0 | 23/24 | 1/12 | 12/12 | 7/24 | 0/12 | yes |
| GRPO, beta 0.04 | 22/24 | 2/12 | 12/12 | 7/24 | 0/12 | yes |

Remaining exact metrics: schema validity 24/24 for SFT and both GRPO arms;
exact missing fields 12/24 for all three; exact evidence 14/24 for SFT and
beta 0, 15/24 for beta 0.04. Every fully solved task in every condition is a
`ready` case.

Training diagnostics:

| | GRPO, beta 0 | GRPO, beta 0.04 |
|---|---:|---:|
| Share of sampled `ready` answers, step 10 → step 96 | 0.54 → 0.66 | 0.54 → 0.61 |
| KL to the SFT reference, step 10 → maximum | not computed | 0.0002 → 0.0039 |
| Mean decision reward component during training | 0.46 | 0.45 |
| Groups with zero reward variance | 0.5 % | 0.5 % |
| Truncated completions | 1.1 % | 1.1 % |
| Adapter change (L2) | 0.57 | 0.54 |

## Reading against the prespecified expectations

Both arms collapsed. The protocol stated in advance that this outcome means the
KL anchor at this strength is not sufficient and unanchored drift alone does
not explain the collapse. The beta 0 arm also reproduced the v3 outcome under
the different parametrization: its 24 greedy completions are identical to those
of the v3 run, although the adapter was trained from scratch on the merged SFT
model and moved more (L2 0.57 versus 0.36).

## Post-hoc interpretation (not prespecified)

Three observations fit together. First, the KL term stayed very small (at most
0.004), so with beta 0.04 it contributed almost nothing to the objective: the
policy did not move far from the SFT policy in distribution. Second, under
sampling the decision was barely better than chance throughout training (mean
decision component 0.45 to 0.46), and the sampled share of `ready` answers moved
only from 0.54 to about 0.6. Third, greedy decoding nevertheless flipped almost
every case to `ready`.

This pattern suggests that the SFT policy's decision is close to a coin flip
for many prompts, largely independent of the case content, and that GRPO shifts
a global preference slightly towards `ready`. Under greedy decoding, a small
global shift is enough to flip the most likely decision almost everywhere. If
this is right, the limiting factor is the policy's ability to discriminate the
cases, not regularization, and greedy accuracy alone exaggerates how much the
policy changed. Testing this would require measuring sampled decision rates per
case before and after GRPO, and interventions that improve discrimination, for
example more or contrastive supervised data or a larger model.

## Limitations

One synthetic task family, 24 validation cases, one seed, a 135M-parameter
model and one KL strength. A larger beta was not tested; the small KL values
suggest that a much larger coefficient would be needed to have any effect. The
final test split remains unused.
