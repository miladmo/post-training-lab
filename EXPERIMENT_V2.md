# Local CPU pilot: prompting, SFT and DPO

Completed 15 September 2026 under the [prespecified v2 protocol](PROTOCOL_V2.md).
The earlier four-step run is retained in [LOCAL_EXPERIMENT.md](LOCAL_EXPERIMENT.md).

## Design

All three conditions used the same SmolLM2-135M-Instruct revision
`12fd25f77366fa6b3b4b768ec3050bf629380bac`, native chat template, two fixed
training-only demonstrations, labelled task text, and greedy decoding capped at
96 new tokens. SFT and DPO each started from the original instruction model and
trained independent rank-8 LoRA adapters for 96 steps: two passes over 48 training
cases, batch size one, initial learning rate 0.0002, seed 42, CPU FP32/four threads.

Evaluation covered all 24 validation cases from different scenario families:
eight each for HR, Sales and Science. Each of the 12 matched pairs holds evidence
fixed and changes requirements. A pair succeeds only when both answers are fully
correct. All cases and preference labels are synthetic and rule-derived; the
24 final test cases remained closed.

## Results

| Condition | Valid schema | Correct decision | Exact missing fields | Exact evidence | Entire task | Complete pairs |
|---|---:|---:|---:|---:|---:|---:|
| BASE | 20/24 | 11/24 | 1/24 | 12/24 | 0/24 | 0/12 |
| SFT | 24/24 | 19/24 | 12/24 | 14/24 | 6/24 | 0/12 |
| DPO | 20/24 | 9/24 | 1/24 | 12/24 | 1/24 | 0/12 |

| Role | BASE entire task | SFT entire task | DPO entire task |
|---|---:|---:|---:|
| HR | 0/8 | 3/8 | 0/8 |
| Sales | 0/8 | 2/8 | 0/8 |
| Science | 0/8 | 1/8 | 1/8 |

Relative to the shared prompting baseline, SFT solved +6 and DPO +1 additional complete validation cases in this single run.

Neither adapted model solved both sides of any complete context-reversal pair.
DPO matched the baseline's schema-valid count but returned fewer correct
decisions (9 versus 11). The isolated additional complete answer is therefore
not evidence of a consistent improvement across task requirements.

The format metric and semantic metrics must be read separately. Invalid schemas
receive zero semantic scores; semantic scores therefore cannot independently
identify the cause of failure in malformed responses. The joint task metric
requires all checks, so a correct decision with wrong evidence is unsuccessful.

Post-hoc inspection of SFT's outputs found that all six complete successes were
`ready` cases; none of the 12 `needs_review` cases fully succeeded. Two examples
show why schema validity and decision accuracy are insufficient:

- `hr_learning_session-4-2`: only the learning goal is evidenced. SFT correctly
  requests review but names only `facilitator` as missing, omitting the also
  required `attendance_list`. Its simpler paired case succeeds.
- `sales_renewal_brief-2-1` and `-2-2`: SFT correctly changes its decision when
  `usage_summary` becomes required and identifies that missing field. However,
  both outputs cite the renewal date (`E1`) instead of the required account
  contact (`E3`). Neither entire answer succeeds.

These examples diagnose validation failures after the run; they were not used
to select a checkpoint or change its configuration.

## CPU cost and saved artifacts

| Adaptation | Preparation incl. loading | Reference compute subset of preparation | Trainer time | Whole training process | Adapter change L2 |
|---|---:|---:|---:|---:|---:|
| SFT | 14.77 s | 0.00 s | 287.83 s | 304.42 s | 2.000666 |
| DPO | 155.91 s | 142.75 s | 534.17 s | 691.41 s | 2.279262 |

Reference computation is a measured subset of preparation, not an additional
stage to add again. Preparation includes dependency/model loading and trainer
initialization; pure preprocessing is not isolated. Trainer time excludes saving.
Whole-process timing also includes startup and saving.

| Evaluation process | BASE | SFT | DPO |
|---|---:|---:|---:|
| All 24 cases, including loading | 156.06 s | 84.80 s | 69.53 s |

Adapters were saved, then loaded in separate evaluation processes. The automated
comparison checked model-file hashes, source hashes, prompt protocol, cases,
targets, software versions and generation settings, and recomputed all reported
scores. Raw predictions, metadata, logs, the comparison and frozen protocol/code
hashes are retained under `runs/local-smol-v2-20260915/` and ignored by Git.
A compact [result bundle](results/cpu-pilot-v2.json) retains all 72 generated
answers, their targets and scores, configuration and provenance hashes without
local paths or model weights.

## Interpretation

This is a small local pipeline experiment on a rule-generated checklist task.
It does not establish professional enterprise quality, general method superiority,
or an autonomous reflection system. The three validation families, templated
cases and matched pairs are related observations; they are not 24 independent
samples of enterprise work. One model and one seed limit any broader inference.

The original four-step run and v2 differ in both prompting and training budget,
and use different-sized evaluation subsets. Their score difference cannot be
attributed to training alone. Within v2, all methods share the same prompt and
validation cases. Any future tuning requires a separately recorded experiment;
the final test should be used after design choices are frozen.

Twenty-eight software tests passed. Preflight checked 96 training alternatives
for prompt boundaries and supervised answer tokens; two-shot sequences fit in
356–392 tokens. Training/reference instrumentation uses the pinned TRL 1.13.0
API. No pretrained GRPO run was part of this protocol.
