# Prespecified CPU pilot protocol, v2

Written on 15 September 2026, before the v2 model runs. This document fixes a
small comparison of prompting, supervised adaptation and preference adaptation.
It does not contain v2 results. The earlier run remains documented unchanged in
`LOCAL_EXPERIMENT.md`.

## Motivation and scope

The original four-step experiment produced input-like task JSON instead of the
required decision object in all six validation cases, for all three conditions.
These were genuine generated schema failures. This pilot tests whether a clearer
shared input format and a larger, fixed training budget make the task executable.
It does not isolate which change explains a difference from the earlier run.

The comparison within v2 holds the task, demonstrations, model, decoding and
scoring constant. SFT receives rule-derived correct completions; DPO receives
rule-derived preference pairs. It tests these adaptation procedures on a narrow
context-sensitive task, not an automatic reflection or recursive optimization
system. No human feedback, AsX material, dissertation data or company data is
used.

## Data and prompt

The existing 96 fictional cases remain unchanged: 48 training, 24 validation and
24 final test cases. HR, Sales and Science appear in every split. Whole scenario
families are disjoint between splits; all cases still share the same generator
and checklist rule. Each context-reversal pair holds evidence constant while
changing the required checklist. A split therefore contains 24, 12 or 12 pairs,
respectively.

Use the model's native chat template with a system instruction, two fixed
user/assistant demonstrations, and clearly labelled text for the current task.
The demonstrations are the training cases `hr_recruiting_kickoff-1-1` and
`hr_recruiting_kickoff-1-2`: the same evidence supports a `ready` decision under
one checklist and a `needs_review` decision under the other. Use these exact
training-only demonstrations in every condition; do not select demonstrations
from validation or test data. The current case's answer is never supplied as
input during evaluation.

The output schema remains exactly `decision`, `missing` and `evidence_ids`.
Rendering and completion-mask preflight checks cover all 48 training cases
(96 chosen/rejected alternatives) and have passed. These are implementation checks,
not model-performance results. Both answers in each DPO preference pair already
have a valid schema, so the preference signal distinguishes task correctness
rather than directly contrasting valid and malformed output.

## Prespecified conditions

| Setting | BASE | SFT | DPO |
|---|---|---|---|
| Starting model | Original instruction model | Same original instruction model | Same original instruction model |
| Adaptation | None | Fresh LoRA, supervised correct completions | Fresh LoRA, rule-derived preference pairs |
| Optimizer steps | 0 | 96 | 96 |
| Training cases | None | Same 48 training cases | Same 48 training cases |
| LoRA | None | Rank 8, alpha 16, `q_proj` and `v_proj` | Rank 8, alpha 16, `q_proj` and `v_proj` |
| Initial learning rate | — | 0.0002 | 0.0002 |
| Training batch size | — | 1 | 1 |
| Random seed | 42 | 42 | 42 |
| Evaluation | All 24 validation cases | Same 24 validation cases | Same 24 validation cases |

Model: [HuggingFaceTB/SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct),
revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`.
All conditions use the same model files on a Windows CPU, FP32 and four Torch
threads. SFT and DPO start independently; DPO does not continue the SFT adapter.
The frozen original model supplies DPO reference log probabilities.

The training length limit is 512 tokens. Check that complete prompts and target
completions fit; do not silently truncate required evidence or targets. Evaluation
uses greedy decoding with at most 96 new tokens and identical prompts across
conditions. The 96-step budget is fixed, not selected from a training sweep.

## Evaluation and reporting

Score completions as generated, without JSON repair. Report counts and rates
overall and separately for HR, Sales and Science for:

- valid output schema;
- correct decision;
- exact missing-field set;
- exact evidence-ID set;
- entire task correct, requiring all four checks.

The existing scorer gates semantic scores on schema validity. Report schema
failures explicitly; zeros under this scoring rule do not independently diagnose
every underlying reasoning decision. For the 12 validation reversal pairs,
report pair success only when both tasks in a pair achieve entire-task success.
Each role contributes eight cases and four pairs.

Retain predictions, case IDs, prompt/data hashes, model revision and file hashes,
settings, package versions and adapter metadata. Record preprocessing, DPO
reference precomputation, training and evaluation time separately. If a total
process time is reported, label its included stages; do not present DPO training
time as the total adaptation cost.

Report all three conditions and any positive, zero or negative differences.
Do not change prompts, examples, training budgets, hyperparameters or scoring
after inspecting this run's validation outputs. A necessary implementation fix
must be documented as a protocol deviation before a rerun, with the affected
run retained. Do not select and report only the best condition.

## Interpretation and stopping point

The 24 final test cases remain closed throughout this pilot. Inspecting validation
can motivate a separately documented future experiment, but it does not justify
changing this run retroactively or claiming final test performance.

One small model, one seed and synthetic cases from a shared generator can show
whether this implementation learns the specified task under these conditions.
They cannot establish general enterprise quality, superiority over the state of
the art, or effective autonomous self-reflection. The result may be useful,
inconclusive or negative; each is a reportable pilot outcome. A frozen code
snapshot and experiment report will accompany the completed runs.
