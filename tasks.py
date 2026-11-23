"""Synthetic checklist tasks and exact-match metrics for post-training examples."""

from collections import Counter
import hashlib
import json
from pathlib import Path


SPLITS = ("train", "validation", "test")
SYSTEM_PROMPT = (
    "Assess this fictional handoff using only its checklist and evidence. "
    "Return exactly one JSON object with decision, missing, evidence_ids. "
    "Decision is ready iff every required field has evidence; otherwise needs_review. "
    "Missing lists required fields without evidence. Evidence_ids lists only IDs "
    "supporting required fields. Sort both arrays lexicographically. "
    "No extra keys or text."
)

# Split whole scenario families; all descriptions and values are invented.
FAMILIES = (
    ("train", "HR", "hr_recruiting_kickoff", "Recruiting kickoff", (
        ("role_brief", "The role brief describes a product coordinator."),
        ("interview_owner", "Robin owns the interview schedule."),
        ("hiring_window", "The proposed hiring window is October."))),
    ("train", "HR", "hr_onboarding_schedule", "Onboarding schedule", (
        ("start_date", "The new colleague starts on 4 May."),
        ("buddy", "Morgan is the onboarding buddy."),
        ("access_ticket", "Workspace access ticket A17 is recorded."))),
    ("validation", "HR", "hr_learning_session", "Learning session", (
        ("learning_goal", "The session covers spreadsheet review."),
        ("facilitator", "Casey facilitates the session."),
        ("attendance_list", "The attendance list contains six volunteers."))),
    ("test", "HR", "hr_internal_transfer", "Internal transfer", (
        ("destination_team", "The destination team is service planning."),
        ("transition_date", "The proposed transition date is 7 June."),
        ("handoff_owner", "Taylor owns the transition handoff."))),
    ("train", "Sales", "sales_demo_prep", "Customer demo preparation", (
        ("customer_goal", "The customer wants faster inventory reporting."),
        ("demo_owner", "Avery will run the demonstration."),
        ("meeting_slot", "The meeting is booked for Thursday at 10."))),
    ("train", "Sales", "sales_quote_handoff", "Quotation handoff", (
        ("requested_scope", "The request covers twelve workspace seats."),
        ("currency", "The requested quotation currency is EUR."),
        ("quote_owner", "Sam prepares the quotation."))),
    ("validation", "Sales", "sales_renewal_brief", "Renewal brief", (
        ("renewal_date", "The fictional contract renews on 12 July."),
        ("usage_summary", "The account uses eight workspace seats."),
        ("account_contact", "Jordan is the account contact."))),
    ("test", "Sales", "sales_account_plan", "Account planning handoff", (
        ("account_goal", "The account plans to add a reporting team."),
        ("next_action", "The next action is a scope review call."),
        ("action_owner", "Quinn owns the review call."))),
    ("train", "Science", "science_sample_handoff", "Sample record handoff", (
        ("sample_id", "The fictional sample identifier is SAMPLE-17."),
        ("storage_record", "The storage record points to cabinet B."),
        ("receiving_team", "The methods team receives the record."))),
    ("train", "Science", "science_instrument_run", "Instrument run record", (
        ("instrument_id", "The fictional instrument is UNIT-4."),
        ("run_owner", "Lee owns the recorded instrument run."),
        ("calibration_record", "Calibration record CAL-21 is attached."))),
    ("validation", "Science", "science_analysis_review", "Analysis review handoff", (
        ("dataset_version", "The fictional dataset version is set-3."),
        ("analysis_script", "The analysis script is summary.R."),
        ("reviewer", "Alex is assigned to review the analysis."))),
    ("test", "Science", "science_protocol_handoff", "Protocol document handoff", (
        ("protocol_version", "The fictional protocol version is draft-2."),
        ("document_owner", "Drew owns the protocol document."),
        ("review_window", "The document review window closes on Friday."))),
)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key.")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"Non-JSON numeric constant: {value}")


def _loads(text):
    return json.loads(text, object_pairs_hook=_unique_object,
                      parse_constant=_reject_constant)


def _string_set(value):
    return (isinstance(value, list)
            and all(isinstance(item, str) and item.strip() for item in value)
            and len(value) == len(set(value)))


def _valid_prediction(value):
    return (isinstance(value, dict)
            and set(value) == {"decision", "missing", "evidence_ids"}
            and value["decision"] in ("ready", "needs_review")
            and all(_string_set(value[key]) and value[key] == sorted(value[key])
                    for key in ("missing", "evidence_ids")))


def _derive_expected(context):
    if (not isinstance(context, dict)
            or set(context) != {"task", "required_fields", "evidence"}
            or not isinstance(context["task"], str) or not context["task"].strip()
            or not _string_set(context["required_fields"])
            or not context["required_fields"]
            or not isinstance(context["evidence"], list)):
        raise ValueError("Invalid task context.")
    evidence = {}
    ids = set()
    for fact in context["evidence"]:
        if (not isinstance(fact, dict) or set(fact) != {"id", "field", "fact"}
                or any(not isinstance(v, str) or not v.strip() for v in fact.values())
                or fact["field"] in evidence or fact["id"] in ids):
            raise ValueError("Evidence fields and IDs must be unique nonempty strings.")
        evidence[fact["field"]] = fact["id"]
        ids.add(fact["id"])
    required = set(context["required_fields"])
    missing = sorted(required - evidence.keys())
    return {"decision": "needs_review" if missing else "ready",
            "missing": missing,
            "evidence_ids": sorted(evidence[field] for field in required & evidence.keys())}


def score_completion(text, expected):
    """Score exact checklist outputs, not professional quality or human preference."""
    if not _valid_prediction(expected):
        raise ValueError("Invalid expected output schema.")
    metrics = dict.fromkeys(("schema_valid", "decision_correct", "missing_fields_exact",
                             "evidence_exact", "task_success"), 0)
    try:
        prediction = _loads(text) if isinstance(text, str) else None
    except (TypeError, ValueError):
        return metrics
    if not _valid_prediction(prediction):
        return metrics
    metrics.update(schema_valid=1,
                   decision_correct=int(prediction["decision"] == expected["decision"]),
                   missing_fields_exact=int(prediction["missing"] == expected["missing"]),
                   evidence_exact=int(prediction["evidence_ids"] == expected["evidence_ids"]))
    metrics["task_success"] = int(all(metrics[key] for key in metrics if key != "task_success"))
    return metrics


def _completion_text(completion):
    """Return generated text from a string or a single assistant message."""
    if isinstance(completion, str):
        return completion
    if (isinstance(completion, list) and len(completion) == 1
            and isinstance(completion[0], dict) and completion[0].get("role") == "assistant"):
        return completion[0].get("content")
    return None


def reward(completions, expected, **kwargs):
    """TRL-compatible bounded reward: mean of four exact, rule-based checks."""
    if len(completions) != len(expected):
        raise ValueError("Completions and expected outputs must have equal lengths.")
    rewards = []
    for completion, target in zip(completions, expected):
        metrics = score_completion(_completion_text(completion), target)
        rewards.append(sum(metrics[key] for key in (
            "schema_valid", "decision_correct", "missing_fields_exact", "evidence_exact")) / 4)
    return rewards


REWARD_COMPONENTS = ("format", "decision", "missing_f1", "evidence_f1")


def _set_f1(predicted, expected):
    """F1 of two sets; two empty sets agree perfectly."""
    predicted, expected = set(predicted), set(expected)
    if not predicted and not expected:
        return 1.0
    overlap = len(predicted & expected)
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(predicted), overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def graded_components(text, expected):
    """Verifiable reward components with partial credit for set-valued fields.

    A completion that is not exactly one valid decision object earns zero on
    every component, so partial credit never rewards malformed output. The
    evaluation metrics in ``score_completion`` stay exact; this function only
    shapes the training signal.
    """
    if not _valid_prediction(expected):
        raise ValueError("Invalid expected output schema.")
    zero = dict.fromkeys(REWARD_COMPONENTS, 0.0)
    try:
        prediction = _loads(text) if isinstance(text, str) else None
    except (TypeError, ValueError):
        return zero
    if not _valid_prediction(prediction):
        return zero
    return {"format": 1.0,
            "decision": float(prediction["decision"] == expected["decision"]),
            "missing_f1": _set_f1(prediction["missing"], expected["missing"]),
            "evidence_f1": _set_f1(prediction["evidence_ids"], expected["evidence_ids"])}


def _component_reward(name):
    def component(completions, expected, **kwargs):
        if len(completions) != len(expected):
            raise ValueError("Completions and expected outputs must have equal lengths.")
        return [graded_components(_completion_text(completion), target)[name]
                for completion, target in zip(completions, expected)]
    component.__name__ = f"reward_{name}"
    component.__doc__ = f"TRL reward function for the graded component '{name}'."
    return component


# One reward function per component so TRL logs each mean separately.
GRADED_REWARDS = tuple(_component_reward(name) for name in REWARD_COMPONENTS)


def reward_answers_ready(completions, expected, **kwargs):
    """Diagnostic only (weight 0 in training): 1 if the completion validly answers 'ready'."""
    out = []
    for completion in completions:
        try:
            text = _completion_text(completion)
            prediction = _loads(text) if isinstance(text, str) else None
        except (TypeError, ValueError):
            prediction = None
        out.append(float(_valid_prediction(prediction) and prediction["decision"] == "ready"))
    return out


def graded_reward(completions, expected, **kwargs):
    """Equal-weight mean of the four graded components, bounded in [0, 1]."""
    parts = [function(completions, expected) for function in GRADED_REWARDS]
    return [sum(values) / len(values) for values in zip(*parts)]


def build_dataset():
    splits = {name: [] for name in SPLITS}
    # Each pair holds evidence fixed and changes the required checklist.
    # Some supplied evidence is irrelevant to the shorter checklist.
    pairs = (((0, 1), (0,), (0, 2)),
             ((0, 2), (2,), (1, 2)),
             ((1, 2), (1, 2), (0, 1, 2)),
             ((0,), (0,), (0, 1, 2)))
    for split, role, family, task, fields in FAMILIES:
        for pair, (supplied, ready_fields, review_fields) in enumerate(pairs, start=1):
            evidence = [{"id": f"E{index + 1}", "field": fields[index][0],
                         "fact": fields[index][1]} for index in supplied]
            for variant, required in enumerate((ready_fields, review_fields), start=1):
                context = {"task": task, "required_fields": [fields[index][0] for index in required],
                           "evidence": evidence}
                expected = _derive_expected(context)
                if expected["decision"] == "needs_review":
                    rejected = {**expected, "decision": "ready", "missing": []}
                else:
                    rejected = {"decision": "needs_review", "missing": [fields[required[0]][0]],
                                "evidence_ids": expected["evidence_ids"][1:]}
                splits[split].append({
                    "id": f"{family}-{pair}-{variant}", "family": family, "role": role,
                    "prompt": [{"role": "system", "content": SYSTEM_PROMPT},
                               {"role": "user", "content": json.dumps(context, separators=(",", ":"))}],
                    "chosen": [{"role": "assistant", "content": json.dumps(expected, separators=(",", ":"))}],
                    "rejected": [{"role": "assistant", "content": json.dumps(rejected, separators=(",", ":"))}],
                    "expected": expected, "provenance": "synthetic-rule-derived",
                })
    audit_dataset(splits)
    return splits


def audit_dataset(splits):
    if not isinstance(splits, dict) or set(splits) != set(SPLITS):
        raise ValueError("Expected train, validation and test splits.")
    seen_ids, seen_prompts, family_splits, family_roles = set(), set(), {}, {}
    summary = {}
    for split in SPLITS:
        rows = splits[split]
        if not isinstance(rows, list) or not rows:
            raise ValueError("Splits must contain nonempty row lists.")
        reversals = {}
        for row in rows:
            if set(row) != {"id", "family", "role", "prompt", "chosen", "rejected", "expected", "provenance"}:
                raise ValueError("Invalid dataset row fields.")
            identifier, family, role = row["id"], row["family"], row["role"]
            if not all(isinstance(v, str) and v.strip() for v in (identifier, family, role)):
                raise ValueError("Invalid row identity.")
            if identifier in seen_ids or family_splits.get(family, split) != split:
                raise ValueError("Row or scenario family overlaps dataset splits.")
            if family_roles.get(family, role) != role or role not in {"HR", "Sales", "Science"}:
                raise ValueError("Invalid or inconsistent family role.")
            seen_ids.add(identifier)
            family_splits[family], family_roles[family] = split, role
            if row["provenance"] != "synthetic-rule-derived":
                raise ValueError("Unexpected data provenance.")
            prompt = row["prompt"]
            if (not isinstance(prompt, list) or len(prompt) != 2
                    or prompt[0] != {"role": "system", "content": SYSTEM_PROMPT}
                    or not isinstance(prompt[1], dict) or set(prompt[1]) != {"role", "content"}
                    or prompt[1]["role"] != "user"):
                raise ValueError("Expected one system instruction and one task context.")
            context = _loads(prompt[1]["content"])
            prompt_key = json.dumps(context, sort_keys=True)
            if prompt_key in seen_prompts:
                raise ValueError("Duplicate task prompt.")
            seen_prompts.add(prompt_key)
            if row["expected"] != _derive_expected(context):
                raise ValueError("Expected output is not grounded in the supplied evidence.")
            for label in ("chosen", "rejected"):
                messages = row[label]
                if (not isinstance(messages, list) or len(messages) != 1
                        or not isinstance(messages[0], dict)
                        or set(messages[0]) != {"role", "content"}
                        or messages[0]["role"] != "assistant"):
                    raise ValueError("Preference alternatives must each be one assistant message.")
                score = score_completion(messages[0]["content"], row["expected"])
                if not score["schema_valid"] or score["task_success"] != (label == "chosen"):
                    raise ValueError("Chosen/rejected alternatives violate their rule-derived labels.")
            key = (family, json.dumps(context["evidence"], sort_keys=True))
            reversals.setdefault(key, set()).add(row["expected"]["decision"])
        if set(row["role"] for row in rows) != {"HR", "Sales", "Science"}:
            raise ValueError("Each split must cover all three fictional roles.")
        if any(decisions != {"ready", "needs_review"} for decisions in reversals.values()):
            raise ValueError("Each evidence bundle needs a context-reversal pair within its split.")
        summary[split] = {"rows": len(rows), "families": len({row["family"] for row in rows}),
                          "roles": dict(sorted(Counter(row["role"] for row in rows).items())),
                          "decisions": dict(sorted(Counter(row["expected"]["decision"] for row in rows).items())),
                          "context_reversal_pairs": len(reversals)}
    return summary


def write_dataset(directory):
    splits = build_dataset()
    summary = audit_dataset(splits)
    output = Path(directory)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Dataset output directory must be absent or empty.")
    output.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for split in SPLITS:
        path = output / f"{split}.jsonl"
        content = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                          for row in splits[split])
        path.write_bytes(content.encode("utf-8"))
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {"schema_version": 1, "provenance": "synthetic-rule-derived",
                "task": "fictional handoff checklist completion and evidence selection",
                "limitations": "Rule correctness is not professional quality or human preference.",
                "splits": summary, "sha256": hashes}
    (output / "manifest.json").write_bytes((json.dumps(manifest, indent=2) + "\n").encode("utf-8"))
    return manifest
