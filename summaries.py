"""Aggregate checked completions by role and matched context-reversal pair."""
import json
from collections import defaultdict

from tasks import score_completion


def _index(records, label):
    indexed = {}
    for record in records:
        case_id = record["id"]
        if case_id in indexed:
            raise ValueError(f"Duplicate {label} case ID: {case_id}")
        indexed[case_id] = record
    return indexed


def _mean(scores):
    return {key: sum(score[key] for score in scores) / len(scores)
            for key in scores[0]}


def aggregate_results(rows, predictions):
    """Recompute scores; a complete pair succeeds only if both tasks fully succeed.

    Roles and evidence-pair membership come from the original task rows. Singleton
    pairs in a limited run are counted separately and excluded from pair accuracy.
    """
    cases = _index(rows, "task")
    saved = _index(predictions, "prediction")
    if not cases:
        raise ValueError("At least one original task row is required.")
    if cases.keys() != saved.keys():
        raise ValueError("Prediction IDs must exactly match the original task IDs.")

    all_scores = []
    role_scores = defaultdict(list)
    decision_scores = defaultdict(list)
    pairs = defaultdict(list)
    for case_id, row in cases.items():
        prediction = saved[case_id]
        if prediction["expected"] != row["expected"]:
            raise ValueError(f"Saved target differs from the original task: {case_id}")
        actual = score_completion(prediction["completion"], row["expected"])
        if prediction["metrics"] != actual:
            raise ValueError(f"Saved metrics differ from the completion: {case_id}")
        if row["role"] not in {"HR", "Sales", "Science"}:
            raise ValueError(f"Unsupported task role: {row['role']}")
        all_scores.append(actual)
        role_scores[row["role"]].append(actual)
        decision_scores[row["expected"]["decision"]].append(actual)
        context = json.loads(row["prompt"][1]["content"])
        pair_key = (row["family"], json.dumps(context["evidence"], sort_keys=True))
        pairs[pair_key].append((row["expected"], actual))

    count = both_correct = incomplete = 0
    for members in pairs.values():
        if len(members) == 1:
            incomplete += 1
            continue
        if len(members) != 2:
            raise ValueError("A context-reversal pair must contain exactly two task rows.")
        if {expected["decision"] for expected, _ in members} != {"ready", "needs_review"}:
            raise ValueError("Context-reversal partners must have opposite expected decisions.")
        count += 1
        both_correct += int(all(score["task_success"] == 1 for _, score in members))

    return {
        "metrics": _mean(all_scores),
        "roles": {role: {"count": len(scores), "metrics": _mean(scores)}
                  for role, scores in sorted(role_scores.items())},
        # Split by the expected decision exposes collapse onto one answer.
        "expected_decisions": {decision: {"count": len(scores), "metrics": _mean(scores)}
                               for decision, scores in sorted(decision_scores.items())},
        "context_pairs": {
            "count": count,
            "both_correct": both_correct,
            "accuracy": both_correct / count if count else None,
            "incomplete_pairs": incomplete,
        },
    }
