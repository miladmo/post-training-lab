"""Fixed prompting conditions for the fictional handoff tasks."""

from copy import deepcopy
from functools import lru_cache
import json

from tasks import build_dataset


EXAMPLE_IDS = ("hr_recruiting_kickoff-1-1", "hr_recruiting_kickoff-1-2")
SYSTEM_PROMPT = (
    "You check a handoff checklist. Match each required field to the field names in the evidence. "
    "Return only a JSON object with exactly these keys: decision, missing, evidence_ids. "
    "Use decision ready when all required fields have evidence; otherwise use needs_review. "
    "The missing array contains required fields without evidence. The evidence_ids array "
    "contains IDs for required fields only. Sort both arrays. Follow the answer format in the examples."
)


def prompt_manifest(style):
    if style not in ("original", "two-shot"):
        raise ValueError(f"Unknown prompt style: {style!r}")
    return {"style": style, "template_version": 1,
            "example_ids": list(EXAMPLE_IDS) if style == "two-shot" else []}


def _question(messages):
    if (not isinstance(messages, list) or len(messages) != 2
            or messages[0].get("role") != "system" or messages[1].get("role") != "user"):
        raise ValueError("Expected the task's system instruction and user context.")
    context = json.loads(messages[1]["content"])
    facts = "\n".join(f"{fact['id']} | {fact['field']} | {fact['fact']}"
                      for fact in context["evidence"])
    return (f"Task: {context['task']}\n"
            f"Required fields: {', '.join(context['required_fields'])}\n"
            f"Evidence:\n{facts}\n"
            "Return the decision, missing fields and supporting evidence IDs as JSON.")


@lru_cache(maxsize=1)
def _examples():
    train = {row["id"]: row for row in build_dataset()["train"]}
    return tuple((_question(train[identifier]["prompt"]), train[identifier]["chosen"][0]["content"])
                 for identifier in EXAMPLE_IDS)


def prompt_messages(messages, style):
    """Return native chat messages without changing the task or its answer schema."""
    prompt_manifest(style)
    if style == "original":
        return deepcopy(messages)
    output = [{"role": "system", "content": SYSTEM_PROMPT}]
    for question, answer in _examples():
        output.extend(({"role": "user", "content": question},
                       {"role": "assistant", "content": answer}))
    output.append({"role": "user", "content": _question(messages)})
    return output
