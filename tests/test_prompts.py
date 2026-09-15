import copy
import json
import unittest

from prompts import prompt_manifest, prompt_messages
from tasks import build_dataset, score_completion


class PromptTests(unittest.TestCase):
    def setUp(self):
        self.splits = build_dataset()

    def test_original_is_unchanged_and_independent(self):
        messages = self.splits["train"][0]["prompt"]
        before = copy.deepcopy(messages)
        rendered = prompt_messages(messages, "original")
        self.assertEqual(rendered, before)
        rendered[0]["content"] = "changed copy"
        self.assertEqual(messages, before)
        self.assertEqual(prompt_manifest("original")["example_ids"], [])

    def test_examples_are_fixed_training_rows_with_exact_answers(self):
        manifest = prompt_manifest("two-shot")
        self.assertEqual(manifest["example_ids"],
                         ["hr_recruiting_kickoff-1-1", "hr_recruiting_kickoff-1-2"])
        train = {row["id"]: row for row in self.splits["train"]}
        for split in ("validation", "test"):
            self.assertFalse(set(manifest["example_ids"]) & {row["id"] for row in self.splits[split]})
        target = self.splits["validation"][0]
        rendered = prompt_messages(target["prompt"], "two-shot")
        self.assertEqual([message["role"] for message in rendered],
                         ["system", "user", "assistant", "user", "assistant", "user"])
        for identifier, index in zip(manifest["example_ids"], (2, 4)):
            source = train[identifier]
            self.assertEqual(rendered[index], source["chosen"][0])
            self.assertEqual(json.loads(rendered[index]["content"]), source["expected"])
            self.assertEqual(score_completion(rendered[index]["content"], source["expected"])["task_success"], 1)

    def test_reversal_preserves_evidence_and_changes_only_target_requirements(self):
        first, second = self.splits["validation"][:2]
        originals = [copy.deepcopy(row["prompt"]) for row in (first, second)]
        rendered = [prompt_messages(row["prompt"], "two-shot") for row in (first, second)]
        self.assertEqual(rendered[0][:-1], rendered[1][:-1])
        texts = [messages[-1]["content"] for messages in rendered]
        self.assertNotEqual(texts[0], texts[1])
        self.assertEqual(texts[0].split("Evidence:\n")[1], texts[1].split("Evidence:\n")[1])
        for row, text, original in zip((first, second), texts, originals):
            context = json.loads(row["prompt"][1]["content"])
            self.assertIn(f"Task: {context['task']}\n", text)
            self.assertIn(f"Required fields: {', '.join(context['required_fields'])}\n", text)
            for fact in context["evidence"]:
                self.assertIn(f"{fact['id']} | {fact['field']} | {fact['fact']}", text)
            self.assertEqual(row["prompt"], original)

    def test_returned_examples_cannot_modify_future_prompts(self):
        messages = self.splits["validation"][0]["prompt"]
        expected = prompt_messages(messages, "two-shot")
        changed = prompt_messages(messages, "two-shot")
        changed[2]["content"] = "damaged example"
        self.assertEqual(prompt_messages(messages, "two-shot"), expected)
        manifest = prompt_manifest("two-shot")
        manifest["example_ids"].clear()
        self.assertEqual(len(prompt_manifest("two-shot")["example_ids"]), 2)

    def test_unknown_styles_fail(self):
        for style in ("few-shot", "", None):
            with self.subTest(style=style):
                with self.assertRaises(ValueError):
                    prompt_manifest(style)
                with self.assertRaises(ValueError):
                    prompt_messages(self.splits["train"][0]["prompt"], style)


if __name__ == "__main__":
    unittest.main()
