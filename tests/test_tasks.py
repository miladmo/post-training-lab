import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import tasks


class TaskTests(unittest.TestCase):
    def setUp(self):
        self.splits = tasks.build_dataset()
        self.row = self.splits["train"][0]

    def test_family_holdout_and_context_reversals(self):
        summary = tasks.audit_dataset(self.splits)
        self.assertEqual([summary[s]["rows"] for s in tasks.SPLITS], [48, 24, 24])
        families = [{row["family"] for row in self.splits[s]} for s in tasks.SPLITS]
        self.assertFalse(families[0] & families[1] or families[0] & families[2] or families[1] & families[2])
        for split in tasks.SPLITS:
            self.assertEqual(summary[split]["context_reversal_pairs"], len(self.splits[split]) // 2)
            self.assertEqual(set(summary[split]["roles"]), {"HR", "Sales", "Science"})
        first, second = self.splits["train"][:2]
        contexts = [json.loads(row["prompt"][1]["content"]) for row in (first, second)]
        self.assertEqual(contexts[0]["evidence"], contexts[1]["evidence"])
        self.assertNotEqual(contexts[0]["required_fields"], contexts[1]["required_fields"])
        self.assertNotEqual(first["expected"]["decision"], second["expected"]["decision"])
        # Evidence for an unrequired field must not be cited.
        self.assertLess(len(first["expected"]["evidence_ids"]), len(contexts[0]["evidence"]))

    def test_chosen_and_rejected_are_distinguished(self):
        for rows in self.splits.values():
            for row in rows:
                with self.subTest(row=row["id"]):
                    chosen = tasks.score_completion(row["chosen"][0]["content"], row["expected"])
                    rejected = tasks.score_completion(row["rejected"][0]["content"], row["expected"])
                    self.assertTrue(all(chosen.values()))
                    self.assertEqual(rejected["schema_valid"], 1)
                    self.assertEqual(rejected["task_success"], 0)

    def test_strict_json_schema_rejects_malformed_outputs(self):
        valid = self.row["chosen"][0]["content"]
        target = self.row["expected"]
        bad = ("prefix " + valid, "```json\n" + valid + "\n```", valid + valid,
               "[]", "null", "{", valid[:-1] + ',"extra":1}',
               valid[:-1] + ',"decision":"ready"}',
               '{"decision":"ready","missing":[],"evidence_ids":["E1","E1"]}',
               '{"decision":"ready","missing":[],"evidence_ids":["E2","E1"]}',
               '{"decision":true,"missing":[],"evidence_ids":[]}',
               '{"decision":"ready","missing":[],"evidence_ids":[NaN]}',
               '{"decision":"ready","missing":"none","evidence_ids":[]}',
               '{"decision":"unknown","missing":[],"evidence_ids":[]}')
        for text in bad:
            with self.subTest(text=text):
                self.assertFalse(any(tasks.score_completion(text, target).values()))

    def test_valid_but_wrong_field_values_fail_exact_checks(self):
        target = self.row["expected"]
        wrong = {**target, "evidence_ids": ["NONEXISTENT"]}
        scored = tasks.score_completion(json.dumps(wrong), target)
        self.assertEqual(scored["schema_valid"], 1)
        self.assertEqual(scored["decision_correct"], 1)
        self.assertEqual(scored["evidence_exact"], 0)
        self.assertEqual(scored["task_success"], 0)
        wrong = {**target, "missing": ["invented_field"]}
        self.assertEqual(tasks.score_completion(json.dumps(wrong), target)["missing_fields_exact"], 0)

    def test_audit_rejects_overlap_duplicates_and_ungrounded_targets(self):
        mutations = []
        overlap = copy.deepcopy(self.splits)
        overlap["validation"][0]["family"] = overlap["train"][0]["family"]
        mutations.append(overlap)
        duplicate = copy.deepcopy(self.splits)
        duplicate["test"][0]["id"] = duplicate["train"][0]["id"]
        mutations.append(duplicate)
        wrong = copy.deepcopy(self.splits)
        wrong["train"][0]["expected"]["evidence_ids"] = ["UNKNOWN"]
        mutations.append(wrong)
        evidence = copy.deepcopy(self.splits)
        context = json.loads(evidence["train"][0]["prompt"][1]["content"])
        context["evidence"][1]["id"] = context["evidence"][0]["id"]
        evidence["train"][0]["prompt"][1]["content"] = json.dumps(context)
        mutations.append(evidence)
        for splits in mutations:
            with self.assertRaises(ValueError):
                tasks.audit_dataset(splits)

    def test_reward_supports_conversational_and_string_completions(self):
        chosen = self.row["chosen"]
        target = self.row["expected"]
        scores = tasks.reward([chosen, chosen[0]["content"], "invalid", self.row["rejected"]],
                              [target] * 4, prompts=[self.row["prompt"]] * 4)
        self.assertEqual(scores[:3], [1.0, 1.0, 0.0])
        self.assertGreaterEqual(scores[3], 0.0)
        self.assertLess(scores[3], 1.0)
        with self.assertRaises(ValueError):
            tasks.reward([chosen], [])

    def test_written_dataset_is_deterministic_and_hash_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / "one", Path(directory) / "two"
            manifest = tasks.write_dataset(first)
            self.assertEqual(manifest, tasks.write_dataset(second))
            for name, digest in manifest["sha256"].items():
                self.assertEqual(hashlib.sha256((first / name).read_bytes()).hexdigest(), digest)
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())
            loaded = {s: [json.loads(line) for line in (first / f"{s}.jsonl").read_text().splitlines()]
                      for s in tasks.SPLITS}
            self.assertEqual(tasks.audit_dataset(loaded), manifest["splits"])
            with self.assertRaises(ValueError):
                tasks.write_dataset(first)


if __name__ == "__main__":
    unittest.main()
