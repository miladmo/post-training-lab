import copy
import json
from pathlib import Path
import tempfile
import unittest

import run
import tasks


ROOT = Path(__file__).resolve().parents[1]


class SelectionTests(unittest.TestCase):
    def test_short_selection_interleaves_all_roles(self):
        rows = tasks.build_dataset()["validation"]
        selected = run.select_rows(rows, 6)
        self.assertEqual([row["role"] for row in selected], ["HR", "Sales", "Science"] * 2)
        self.assertEqual(len({row["id"] for row in selected}), 6)
        self.assertEqual(len(run.select_rows(rows, None)), len(rows))

    def test_unequal_role_groups_do_not_drop_rows(self):
        rows = [{"id": "h1", "role": "HR"}, {"id": "h2", "role": "HR"},
                {"id": "s1", "role": "Sales"}, {"id": "c1", "role": "Science"}]
        selected = run.select_rows(rows, None)
        self.assertEqual([row["id"] for row in selected], ["h1", "s1", "c1", "h2"])


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        rows = run.select_rows(tasks.build_dataset()["validation"], 3)
        self.predictions = [{
            "id": row["id"], "role": row["role"], "family": row["family"],
            "prompt_sha256": run.fingerprint(row["prompt"]), "expected": row["expected"],
            "completion": row["chosen"][0]["content"],
            "metrics": tasks.score_completion(row["chosen"][0]["content"], row["expected"]),
            "new_tokens": 40, "seconds": 0.1,
        } for row in rows]
        self.meta = {
            "model": "fictional/test-model", "revision": "a" * 40,
            "model_files_sha256": {"model.safetensors": "c" * 64},
            "split": "validation", "count": 3, "data_sha256": run.fingerprint(rows),
            "generation": {"do_sample": False, "max_new_tokens": 96},
            "thinking": False, "dtype": "float32", "device": "cpu", "threads": 4,
            "seed": 42, "versions": {"torch": "test", "trl": "test"},
            "condition": "base", "metrics": dict.fromkeys(self.predictions[0]["metrics"], 1.0),
            "seconds": 0.3,
        }

    def write_evaluation(self, name, meta=None, predictions=None):
        directory = self.root / name
        directory.mkdir()
        (directory / "evaluation.json").write_text(json.dumps(meta or self.meta), encoding="utf-8")
        (directory / "predictions.json").write_text(
            json.dumps(self.predictions if predictions is None else predictions), encoding="utf-8")
        return directory

    def test_matching_base_and_adapter_results_are_comparable(self):
        base = self.write_evaluation("base")
        adapter = self.write_evaluation("sft", {**self.meta, "condition": "sft", "seconds": 0.4})
        report = run.compare([base, adapter])
        self.assertEqual([row["condition"] for row in report], ["base", "sft"])
        self.assertEqual([row["count"] for row in report], [3, 3])

    def test_settings_and_case_changes_are_rejected(self):
        base = self.write_evaluation("base")
        changes = {"revision": "b" * 40, "split": "test", "data_sha256": "different",
                   "model_files_sha256": {"model.safetensors": "d" * 64},
                   "generation": {"do_sample": False, "max_new_tokens": 48},
                   "seed": 7, "versions": {"torch": "different", "trl": "test"}}
        for key, value in changes.items():
            with self.subTest(setting=key):
                other = self.write_evaluation(key, {**self.meta, key: value})
                with self.assertRaises(ValueError):
                    run.compare([base, other])
        predictions = copy.deepcopy(self.predictions)
        predictions[0]["prompt_sha256"] = "changed-prompt"
        other = self.write_evaluation("changed-case", predictions=predictions)
        with self.assertRaises(ValueError):
            run.compare([base, other])

    def test_incomplete_evaluation_is_rejected(self):
        incomplete = self.write_evaluation("incomplete", predictions=self.predictions[:-1])
        with self.assertRaises(ValueError):
            run.compare([incomplete])

    def test_metrics_must_match_stored_completions(self):
        predictions = copy.deepcopy(self.predictions)
        predictions[0]["completion"] = "invalid JSON"
        stale = self.write_evaluation("stale-metrics", predictions=predictions)
        with self.assertRaises(ValueError):
            run.compare([stale])
        meta = copy.deepcopy(self.meta)
        meta["metrics"]["task_success"] = 0.0
        stale_summary = self.write_evaluation("stale-summary", meta=meta)
        with self.assertRaises(ValueError):
            run.compare([stale_summary])

    def test_duplicate_case_ids_are_rejected(self):
        predictions = copy.deepcopy(self.predictions)
        predictions[1]["id"] = predictions[0]["id"]
        duplicates = self.write_evaluation("duplicates", predictions=predictions)
        with self.assertRaises(ValueError):
            run.compare([duplicates])


if __name__ == "__main__":
    unittest.main()
