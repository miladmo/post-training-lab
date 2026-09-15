import copy
import json
import unittest

from summaries import aggregate_results
from tasks import build_dataset, score_completion


def predictions_for(rows):
    return [{"id": row["id"], "expected": copy.deepcopy(row["expected"]),
             "completion": row["chosen"][0]["content"],
             "metrics": score_completion(row["chosen"][0]["content"], row["expected"])}
            for row in rows]


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.rows = build_dataset()["validation"]

    def test_all_correct_validation_has_twelve_complete_pairs(self):
        result = aggregate_results(self.rows, predictions_for(self.rows))
        self.assertEqual(len(result["metrics"]), 5)
        self.assertTrue(all(value == 1 for value in result["metrics"].values()))
        self.assertEqual(result["context_pairs"], {
            "count": 12, "both_correct": 12, "accuracy": 1.0, "incomplete_pairs": 0,
        })
        self.assertEqual(json.loads(json.dumps(result)), result)

    def test_one_wrong_partner_fails_the_pair_even_with_correct_decision(self):
        predictions = predictions_for(self.rows)
        expected = predictions[0]["expected"]
        changed = {**expected, "evidence_ids": []}
        self.assertTrue(expected["evidence_ids"])
        predictions[0]["completion"] = json.dumps(changed)
        predictions[0]["metrics"] = score_completion(predictions[0]["completion"], expected)
        result = aggregate_results(self.rows, predictions)
        self.assertEqual(result["metrics"]["decision_correct"], 1)
        self.assertEqual(result["metrics"]["task_success"], 23 / 24)
        self.assertEqual(result["context_pairs"]["both_correct"], 11)
        self.assertEqual(result["context_pairs"]["accuracy"], 11 / 12)

    def test_role_summaries_use_original_rows_and_prediction_ids(self):
        predictions = predictions_for(self.rows)
        predictions = predictions[7:] + predictions[:7]
        result = aggregate_results(self.rows, predictions)
        self.assertEqual({role: summary["count"] for role, summary in result["roles"].items()},
                         {"HR": 8, "Sales": 8, "Science": 8})
        self.assertTrue(all(summary["metrics"]["task_success"] == 1 for summary in result["roles"].values()))

    def test_wrong_saved_targets_and_metrics_are_rejected(self):
        for field in ("expected", "metrics"):
            with self.subTest(field=field):
                predictions = predictions_for(self.rows)
                if field == "expected":
                    target = predictions[0]["expected"]
                    target["decision"] = "needs_review" if target["decision"] == "ready" else "ready"
                else:
                    predictions[0]["metrics"]["task_success"] = 0
                with self.assertRaises(ValueError):
                    aggregate_results(self.rows, predictions)

    def test_duplicate_missing_and_extra_prediction_ids_are_rejected(self):
        good = predictions_for(self.rows)
        extra = {**good[0], "id": "extra-case"}
        for predictions in (good + [good[0]], good[:-1], good + [extra]):
            with self.subTest(count=len(predictions)):
                with self.assertRaises(ValueError):
                    aggregate_results(self.rows, predictions)
        with self.assertRaises(ValueError):
            aggregate_results(self.rows + [self.rows[0]], good)

    def test_three_partial_pairs_have_no_pair_accuracy(self):
        rows = [next(row for row in self.rows if row["role"] == role)
                for role in ("HR", "Sales", "Science")]
        result = aggregate_results(rows, predictions_for(rows))
        self.assertEqual(result["context_pairs"], {
            "count": 0, "both_correct": 0, "accuracy": None, "incomplete_pairs": 3,
        })
        self.assertEqual([summary["count"] for summary in result["roles"].values()], [1, 1, 1])

    def test_six_rows_can_cover_three_complete_pairs(self):
        rows = [row for role in ("HR", "Sales", "Science")
                for row in [entry for entry in self.rows if entry["role"] == role][:2]]
        result = aggregate_results(rows, predictions_for(rows))
        self.assertEqual(result["context_pairs"], {
            "count": 3, "both_correct": 3, "accuracy": 1.0, "incomplete_pairs": 0,
        })

    def test_invalid_pair_cardinality_is_not_silently_counted(self):
        rows = self.rows[:2] + [{**self.rows[0], "id": "third-partner"}]
        with self.assertRaisesRegex(ValueError, "exactly two"):
            aggregate_results(rows, predictions_for(rows))

    def test_same_decision_pair_is_not_a_context_reversal(self):
        rows = [self.rows[0], {**self.rows[0], "id": "same-decision-partner"}]
        with self.assertRaisesRegex(ValueError, "opposite"):
            aggregate_results(rows, predictions_for(rows))


if __name__ == "__main__":
    unittest.main()
