import json
import unittest

import reward_model as rm
import run
import tasks

from test_grpo import HAS_TRAINING_STACK, ROOT


class SelectionAndAgreementTests(unittest.TestCase):
    def setUp(self):
        self.row = tasks.build_dataset()["validation"][0]
        self.answer = self.row["chosen"][0]["content"]

    def test_corruptions_all_fail_the_strict_schema(self):
        variants = rm.corruptions(self.answer)
        self.assertEqual(set(variants), {"trailing_text", "markdown_fence", "truncated", "renamed_key"})
        for name, text in variants.items():
            with self.subTest(corruption=name):
                self.assertEqual(tasks.score_completion(text, self.row["expected"])["schema_valid"], 0)

    def test_ranks_and_spearman(self):
        self.assertEqual(rm.rankdata([3, 1, 2, 2]), [4.0, 1.0, 2.5, 2.5])
        self.assertAlmostEqual(rm.spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertAlmostEqual(rm.spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)
        self.assertIsNone(rm.spearman([1, 2, 3], [0, 0, 0]))
        with self.assertRaises(ValueError):
            rm.spearman([1], [1])

    def test_canonical_answer_ignores_key_order_and_rejects_invalid(self):
        reordered = json.dumps(dict(reversed(list(json.loads(self.answer).items()))))
        self.assertEqual(rm.canonical_answer(self.answer), rm.canonical_answer(reordered))
        self.assertIsNone(rm.canonical_answer("not json"))
        self.assertIsNone(rm.canonical_answer(rm.corruptions(self.answer)["renamed_key"]))

    def test_selection_rules(self):
        def cand(text, rm_score, rule):
            return {"completion": text, "rm_score": rm_score, "rule_reward": rule,
                    "canonical": rm.canonical_answer(text)}

        wrong = json.dumps({**json.loads(self.answer), "decision": "needs_review"})
        candidates = [cand("garbage", 0.9, 0.0), cand(wrong, 0.1, 0.5), cand(self.answer, 0.5, 1.0),
                      cand(wrong, 0.2, 0.5)]
        chosen = rm.select(candidates)
        self.assertEqual(chosen, {"first_sample": 0, "majority_vote": 1, "reward_model": 0, "oracle": 2})
        tied = rm.select([cand("x", 1.0, 0.0), cand("y", 1.0, 0.0)])
        self.assertEqual(tied, dict.fromkeys(rm.SELECTIONS, 0))

    def test_commands_parse_and_reject_misuse(self):
        parser = run.build_parser()
        args = parser.parse_args(["best-of-n", "--reward-adapter", "r", "--output", "o", "--n", "4"])
        self.assertEqual((args.n, args.temperature, args.max_new_tokens), (4, 1.0, 96))
        with self.assertRaises(SystemExit):
            parser.parse_args(["best-of-n", "--reward-adapter", "r", "--output", "o", "--n", "1"])
        args = parser.parse_args(["train", "--method", "reward", "--init-adapter", "x", "--output", "o"])
        with self.assertRaises(ValueError):
            rm.train_reward(args)


@unittest.skipUnless(HAS_TRAINING_STACK, "torch, trl and peft are not installed")
class RewardModelSmokeTest(unittest.TestCase):
    """Reward model training, reliability checks and best-of-N on a tiny random model."""

    def test_reward_model_pipeline(self):
        import sys
        import tempfile
        from pathlib import Path

        sys.path.insert(0, str(ROOT / "tools"))
        from make_tiny_snapshot import MODEL, REVISION, build

        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            snapshot = build(root / "snapshots")["snapshot"]
            common = ["--model", MODEL, "--revision", REVISION, "--snapshot", snapshot,
                      "--prompt-style", "two-shot", "--threads", "1"]
            run.main(["train", *common, "--method", "reward", "--steps", "1", "--max-length", "4096",
                      "--output", str(root / "rm")])
            run.main(["reward-eval", *common, "--reward-adapter", str(root / "rm" / "adapter"),
                      "--limit", "2", "--max-length", "4096", "--output", str(root / "rm-eval")])
            summary = json.loads((root / "rm-eval" / "reward_eval.json").read_text())
            self.assertIn("pairwise_accuracy", summary)
            self.assertEqual(set(summary["prefers_valid_over_malformed"]),
                             set(rm.corruptions('{"decision":"ready","missing":[],"evidence_ids":[]}')))
            run.main(["best-of-n", *common, "--reward-adapter", str(root / "rm" / "adapter"),
                      "--n", "2", "--limit", "2", "--max-new-tokens", "6", "--max-length", "4096",
                      "--output", str(root / "bon")])
            result = json.loads((root / "bon" / "best_of_n.json").read_text())
            self.assertEqual(set(result["selections"]), set(rm.SELECTIONS))
            self.assertEqual(result["sampling"]["n"], 2)


if __name__ == "__main__":
    unittest.main()
