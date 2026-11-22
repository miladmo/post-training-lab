import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import run
import tasks


ROOT = Path(__file__).resolve().parents[1]
HAS_TRAINING_STACK = all(importlib.util.find_spec(name) for name in ("torch", "trl", "peft"))


class GradedRewardTests(unittest.TestCase):
    def setUp(self):
        self.row = tasks.build_dataset()["train"][0]
        self.target = self.row["expected"]

    def answer(self, **changes):
        return json.dumps({**self.target, **changes})

    def test_correct_answer_earns_full_credit_on_every_component(self):
        parts = tasks.graded_components(self.answer(), self.target)
        self.assertEqual(parts, dict.fromkeys(tasks.REWARD_COMPONENTS, 1.0))
        self.assertEqual(tasks.graded_reward([self.row["chosen"]], [self.target]), [1.0])

    def test_malformed_output_earns_nothing(self):
        for text in ("", "not json", '{"decision": "ready"}', self.answer() + " extra", None):
            with self.subTest(text=text):
                self.assertEqual(tasks.graded_components(text, self.target),
                                 dict.fromkeys(tasks.REWARD_COMPONENTS, 0.0))

    def test_set_fields_receive_partial_credit(self):
        self.assertEqual(tasks._set_f1(["a", "b"], ["a", "b"]), 1.0)
        self.assertEqual(tasks._set_f1([], []), 1.0)
        self.assertEqual(tasks._set_f1(["a"], []), 0.0)
        self.assertEqual(tasks._set_f1(["c"], ["a", "b"]), 0.0)
        self.assertAlmostEqual(tasks._set_f1(["a"], ["a", "b"]), 2 / 3)

    def test_partial_answer_scores_between_zero_and_one(self):
        expected = {"decision": "needs_review", "missing": ["buddy", "start_date"],
                    "evidence_ids": ["E1", "E2"]}
        partial = json.dumps({"decision": "needs_review", "missing": ["buddy"],
                              "evidence_ids": ["E1", "E2"]})
        parts = tasks.graded_components(partial, expected)
        self.assertEqual(parts["format"], 1.0)
        self.assertEqual(parts["decision"], 1.0)
        self.assertAlmostEqual(parts["missing_f1"], 2 / 3)
        self.assertEqual(parts["evidence_f1"], 1.0)
        score = tasks.graded_reward([partial], [expected])[0]
        self.assertTrue(0 < score < 1)
        self.assertEqual(tasks.score_completion(partial, expected)["task_success"], 0)

    def test_component_functions_are_named_for_logging(self):
        names = [function.__name__ for function in tasks.GRADED_REWARDS]
        self.assertEqual(names, [f"reward_{name}" for name in tasks.REWARD_COMPONENTS])
        with self.assertRaises(ValueError):
            tasks.GRADED_REWARDS[0](["x"], [])


class GRPOConfigurationTests(unittest.TestCase):
    def parse(self, *extra):
        return run.build_parser().parse_args(
            ["train", "--method", "grpo", "--output", "unused", *extra])

    def test_defaults_follow_protocol_v3(self):
        settings = run.grpo_settings(self.parse())
        self.assertEqual(settings["loss_type"], "dapo")
        self.assertEqual(settings["num_generations"], 8)
        self.assertEqual(settings["prompts_per_step"], 2)
        self.assertEqual(settings["epsilon_high"], 0.28)
        self.assertEqual(settings["scale_rewards"], "group")
        self.assertTrue(settings["mask_truncated_completions"])
        self.assertEqual(settings["reward"], "graded")
        self.assertEqual(settings["beta"], 0.0)

    def test_dr_grpo_removes_std_scaling_and_clip_higher(self):
        settings = run.grpo_settings(self.parse("--grpo-loss", "dr_grpo", "--keep-truncated"))
        self.assertEqual(settings["scale_rewards"], "none")
        self.assertEqual(settings["epsilon_high"], 0.2)
        self.assertFalse(settings["mask_truncated_completions"])

    def test_invalid_group_and_temperature_are_rejected(self):
        for extra in (("--num-generations", "1"), ("--temperature", "0")):
            with self.subTest(extra=extra), self.assertRaises(SystemExit):
                self.parse(*extra)

    def test_init_adapter_is_limited_to_grpo(self):
        args = run.build_parser().parse_args(
            ["train", "--method", "sft", "--init-adapter", "x", "--output", "unused"])
        with self.assertRaises(ValueError):
            run.train(args)

    def test_adapter_metadata_must_match_base_and_prompt(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            run_dir = Path(directory)
            (run_dir / "adapter").mkdir()
            args = run.build_parser().parse_args(
                ["train", "--method", "grpo", "--output", "unused", "--prompt-style", "two-shot",
                 "--model", "fictional/model", "--revision", "a" * 40])
            meta = {"model": "fictional/model", "revision": "a" * 40, "method": "sft",
                    "prompt": run.prompt_manifest("two-shot")}
            (run_dir / "training.json").write_text(json.dumps(meta), encoding="utf-8")
            self.assertEqual(run.load_adapter_metadata(run_dir / "adapter", args), meta)
            self.assertEqual(run.condition_name(meta), "sft")
            for key, value in (("revision", "b" * 40), ("prompt", run.prompt_manifest("original"))):
                with self.subTest(changed=key):
                    (run_dir / "training.json").write_text(json.dumps({**meta, key: value}))
                    with self.assertRaises(ValueError):
                        run.load_adapter_metadata(run_dir / "adapter", args)

    def test_diagnostics_summarize_the_reward_signal(self):
        history = [
            {"reward": 0.2, "frac_reward_zero_std": 0.5, "rewards/reward_format/mean": 0.8,
             "completions/clipped_ratio": 0.1, "completions/mean_length": 40},
            {"reward": 0.6, "frac_reward_zero_std": 0.25, "rewards/reward_format/mean": 1.0,
             "completions/clipped_ratio": 0.0, "completions/mean_length": 36},
            {"train_runtime": 10.0},
        ]
        summary = run.grpo_diagnostics(history)
        self.assertEqual(summary["logged_steps"], 2)
        self.assertEqual((summary["reward_first"], summary["reward_last"]), (0.2, 0.6))
        self.assertAlmostEqual(summary["frac_reward_zero_std_mean"], 0.375)
        self.assertAlmostEqual(summary["component_means"]["format"], 0.9)
        self.assertEqual(run.grpo_diagnostics([{"train_runtime": 1.0}]), {})


@unittest.skipUnless(HAS_TRAINING_STACK, "torch, trl and peft are not installed")
class TinyModelSmokeTest(unittest.TestCase):
    """End-to-end SFT -> GRPO -> evaluation on a tiny random model (no downloads)."""

    def test_sft_then_grpo_then_evaluate(self):
        import sys

        sys.path.insert(0, str(ROOT / "tools"))
        from make_tiny_snapshot import MODEL, REVISION, build

        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            snapshot = build(root / "snapshots")["snapshot"]
            common = ["--model", MODEL, "--revision", REVISION, "--snapshot", snapshot,
                      "--prompt-style", "two-shot", "--threads", "1"]
            run.main(["train", *common, "--method", "sft", "--steps", "1", "--max-length", "4096",
                      "--output", str(root / "sft")])
            run.main(["train", *common, "--method", "grpo", "--steps", "1", "--max-length", "4096",
                      "--init-adapter", str(root / "sft" / "adapter"), "--num-generations", "2",
                      "--prompts-per-step", "1", "--max-completion-length", "8",
                      "--learning-rate", "5e-5", "--output", str(root / "grpo")])
            meta = json.loads((root / "grpo" / "training.json").read_text())
            self.assertEqual(meta["condition"], "sft+grpo")
            self.assertEqual(meta["init_condition"], "sft")
            self.assertIn("frac_reward_zero_std_mean", meta["grpo_diagnostics"])
            run.main(["evaluate", *common, "--adapter", str(root / "grpo" / "adapter"),
                      "--limit", "3", "--max-new-tokens", "8", "--output", str(root / "eval")])
            evaluation = json.loads((root / "eval" / "evaluation.json").read_text())
            self.assertEqual(evaluation["condition"], "sft+grpo")
            self.assertEqual(evaluation["count"], 3)


if __name__ == "__main__":
    unittest.main()
