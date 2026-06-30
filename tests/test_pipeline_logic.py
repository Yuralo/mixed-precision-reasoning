import unittest

from src.compare_outputs import compare_runs
from src.feature_logging import build_feature_tables
from src.oracle_recovery import oracle_recovery_report


def output(example_id, answer, correct):
    return {
        "example_id": example_id,
        "dataset": "gsm8k",
        "split": "test",
        "question": "q",
        "reference": "#### 7",
        "reference_answer": "7",
        "generation": f"#### {answer}",
        "predicted_answer": str(answer),
        "correct": correct,
        "prompt_tokens": 4,
        "generation_tokens": 1,
        "model": {"quantization": "none"},
    }


class PipelineLogicTests(unittest.TestCase):
    def test_comparison_and_aggregation(self):
        comparisons, summary = compare_runs(
            [output("a", 7, True), output("b", 7, True), output("c", 6, False)],
            [output("a", 6, False), output("b", 7, True), output("c", 7, True)],
        )
        self.assertEqual(summary["fp_correct_quant_wrong"], 1)
        self.assertEqual(summary["critical_failure_rate"], 0.5)
        fp_tokens = [
            {"example_id": key, "token_position": 0, "token_id": 7, "entropy": 0.2, "logit_margin": 3.0}
            for key in ("a", "b")
        ]
        q_tokens = [
            {
                "example_id": key,
                "dataset": "gsm8k",
                "split": "test",
                "run_type": "fake",
                "token_position": 0,
                "token_id": token,
                "token": str(token),
                "token_probability": 0.6,
                "surprisal": 0.5,
                "entropy": entropy,
                "logit_margin": margin,
                "top2_token_id": 1,
                "top2_probability": 0.2,
            }
            for key, token, entropy, margin in (("a", 6, 1.2, 0.1), ("b", 7, 0.3, 2.0))
        ]
        token_rows, example_rows = build_feature_tables(fp_tokens, q_tokens, comparisons)
        self.assertEqual(len(token_rows), 2)
        by_id = {row["example_id"]: row for row in example_rows}
        self.assertEqual(by_id["a"]["target_quantization_failure"], 1)
        self.assertEqual(by_id["a"]["max_entropy"], 1.2)

        oracle = oracle_recovery_report(comparisons, [0.0, 0.5, 1.0])
        self.assertEqual(oracle["best_selector_accuracy"], 1.0)
        self.assertEqual(oracle["at_most_budget_curve"][-1]["accuracy"], 1.0)
        self.assertAlmostEqual(oracle["exact_budget_curve"][-1]["accuracy"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
