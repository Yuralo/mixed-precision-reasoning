import unittest

from src.paired_analysis import aggregate_token_trace, first_divergence
from src.temperature_experiment import analyze_temperature_outputs
from src.trajectory_metrics import analyze_trajectory


def output(example_id, correct, tokens=10, answer="1"):
    return {
        "example_id": example_id,
        "correct": correct,
        "generation_tokens": tokens,
        "predicted_answer": answer,
    }


class NewAnalysisTests(unittest.TestCase):
    def test_trajectory_structure(self):
        metrics = analyze_trajectory(
            "1. Compute 3 + 4 = 7. Wait, check it.\nFINAL_ANSWER: 7", 12
        )
        self.assertGreaterEqual(metrics["arithmetic_expression_count"], 1)
        self.assertGreaterEqual(metrics["self_correction_count"], 1)
        self.assertTrue(metrics["has_explicit_answer"])

    def test_trace_and_divergence(self):
        fp = [
            {"token_id": 1, "token": "A", "entropy": 1.0, "logit_margin": 2.0, "token_probability": 0.7},
            {"token_id": 2, "token": "B", "entropy": 2.0, "logit_margin": 1.0, "token_probability": 0.6},
        ]
        quant = [
            {"token_id": 1, "token": "A", "entropy": 1.2, "logit_margin": 1.8, "token_probability": 0.65},
            {"token_id": 3, "token": "C", "entropy": 2.2, "logit_margin": 0.8, "token_probability": 0.55},
        ]
        self.assertEqual(first_divergence(fp, quant)["first_divergence_position"], 1)
        self.assertAlmostEqual(aggregate_token_trace(fp)["mean_entropy"], 1.5)

    def test_temperature_rescue_overlap(self):
        fp = [output("a", False), output("b", True), output("c", False)]
        quant = [output("a", True), output("b", False), output("c", False)]
        sampled = [
            {**output("a", True, answer="1"), "temperature": 0.3},
            {**output("a", False, answer="2"), "temperature": 0.3},
            {**output("b", True), "temperature": 0.3},
            {**output("b", True), "temperature": 0.3},
            {**output("c", False), "temperature": 0.3},
            {**output("c", False), "temperature": 0.3},
        ]
        report = analyze_temperature_outputs(fp, quant, sampled)
        temp = report["temperatures"]["0.3"]
        self.assertEqual(temp["quant_rescues_reproduced_by_any_sample"], 1)
        self.assertEqual(temp["quant_rescue_coverage"], 1.0)
        self.assertEqual(temp["quant_rescues_reproduced_by_majority"], 0)
        self.assertEqual(temp["quant_rescue_majority_coverage"], 0.0)
        self.assertAlmostEqual(temp["per_completion_accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
