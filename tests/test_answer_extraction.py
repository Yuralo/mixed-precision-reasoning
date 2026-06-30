import unittest

from src.answer_extraction import extract_answer, extract_explicit_answer, extract_hash_answer, is_correct


class AnswerExtractionTests(unittest.TestCase):
    def test_prefers_hash_answer(self):
        self.assertEqual(extract_answer("First 10, finally #### 1,234.50"), "1234.5")

    def test_falls_back_to_last_number(self):
        self.assertEqual(extract_answer("We get 3 + 4 = 7."), "7")

    def test_correctness(self):
        self.assertTrue(is_correct("Therefore #### 8", "work #### 8.0"))

    def test_strict_hash_extraction(self):
        self.assertIsNone(extract_hash_answer("The final number is 8"))
        self.assertEqual(extract_hash_answer("Therefore #### 8"), "8")

    def test_qwen_explicit_answer_styles(self):
        self.assertEqual(extract_explicit_answer("Final answer: $18"), "18")
        self.assertEqual(extract_explicit_answer(r"Therefore, the answer is \\boxed{540}."), "540")
        self.assertEqual(extract_explicit_answer("The total is **3 bolts**."), "3")
        self.assertIsNone(extract_explicit_answer("The unfinished calculation is 60 cups"))


if __name__ == "__main__":
    unittest.main()
