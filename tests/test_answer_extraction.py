import unittest

from src.answer_extraction import extract_answer, extract_hash_answer, is_correct


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


if __name__ == "__main__":
    unittest.main()
