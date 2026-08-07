import argparse
import unittest

from benchmarks.cli import exact_tokens, median, positive_int


class BenchmarkTests(unittest.TestCase):
    def test_positive_int(self):
        self.assertEqual(positive_int("4"), 4)
        with self.assertRaises(argparse.ArgumentTypeError):
            positive_int("0")

    def test_exact_tokens(self):
        class Tokenizer:
            def encode(self, text, add_special_tokens=False):
                return [1, 2, 3]

        self.assertEqual(exact_tokens(Tokenizer(), 5, 1), [2, 3, 1, 2, 3])

    def test_median(self):
        self.assertEqual(median([(1.0, 4.0), (3.0, 2.0)]), (2.0, 3.0))


if __name__ == "__main__":
    unittest.main()
