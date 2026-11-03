"""The selection code must return the paper's selections from the paper's recalls.

Usage:  python -m unittest tests.test_paper_selection
"""
import unittest

from examples.reproduce_paper_selection import main


class PaperSelection(unittest.TestCase):
    def test_every_selection_matches_the_paper(self):
        self.assertEqual(main(), 0)


if __name__ == "__main__":
    unittest.main()
