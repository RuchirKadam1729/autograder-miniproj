"""Unit tests for synoptic parsing and grading utilities."""

import io
import unittest

import pandas as pd

from src.grading import fill_missing_questions, round_to_half
from src.synoptic import SynopticEntry, build_synoptic_map, find_entry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _make_synoptic() -> dict:
    df = _make_df([
        {"question": "Q1", "subpart": "-", "max_marks": 10, "content": "Q1 full scheme"},
        {"question": "Q2", "subpart": "a", "max_marks": 5,  "content": "Q2a scheme"},
        {"question": "Q2", "subpart": "b", "max_marks": 5,  "content": "Q2b scheme"},
    ])
    return build_synoptic_map(df)


# ---------------------------------------------------------------------------
# Synoptic tests
# ---------------------------------------------------------------------------

class TestBuildSynopticMap(unittest.TestCase):

    def test_keys_are_correct(self):
        sm = _make_synoptic()
        self.assertIn("Q1.-", sm)
        self.assertIn("Q2.a", sm)
        self.assertIn("Q2.b", sm)

    def test_entry_fields(self):
        sm = _make_synoptic()
        entry = sm["Q1.-"]
        self.assertIsInstance(entry, SynopticEntry)
        self.assertEqual(entry.marks, 10.0)
        self.assertEqual(entry.content, "Q1 full scheme")

    def test_key_equals_question_dot_subpart(self):
        sm = _make_synoptic()
        for key, entry in sm.items():
            self.assertEqual(key, entry.key)


class TestFindEntry(unittest.TestCase):

    def test_exact_match(self):
        sm = _make_synoptic()
        entry = find_entry("Q2", "a", sm)
        self.assertIsNotNone(entry)
        self.assertEqual(entry.key, "Q2.a")

    def test_dash_fallback(self):
        sm = _make_synoptic()
        entry = find_entry("Q1", "a", sm)  # Q1 only has "-"
        self.assertIsNotNone(entry)
        self.assertEqual(entry.key, "Q1.-")

    def test_missing_returns_none(self):
        sm = _make_synoptic()
        entry = find_entry("Q99", "z", sm)
        self.assertIsNone(entry)


# ---------------------------------------------------------------------------
# Grading utility tests
# ---------------------------------------------------------------------------

class TestRoundToHalf(unittest.TestCase):

    def test_rounds_down(self):
        self.assertEqual(round_to_half(3.2), 3.0)

    def test_rounds_up(self):
        self.assertEqual(round_to_half(3.3), 3.5)

    def test_already_half(self):
        self.assertEqual(round_to_half(4.5), 4.5)

    def test_integer(self):
        self.assertEqual(round_to_half(7.0), 7.0)


class TestFillMissingQuestions(unittest.TestCase):

    def test_adds_missing_questions(self):
        sm = _make_synoptic()
        graded = _make_df([
            {"question": "Q1", "subpart": "-", "marks_awarded": 8, "max_marks": 10,
             "breakdown": [], "confidence": "high", "method": "llm"},
        ])
        result = fill_missing_questions(graded, sm)
        # Q2 should now appear with 0 marks
        q2_rows = result[result["question"] == "Q2"]
        self.assertFalse(q2_rows.empty)
        self.assertEqual(q2_rows.iloc[0]["marks_awarded"], 0.0)

    def test_no_duplicates_when_all_present(self):
        sm = _make_synoptic()
        graded = _make_df([
            {"question": "Q1", "subpart": "-", "marks_awarded": 8, "max_marks": 10,
             "breakdown": [], "confidence": "high", "method": "llm"},
            {"question": "Q2", "subpart": "a", "marks_awarded": 4, "max_marks": 5,
             "breakdown": [], "confidence": "high", "method": "llm"},
            {"question": "Q2", "subpart": "b", "marks_awarded": 3, "max_marks": 5,
             "breakdown": [], "confidence": "high", "method": "llm"},
        ])
        result = fill_missing_questions(graded, sm)
        self.assertEqual(len(result), 3)


if __name__ == "__main__":
    unittest.main()
