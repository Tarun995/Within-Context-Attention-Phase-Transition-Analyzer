"""
test_answer_matching.py

Regression tests for answer_matches() in attn_phase.tasks.

These tests exist because answer_matches() had a real bug during development:
the original implementation regex-matched only the first integer from the
continuation, which silently mis-scored every non-arithmetic task:
  - lookup expected "v0" -> regex extracted "0" -> spurious mismatch
  - sorting expected "1 3 4 6" -> regex extracted "1" -> always wrong

These tests lock in the correct behaviour so that bug cannot silently
reappear after any future refactor.
"""

import pytest
from attn_phase.tasks import answer_matches


# ---------------------------------------------------------------------------
# Case 1: single integer expected (mod_arith)
# ---------------------------------------------------------------------------

class TestSingleIntegerExpected:

    def test_exact_match(self):
        solved, extracted = answer_matches("53", "53")
        assert solved is True
        assert extracted == "53"

    def test_match_with_continuation(self):
        # Model often keeps generating after the answer
        solved, extracted = answer_matches("53 , 12 + 9 mod 97 =", "53")
        assert solved is True
        assert extracted == "53"

    def test_wrong_integer(self):
        solved, extracted = answer_matches("68", "53")
        assert solved is False
        assert extracted == "68"

    def test_negative_expected(self):
        solved, extracted = answer_matches("-5 , next", "-5")
        assert solved is True
        assert extracted == "-5"

    def test_zero_expected(self):
        solved, extracted = answer_matches("0 , ...", "0")
        assert solved is True
        assert extracted == "0"

    def test_empty_continuation(self):
        solved, extracted = answer_matches("", "53")
        assert solved is False
        assert extracted is None


# ---------------------------------------------------------------------------
# Case 2: single non-numeric token expected (lookup, copy)
# ---------------------------------------------------------------------------

class TestSingleNonNumericToken:

    def test_lookup_correct(self):
        # The old regex-only approach extracted "0" from "v0" -> wrong
        solved, extracted = answer_matches("v0 , k0 = v0 ,", "v0")
        assert solved is True
        assert extracted == "v0"

    def test_lookup_wrong(self):
        solved, extracted = answer_matches("v3 , k0 = v0 ,", "v0")
        assert solved is False
        assert extracted == "v3"

    def test_copy_token_correct(self):
        solved, extracted = answer_matches("tok22 , tok22 tok22", "tok22")
        assert solved is True
        assert extracted == "tok22"

    def test_copy_token_wrong(self):
        solved, extracted = answer_matches("tok5 , tok22 tok22", "tok22")
        assert solved is False
        assert extracted == "tok5"

    def test_single_letter_token(self):
        solved, extracted = answer_matches("g , j -> b ,", "g")
        assert solved is True
        assert extracted == "g"

    def test_single_letter_wrong(self):
        solved, extracted = answer_matches("b , j -> b ,", "g")
        assert solved is False
        assert extracted == "b"


# ---------------------------------------------------------------------------
# Case 3: multi-token expected (sorting)
# ---------------------------------------------------------------------------

class TestMultiTokenExpected:

    def test_sorting_correct(self):
        solved, extracted = answer_matches("1 3 4 6 , sort 1 8 4 9", "1 3 4 6")
        assert solved is True
        assert extracted == "1 3 4 6"

    def test_sorting_wrong(self):
        # Old code extracted only "1" (first token) -> looked correct for
        # any sorted sequence starting with 1. New code checks all tokens.
        solved, extracted = answer_matches("1 1 5 6 , sort 1 8 4 9", "1 3 4 6")
        assert solved is False
        assert extracted == "1 1 5 6"

    def test_two_token_expected(self):
        solved, extracted = answer_matches("yes no maybe", "yes no")
        assert solved is True
        assert extracted == "yes no"

    def test_multi_token_partial_match_is_failure(self):
        # Matching only the first token of a multi-token answer is wrong
        solved, extracted = answer_matches("1 9 9 9", "1 3 4 6")
        assert solved is False

    def test_empty_continuation_multi(self):
        solved, extracted = answer_matches("", "1 3 4 6")
        assert solved is False