"""
tests/test_audit_data.py — Pure-function tests for data.py that don't touch
the network. load_truthful_qa_mc1() itself requires a HuggingFace Hub
download and is NOT tested here — see the "Running the real thing" section
of the README/verify script for how to smoke-test that part with network
access.
"""

from attn_phase.audit.data import build_choice_prompt, MCQuestion


def test_build_choice_prompt_format():
    prompt = build_choice_prompt("What color is the sky?", "Blue")
    assert prompt == "Q: What color is the sky?\nA: Blue"


def test_mcquestion_fields():
    q = MCQuestion(question_id=0, question="Q?",
                    choices=["a", "b", "c"], correct_idx=1)
    assert q.choices[q.correct_idx] == "b"
