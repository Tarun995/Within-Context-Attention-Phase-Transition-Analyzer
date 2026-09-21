"""
audit/data.py — TruthfulQA (multiple-choice, mc1) loading and prompt formatting.

WHY mc1, NOT free-generation:
TruthfulQA's mc1_targets format gives exactly one correct choice per question
out of several plausible-but-wrong ones, with no free-text answer-matching
step required. That removes an entire failure surface (fuzzy string matching,
partial credit, "close enough" judgment calls) that would otherwise sit
between the model's output and a correctness label — and the project's own
tasks.py/answer_matches already had a scoring bug once (see docs/FINDINGS.md,
Bug #1). mc1's clean binary correctness is worth the narrower scope for a
first baseline.

REQUIRES NETWORK: load_truthful_qa_mc1() downloads from the HuggingFace Hub
the first time it's called (cached locally after). This will fail with no
internet connection — that's expected here, not a code bug.
"""

from dataclasses import dataclass, field


@dataclass
class MCQuestion:
    """One TruthfulQA mc1 question: several choices, exactly one correct."""
    question_id: int
    question: str
    choices: list[str]
    correct_idx: int  # index into `choices` of the single correct answer


def load_truthful_qa_mc1(split: str = "validation", limit: int | None = None
                          ) -> list[MCQuestion]:
    """
    Loads TruthfulQA's multiple_choice config and reshapes it into a flat
    list of MCQuestion. TruthfulQA only ships a `validation` split (no
    train/test) — callers do their own train/test split downstream (see
    linear_probe.py), same as the original papers using this benchmark do.

    `limit`: if set, only load the first N questions — useful for a fast
    smoke test before committing to a full run (817 questions total in mc1).
    """
    from datasets import load_dataset

    # NOTE: the dataset was renamed on the Hub from the bare "truthful_qa"
    # (a legacy loading-script repo with no namespace) to the namespaced
    # "truthfulqa/truthful_qa" (parquet format). The old id now fails with
    # an HfUriError on current huggingface_hub/datasets versions — same
    # migration pattern as e.g. squad -> rajpurkar/squad. Use the new id.
    ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice")[split]
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    questions = []
    for i, row in enumerate(ds):
        mc1 = row["mc1_targets"]
        choices = mc1["choices"]
        labels = mc1["labels"]
        # mc1 is constructed so exactly one label == 1; fail loudly if a
        # dataset version ever violates that assumption instead of silently
        # picking the first match.
        correct_indices = [j for j, lab in enumerate(labels) if lab == 1]
        if len(correct_indices) != 1:
            raise ValueError(
                f"Question {i}: expected exactly 1 correct mc1 choice, "
                f"got {len(correct_indices)}. Dataset format may have "
                f"changed — do not proceed without checking."
            )
        questions.append(MCQuestion(
            question_id=i,
            question=row["question"],
            choices=choices,
            correct_idx=correct_indices[0],
        ))
    return questions


def build_choice_prompt(question: str, choice: str) -> str:
    """
    Formats a (question, choice) pair as a single continuation prompt for
    teacher-forced log-likelihood scoring. Kept deliberately simple (no
    few-shot examples, no special formatting) — matches the zero-shot
    cloze-style scoring TruthfulQA's own mc1 evaluation uses, so probe
    accuracy stays comparable to published baselines.
    """
    return f"Q: {question}\nA: {choice}"