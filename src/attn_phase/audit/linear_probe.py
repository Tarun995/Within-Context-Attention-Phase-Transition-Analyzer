"""
audit/linear_probe.py — Baseline 1: hidden-state linear probe.

Reproduces an Azaria & Mitchell-style probe: a linear classifier on a
chosen layer's residual-stream hidden state at the final token, predicting
whether the MODEL'S OWN top-scored answer is correct — not which choice is
correct. This is the framing the project plan specifies (predicting
correctness of the model's output from its internal state), and it's the
framing the later causal test operates on: the causal question is "does
patching this hidden state change whether the model's output shifts from
correct to incorrect or vice versa," which only makes sense against a
correctness-of-own-answer label, not a which-choice-is-right label.

Pipeline per question:
    1. predict_answer() picks the model's own top-choice answer.
    2. Capture the hidden state for (question + that predicted choice).
    3. Label = 1 if the predicted choice was actually correct, else 0.
    4. Train a logistic regression: hidden_state -> label.

VALIDATE BEFORE TRUSTING: compare probe accuracy against Azaria & Mitchell
(2023)'s reported numbers on a comparable setup before running any causal
test on this probe's signal — a badly-reproduced probe makes a causal null
result ambiguous (see CAUSAL_AUDIT_PROJECT_PLAN.md, "Biggest risk").
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class ProbeDataset:
    """X: [n_questions, hidden_dim] hidden states. y: [n_questions] binary
    correctness labels (1 = model's own answer was correct)."""
    X: np.ndarray
    y: np.ndarray
    question_ids: list[int]


def build_probe_dataset(model, tokenizer, questions: list, layer_idx: int,
                         device: str, verbose: bool = True) -> ProbeDataset:
    """
    Runs predict_answer + capture_hidden_state for every question in
    `questions` (a list of data.MCQuestion). Slow — one forward pass per
    choice for scoring (predict_answer) plus one more for hidden-state
    capture, per question. Fine for a few hundred TruthfulQA questions on
    CPU or a laptop GPU; not intended for full-benchmark sweeps at scale.
    """
    from attn_phase.audit.activations import (
        predict_answer, capture_hidden_state,
    )
    from attn_phase.audit.data import build_choice_prompt

    X, y, qids = [], [], []
    for i, mcq in enumerate(questions):
        result = predict_answer(model, tokenizer, mcq, device)
        predicted_choice = mcq.choices[result["predicted_idx"]]
        prompt = build_choice_prompt(mcq.question, predicted_choice)
        hidden_vec = capture_hidden_state(model, tokenizer, prompt,
                                           layer_idx, device)
        X.append(hidden_vec.cpu().numpy())
        y.append(int(result["correct"]))
        qids.append(mcq.question_id)

        if verbose and (i + 1) % 50 == 0:
            print(f"  ... {i + 1}/{len(questions)} questions processed")

    return ProbeDataset(X=np.stack(X), y=np.array(y), question_ids=qids)


def train_and_evaluate_probe(dataset: ProbeDataset, test_size: float = 0.3,
                              random_state: int = 42) -> dict:
    """
    Trains a logistic regression probe with a held-out test split and
    reports accuracy + AUC on the held-out set, plus the label balance
    (a probe on a heavily imbalanced label set — e.g. if the model is
    correct 90% of the time — can look "accurate" while just predicting
    the majority class; check base_rate against accuracy before trusting
    the number).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, roc_auc_score

    X_train, X_test, y_train, y_test = train_test_split(
        dataset.X, dataset.y, test_size=test_size,
        random_state=random_state, stratify=dataset.y,
    )

    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    base_rate = float(np.mean(dataset.y))
    accuracy = float(accuracy_score(y_test, y_pred))
    try:
        auc = float(roc_auc_score(y_test, y_proba))
    except ValueError:
        # Only one class present in y_test (small sample / extreme
        # imbalance) — AUC undefined, report None rather than crash.
        auc = None

    return {
        "accuracy": accuracy,
        "auc": auc,
        "base_rate": base_rate,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "classifier": clf,
    }
