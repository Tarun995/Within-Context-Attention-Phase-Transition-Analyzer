"""
audit/attention_score.py — Baseline 2: attention-concentration score.

An "AttentionScore"-style method: how strongly the model attends to a
small number of positions (concentrated, low-entropy attention) versus
spreading attention broadly (diffuse, high-entropy attention), used
directly as a correctness signal — matching how these methods are
typically used in the literature, as a raw statistic, not a trained
classifier (unlike baseline 1's linear probe, which needed logistic
regression because a hidden-state vector has no natural single-number
reading; a concentration score already IS one number).

WHERE THE SCORE IS COMPUTED, AND WHY IT MATCHES THE CAUSAL TEST'S OWN
PATCH POINT: exactly like the causal test's prefix-boundary position (see
causal_test.py) — the attention distribution FROM the token right after
"Q: {question}\\nA:" back to everything before it. This isn't a
coincidence: it means baseline 2, unlike baseline 1, has its natural
correlational-signal capture point and its causal-test intervention point
be the SAME position, which is what makes the concentration-based causal
test in causal_test_concentration.py possible without inventing a new
capture location.

VALIDATE BEFORE TRUSTING: run tests/test_attention_score.py first — it
checks the entropy math itself (uniform attention -> max entropy, peaked
attention -> near-zero entropy) against a randomly-initialized model, no
network needed.

CRITICAL SETUP REQUIREMENT, FOUND WHILE BUILDING THIS: the model MUST be
loaded with `attn_implementation="eager"` for output_attentions=True to
return real attention weights. Recent `transformers` versions default to
sdpa/flash-attention backends that silently return an EMPTY attentions
tuple even when output_attentions=True is passed — no error, no warning,
just nothing there. A naive script would then compute entropy over an
empty tuple and either crash confusingly or (worse) a naive test would
loop over the empty tuple and pass without checking anything. Always load
with `GPT2LMHeadModel.from_pretrained(model_name, attn_implementation="eager")`
for anything in this module to work correctly — see
run_attention_score_probe.py for the real-run version of this.
"""

import numpy as np
import torch


def compute_attention_concentration(model, tokenizer, question: str,
                                     layer_idx: int, device: str) -> float:
    """
    Runs 'Q: {question}\\nA:' through the model with output_attentions=True
    and returns a single concentration score for `layer_idx`: the NEGATIVE
    mean entropy of the attention distribution at the final prefix
    position (what it attends to, looking backward over the question),
    averaged across all heads.

    Higher score = MORE concentrated (peaked, low-entropy) attention.
    Lower score = MORE diffuse (spread-out, high-entropy) attention.
    """
    prefix = f"Q: {question}\nA:"
    input_ids = tokenizer.encode(prefix, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(input_ids, output_attentions=True)

    # attentions[layer_idx]: [1, n_heads, seq_len, seq_len], post-softmax,
    # already causally masked by the model itself.
    attn = out.attentions[layer_idx]
    dist = attn[0, :, -1, :].clamp_min(1e-12)  # [n_heads, seq_len_so_far]
    entropy_per_head = -(dist * dist.log()).sum(dim=-1)  # [n_heads]
    mean_entropy = entropy_per_head.mean().item()
    return -mean_entropy


def build_concentration_dataset(model, tokenizer, questions: list,
                                 layer_idx: int, device: str,
                                 verbose: bool = True):
    """
    For each question: the model's own predicted answer + correctness
    (reuses baseline 1's activations.predict_answer, so both baselines are
    scored on identical model behavior), plus the attention-concentration
    score at `layer_idx`.

    Returns (scores, labels, questions) as parallel arrays/lists — no
    dataclass wrapper needed here since concentration_pools.py builds its
    own richer records from these.
    """
    from attn_phase.audit.activations import predict_answer

    scores, labels = [], []
    for i, mcq in enumerate(questions):
        result = predict_answer(model, tokenizer, mcq, device)
        score = compute_attention_concentration(model, tokenizer,
                                                  mcq.question, layer_idx,
                                                  device)
        scores.append(score)
        labels.append(int(result["correct"]))

        if verbose and (i + 1) % 100 == 0:
            print(f"  ... {i + 1}/{len(questions)} questions scored")

    return np.array(scores), np.array(labels), list(questions)


def compute_attention_concentration_multilayer(model, tokenizer,
                                                 question: str,
                                                 layer_indices, device: str
                                                 ) -> float:
    """
    Same idea as compute_attention_concentration, but aggregates across
    MULTIPLE layers as well as heads — matching the project plan's
    original wording ("aggregated across heads/layers"), which the
    single-layer version only did half of. Uses ONE forward pass (not one
    per layer), reading every requested layer's attention out of the same
    output_attentions=True call — cheaper than sweeping layers
    individually AND methodologically closer to what was originally
    specified.

    `layer_indices`: which layers to average over (e.g. range(12) for
    every layer of GPT-2 small, or a specific subset).
    """
    prefix = f"Q: {question}\nA:"
    input_ids = tokenizer.encode(prefix, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(input_ids, output_attentions=True)

    per_layer_scores = []
    for layer_idx in layer_indices:
        attn = out.attentions[layer_idx]
        dist = attn[0, :, -1, :].clamp_min(1e-12)
        entropy_per_head = -(dist * dist.log()).sum(dim=-1)
        per_layer_scores.append(-entropy_per_head.mean().item())

    return float(np.mean(per_layer_scores))


def build_concentration_dataset_multilayer(model, tokenizer, questions: list,
                                            layer_indices, device: str,
                                            verbose: bool = True):
    """Multi-layer counterpart to build_concentration_dataset — one
    forward pass per question (not one per layer), aggregating across all
    of `layer_indices` at once."""
    from attn_phase.audit.activations import predict_answer

    scores, labels = [], []
    for i, mcq in enumerate(questions):
        result = predict_answer(model, tokenizer, mcq, device)
        score = compute_attention_concentration_multilayer(
            model, tokenizer, mcq.question, layer_indices, device)
        scores.append(score)
        labels.append(int(result["correct"]))

        if verbose and (i + 1) % 100 == 0:
            print(f"  ... {i + 1}/{len(questions)} questions scored")

    return np.array(scores), np.array(labels), list(questions)


def evaluate_concentration_auc(scores: np.ndarray, labels: np.ndarray):
    """
    Direct AUC of the raw concentration score against correctness — no
    classifier trained, since this is already a single scalar per
    question (unlike baseline 1's hidden-state vector, which needed
    logistic regression to reduce to one number). Returns None if only one
    class is present (AUC undefined), same convention as
    linear_probe.train_and_evaluate_probe.
    """
    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(labels, scores))
    except ValueError:
        return None
