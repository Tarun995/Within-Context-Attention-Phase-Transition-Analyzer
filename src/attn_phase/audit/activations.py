"""
audit/activations.py — Hidden-state capture, patching, and log-likelihood
scoring for the causal audit, generalized from patch.py to work on arbitrary
QA prompts (TruthfulQA/TriviaQA) instead of only the synthetic mod_arith
task family.

WHY A BLOCK-LEVEL FORWARD HOOK, NOT `output_hidden_states=True`:
Same reasoning patch.py already documents for attn_output: a plain
`register_forward_hook` on `model.transformer.h[layer_idx]` only touches
the module's public return value, so it's robust to internal implementation
changes across `transformers` versions. Relying on `output_hidden_states`
instead would work too, but would introduce a SECOND, differently-shaped
API surface alongside patch.py's existing attn_output hook — this file
keeps the same hook pattern for consistency, and because the exact indexing
of the `hidden_states` output tuple (whether the final entry reflects the
model's closing LayerNorm or not) is not guaranteed to match your intuition
across versions; hooking the block directly avoids depending on it.

This module captures the RESIDUAL-STREAM output of a transformer block
(before that block's contribution is summed further downstream) — the
standard target for an Azaria & Mitchell-style hidden-state linear probe.
This is a different capture point from patch.py's attn_output (the
attention sub-module's output within a block); the two are complementary,
not duplicates: patch.py tests one component's causal role, this module
sets up the probe half of the causal audit, with its OWN patch function
below for the later causal test on the probe's signal.

BEFORE TRUSTING ANY RESULT FROM THIS FILE: run
tests/test_audit_activations.py first, same reason as patch.py's own
warning — a silently no-op hook looks identical to a genuine null result.
"""

import torch


def _get_block_module(model, layer_idx: int):
    from attn_phase.audit.model_adapter import get_block_module
    return get_block_module(model, layer_idx)


# ---------------------------------------------------------------------------
# Hidden-state capture (probe input) — operates on token ids directly so it
# has no tokenizer dependency; wrappers below add tokenizer convenience.
# ---------------------------------------------------------------------------

def capture_hidden_state_from_ids(model, input_ids: torch.Tensor,
                                   layer_idx: int) -> torch.Tensor:
    """
    Runs one forward pass on already-tokenized `input_ids` ([1, seq_len])
    and captures the given block's residual-stream output at the FINAL
    token position. Returns a detached [hidden_dim] tensor — the probe's
    input feature for this prompt.
    """
    captured = {}

    def hook(module, inputs, output):
        block_out = output[0] if isinstance(output, tuple) else output
        captured["vec"] = block_out[0, -1, :].detach().clone()

    handle = _get_block_module(model, layer_idx).register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(input_ids)
    finally:
        handle.remove()

    if "vec" not in captured:
        raise RuntimeError(
            f"Hook on block {layer_idx} never fired — check that "
            f"model.transformer.h[{layer_idx}] exists for this model."
        )
    return captured["vec"]


def capture_hidden_state(model, tokenizer, text: str, layer_idx: int,
                          device: str) -> torch.Tensor:
    """Tokenizer-convenience wrapper around capture_hidden_state_from_ids."""
    input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
    return capture_hidden_state_from_ids(model, input_ids, layer_idx)


# ---------------------------------------------------------------------------
# Patching (for the later causal test — Week 4-6 of the plan) — same
# donor -> recipient splice pattern as patch.py's run_with_patch, retargeted
# at the block's residual-stream output instead of the attention output.
# ---------------------------------------------------------------------------

def run_with_hidden_patch(model, input_ids: torch.Tensor, layer_idx: int,
                           patch_vec, device: str, position=-1,
                           return_attentions: bool = False):
    """
    Runs one forward pass on `input_ids`. If patch_vec is given (a
    [hidden_dim] tensor), overwrites the block's residual-stream output at
    `position` with patch_vec before it propagates further through the
    rest of the network. Pass patch_vec=None for an unpatched baseline
    pass.

    `position` can be:
      - an int (default -1): patch a single token position.
      - a slice (e.g. slice(0, 6)): patch every position in the range
        with the SAME patch_vec, broadcast across all of them. Used for
        range-patching, where a single-position patch leaves other
        positions the model can still read from untouched (see
        causal_test.py's docstring on single- vs range-patching).
      - a list of ints: same broadcasting, for non-contiguous positions.

    `position` matters a lot for causal testing: patching the very last
    token of a sequence only affects prediction of the NEXT (not-yet-seen)
    token — it has no effect on how earlier tokens in the same sequence
    were processed, because of causal (autoregressive) masking. To
    influence a downstream computation that depends on tokens AFTER the
    patch point (e.g. patching right after "Q: ...\\nA:" so it influences
    every subsequent answer token's score), pass the specific position(s)
    you want patched, not -1.

    `return_attentions`: if True, also requests output_attentions=True on
    the forward pass and returns (logits, attentions) instead of just
    logits — used by causal_test_concentration.py to check whether a
    patch shifted the recipient's OWN attention-concentration pattern,
    not just its output correctness. Default False keeps the original
    single-return-value contract for existing callers.

    Returns full logits, shape [seq_len, vocab_size] (not just the last
    position) so callers that need scores for every position — e.g.
    teacher-forced choice scoring — don't need a second forward pass.
    """
    def hook(module, inputs, output):
        if patch_vec is None:
            return output
        block_out = output[0] if isinstance(output, tuple) else output
        block_out = block_out.clone()
        block_out[0, position, :] = patch_vec.to(block_out.dtype)
        if isinstance(output, tuple):
            return (block_out,) + tuple(output[1:])
        return block_out

    handle = _get_block_module(model, layer_idx).register_forward_hook(hook)
    try:
        with torch.no_grad():
            out = model(input_ids.to(device), output_attentions=return_attentions)
        logits = out.logits[0].detach()
        if return_attentions:
            attentions = tuple(a.detach() for a in out.attentions)
            return logits, attentions
        return logits
    finally:
        handle.remove()


# ---------------------------------------------------------------------------
# Choice scoring by teacher-forced log-likelihood (zero-shot cloze scoring,
# matching TruthfulQA mc1's own evaluation convention)
# ---------------------------------------------------------------------------

def score_choice_loglik(model, tokenizer, question: str, choice: str,
                         device: str) -> float:
    """
    Teacher-forces `choice` after `question` and returns the AVERAGE
    per-token log-probability of the choice tokens (length-normalized, so
    longer choices aren't penalized just for having more tokens — standard
    practice for cloze-style multiple-choice scoring).
    """
    from attn_phase.audit.data import build_choice_prompt

    prefix = f"Q: {question}\nA:"
    full = build_choice_prompt(question, choice)

    prefix_ids = tokenizer.encode(prefix, return_tensors="pt").to(device)
    full_ids = tokenizer.encode(full, return_tensors="pt").to(device)

    n_prefix = prefix_ids.shape[1]
    n_choice_tokens = full_ids.shape[1] - n_prefix
    if n_choice_tokens <= 0:
        raise ValueError(
            f"Choice {choice!r} tokenized to 0 new tokens after the "
            f"question prefix — check formatting."
        )

    with torch.no_grad():
        out = model(full_ids)
    logits = out.logits[0]  # [seq_len, vocab]
    log_probs = torch.log_softmax(logits, dim=-1)

    # Token at position i is predicted by logits at position i-1.
    total_logprob = 0.0
    for pos in range(n_prefix, full_ids.shape[1]):
        target_id = full_ids[0, pos]
        total_logprob += log_probs[pos - 1, target_id].item()

    return total_logprob / n_choice_tokens


def predict_answer(model, tokenizer, mcq, device: str) -> dict:
    """
    Scores every choice in `mcq` by log-likelihood, picks the argmax as the
    model's predicted answer, and reports whether that matches the labeled
    correct choice.

    Returns:
        {
          "predicted_idx": int,
          "correct": bool,
          "choice_logliks": [float, ...],   # same order as mcq.choices
        }
    """
    logliks = [
        score_choice_loglik(model, tokenizer, mcq.question, choice, device)
        for choice in mcq.choices
    ]
    predicted_idx = max(range(len(logliks)), key=lambda i: logliks[i])
    return {
        "predicted_idx": predicted_idx,
        "correct": predicted_idx == mcq.correct_idx,
        "choice_logliks": logliks,
    }
