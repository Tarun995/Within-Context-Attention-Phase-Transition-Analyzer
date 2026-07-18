"""
tests/test_patch.py — Sanity checks for the patching mechanism itself.

WHY THIS FILE EXISTS AND MUST PASS FIRST:
patch.py relies on a forward hook returning a modified output tuple from
GPT2Attention.forward(). Whether PyTorch/transformers actually uses that
returned value (rather than the hook running but the original internal
tensor still being used downstream) depends on internals that vary by
library version. If the hook silently no-ops, every P1 pair would report
"no shift" — which looks identical to a genuine null causal result, but
means nothing. This file catches that failure mode BEFORE it can be
mistaken for a real finding.

Two checks:
1. Patching with a different vector must change the logits (mechanism
   actually intervenes).
2. Patching with the SAME vector the model already produced must leave
   logits unchanged (round-trip sanity — confirms the patch is a faithful
   replacement, not introducing some other side effect like dtype/shape
   corruption).

If either check fails, do not trust P1 pilot results — the hook isn't
doing what this module assumes for your installed transformers version.
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from attn_phase.patch import (
    capture_final_position_output,
    run_with_patch,
)

PROMPT = "12 + 34 mod 97 = 46 , 5 + 6 mod 97 = 11 , 20 + 21 mod 97 ="
LAYER_IDX = -1


def _load():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    return tokenizer, model


def test_patch_with_different_vector_changes_logits():
    tokenizer, model = _load()
    device = "cpu"

    baseline_logits = run_with_patch(model, tokenizer, PROMPT, LAYER_IDX,
                                      None, device)

    real_vec = capture_final_position_output(model, tokenizer, PROMPT,
                                               LAYER_IDX, device)
    noise_vec = torch.randn_like(real_vec) * real_vec.std() * 5

    patched_logits = run_with_patch(model, tokenizer, PROMPT, LAYER_IDX,
                                     noise_vec, device)

    assert not torch.allclose(baseline_logits, patched_logits, atol=1e-4), (
        "Patching with a large random vector produced IDENTICAL logits. "
        "This means the hook is not actually affecting the forward pass — "
        "do not trust P1 pilot results until this is fixed. Likely cause: "
        "your installed transformers version doesn't route through this "
        "module's forward() the way patch.py assumes (check attention "
        "implementation / SDPA path)."
    )


def test_patch_with_same_vector_is_a_noop():
    tokenizer, model = _load()
    device = "cpu"

    baseline_logits = run_with_patch(model, tokenizer, PROMPT, LAYER_IDX,
                                      None, device)

    real_vec = capture_final_position_output(model, tokenizer, PROMPT,
                                               LAYER_IDX, device)

    roundtrip_logits = run_with_patch(model, tokenizer, PROMPT, LAYER_IDX,
                                       real_vec, device)

    assert torch.allclose(baseline_logits, roundtrip_logits, atol=1e-3), (
        "Patching with the model's OWN output vector changed the logits. "
        "The patch mechanism is doing something other than a faithful "
        "replacement (check dtype/shape handling in run_with_patch) — do "
        "not trust P1 pilot results until this passes."
    )