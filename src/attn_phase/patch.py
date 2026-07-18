"""
patch.py — Causal activation patching for Phase P1.

Tests whether the attention layer's OUTPUT at the final prompt position
causally drives task solving, by capturing that vector from a solved task's
forward pass and splicing it into a failed task's forward pass at the same
position (and vice versa), then checking whether next-token prediction
shifts toward / away from the correct answer.

WHY ATTN_OUTPUT, NOT RAW ATTENTION WEIGHTS:
The project's existing metrics (post_plateau_var etc.) are computed from
post-softmax attention weights. Directly overwriting those weights mid
forward-pass requires patching GPT2Attention's internal `_attn` computation,
whose exact signature has changed across `transformers` library versions
(attention-implementation refactors, SDPA/flash paths, etc.) — depending on
it is fragile and version-specific in a way I can't verify against your
exact installed version.

Patching the attention module's OUTPUT (attn_output — the value-weighted,
output-projected vector that gets added to the residual stream, i.e.
"hook_attn_out" / "z" in TransformerLens terminology) is:
  (a) a standard, well-established causal-patching target in the
      mech-interp literature,
  (b) directly downstream of the attention weights the project measures, so
      it's a faithful test of whether that layer's attention-driven
      computation matters causally, and
  (c) implementable with a plain forward hook that only touches the
      module's public return value — robust to internal implementation
      changes.

WHY THE FINAL PROMPT POSITION:
That position's hidden state is what next-token prediction is conditioned
on. Patching it directly tests the question P1 asks: does this component's
activation drive whether the model gets the answer right.

WHAT THIS DOES NOT TEST: whether patching mid-sequence positions (where
post_plateau_var's variance is actually measured, not just the final
token) shows a different or larger effect. That's a natural extension if
this pilot finds signal — intentionally not built here, per the roadmap's
own guardrail to keep the pilot small before scaling.

BEFORE TRUSTING ANY RESULT FROM THIS FILE: run tests/test_patch.py first.
It verifies the hook mechanism actually changes the forward pass rather
than silently no-op'ing — see that file's docstring for why this check
matters and can't be skipped.
"""

import csv
import os
from datetime import datetime, timezone

import torch

from attn_phase.tasks import answer_matches, extract_model_answer


MANIFEST_COLUMNS = [
    "pair_id", "task_family", "layer", "position",
    "direction", "seed", "pre_patch_output_correct",
    "post_patch_output_correct", "shift_observed", "timestamp",
]


# ---------------------------------------------------------------------------
# Manifest (P1-infra) — resumable, crash-safe experiment tracking
# ---------------------------------------------------------------------------

def load_manifest(path: str) -> dict:
    """Returns {pair_id: row_dict} for already-completed pairs.
    Empty dict if the manifest doesn't exist yet (first run)."""
    if not os.path.exists(path):
        return {}
    rows = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows[row["pair_id"]] = row
    return rows


def append_manifest_row(path: str, row: dict) -> None:
    """Appends one row and flushes to disk immediately — called after
    EVERY pair, not batched, so a killed process never loses more than
    one pair's worth of work."""
    file_exists = os.path.exists(path)
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def manifest_bool(value) -> bool:
    """CSV round-trips booleans as the strings 'True'/'False', not real
    Python bools. Normalize explicitly wherever a manifest value is
    checked — this exact bug is documented in the project's own roadmap
    appendix as something that silently undercounts fresh results if
    missed."""
    return value in (True, "True", "true", "1")


def check_coverage(manifest_rows: dict, expected_pair_ids: list) -> tuple:
    """Returns (is_complete, missing_pair_ids). A pair only counts as done
    if it has a real recorded outcome, not just a row existing — catches a
    batch that silently short-circuited partway through."""
    missing = []
    for pid in expected_pair_ids:
        row = manifest_rows.get(pid)
        if row is None or row.get("shift_observed", "") == "":
            missing.append(pid)
    return len(missing) == 0, missing


# ---------------------------------------------------------------------------
# Activation capture / patch hooks
# ---------------------------------------------------------------------------

def _get_attn_module(model, layer_idx: int):
    return model.transformer.h[layer_idx].attn


def capture_final_position_output(model, tokenizer, prompt: str,
                                   layer_idx: int, device: str) -> torch.Tensor:
    """
    Runs one forward pass and captures the attention module's output vector
    at the FINAL token position of the prompt, for the given layer.
    Returns a detached [hidden_dim] tensor.
    """
    captured = {}

    def hook(module, inputs, output):
        attn_output = output[0] if isinstance(output, tuple) else output
        captured["vec"] = attn_output[0, -1, :].detach().clone()

    handle = _get_attn_module(model, layer_idx).register_forward_hook(hook)
    try:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            model(input_ids)
    finally:
        handle.remove()

    if "vec" not in captured:
        raise RuntimeError(
            f"Hook on layer {layer_idx} never fired — check that "
            f"model.transformer.h[{layer_idx}].attn exists for this model "
            f"and returns a tuple from forward()."
        )
    return captured["vec"]


def run_with_patch(model, tokenizer, prompt: str, layer_idx: int,
                    patch_vec, device: str) -> torch.Tensor:
    """
    Runs one forward pass on `prompt`. If patch_vec is given (a
    [hidden_dim] tensor), overwrites the attention module's output at the
    FINAL token position with patch_vec before it's added to the residual
    stream. Pass patch_vec=None for an unpatched baseline pass (still goes
    through the hook, so timing/behavior is identical either way).

    Returns logits for the next token, shape [vocab_size].
    """
    def hook(module, inputs, output):
        if patch_vec is None:
            return output
        attn_output = output[0] if isinstance(output, tuple) else output
        attn_output = attn_output.clone()
        attn_output[0, -1, :] = patch_vec.to(attn_output.dtype)
        if isinstance(output, tuple):
            return (attn_output,) + tuple(output[1:])
        return attn_output

    handle = _get_attn_module(model, layer_idx).register_forward_hook(hook)
    try:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(input_ids)
        return out.logits[0, -1, :].detach()
    finally:
        handle.remove()


def expected_first_token_id(tokenizer, expected_answer: str) -> int:
    """
    The token id the model would need to output first to be on track for
    the correct answer. Encoded WITH a leading space — prompts here end
    right after "=" with no trailing space, and GPT-2's BPE tokenizes
    " 53" differently from "53", so this must match how the model would
    naturally continue.
    """
    ids = tokenizer.encode(" " + expected_answer.strip())
    return ids[0]


def next_token_matches_expected(logits: torch.Tensor, tokenizer,
                                 expected_answer: str) -> bool:
    """
    Greedy-picks the argmax next token and compares its TOKEN ID (not
    decoded string) to the first token of the expected answer. Comparing
    ids avoids whitespace/decoding edge cases. This is a next-token proxy
    for correctness — sufficient for "did the patch shift the output
    toward correct," which is the question P1 asks. Full multi-token
    correctness (answer_matches from tasks.py) is used for the baseline
    solved/failed classification that selects donor/recipient pairs, since
    that's what the rest of the pipeline already trusts.
    """
    predicted_id = int(torch.argmax(logits).item())
    return predicted_id == expected_first_token_id(tokenizer, expected_answer)


def score_full_generation(model, tokenizer, task: dict, device: str) -> bool:
    """Baseline correctness using the SAME full-generation scoring the rest
    of the pipeline trusts — used to classify pool instances as solved vs.
    failed before pairing them for patching."""
    input_ids = tokenizer.encode(task["prompt"], return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            input_ids, max_new_tokens=10, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(out[0], skip_special_tokens=True)
    continuation = extract_model_answer(generated, task["prompt"])
    solved, _ = answer_matches(continuation, task["expected_answer"])
    return solved


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Post-plateau RANGE patching (extension pilot — patches every position
# from plateau onset to end, not just the final token, since that's the
# actual span post_plateau_var is measured over)
# ---------------------------------------------------------------------------

def get_plateau_onset_position(model, tokenizer, prompt: str, layer_idx: int,
                                device: str, causal_mask_region: int = 10):
    """
    Runs a forward pass and computes THIS instance's own plateau onset
    position, using the exact same attention_entropy -> smooth ->
    plateau_onset pipeline metrics.py uses for post_plateau_var. Returns
    (onset_pos or None, seq_len). onset_pos is None if the curve never
    reaches the plateau threshold — such instances are excluded from
    range-patching pairs, since there's no well-defined post-plateau region
    to patch.
    """
    from attn_phase.metrics import attention_entropy, smooth, plateau_onset

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(input_ids, output_attentions=True)
    if not out.attentions:
        raise RuntimeError(
            "model(...).attentions came back empty even with "
            "output_attentions=True passed at call time. This model was "
            "likely loaded without output_attentions=True at "
            "from_pretrained() time — some attention implementations "
            "(e.g. SDPA, the transformers-library default on newer "
            "versions) only return attention weights if that flag is set "
            "at load time, not just at call time. Fix: load with "
            "GPT2LMHeadModel.from_pretrained(model_name, "
            "output_attentions=True), matching runner.py's existing pattern."
        )
    attn_layer = out.attentions[layer_idx][0]
    entropy_curve = attention_entropy(attn_layer)
    entropy_smoothed = smooth(entropy_curve)
    onset_pos, _ = plateau_onset(entropy_smoothed,
                                  causal_mask_region=causal_mask_region)
    return onset_pos, input_ids.shape[1]


def capture_range_outputs(model, tokenizer, prompt: str, layer_idx: int,
                           device: str) -> torch.Tensor:
    """
    Runs one forward pass and captures the attention module's output at
    EVERY token position (not just the final one), for the given layer.
    Returns a [seq_len, hidden_dim] tensor. Callers slice out whichever
    position range they need (e.g. positions[onset_pos:]).
    """
    captured = {}

    def hook(module, inputs, output):
        attn_output = output[0] if isinstance(output, tuple) else output
        captured["all"] = attn_output[0].detach().clone()  # [seq, hidden]

    handle = _get_attn_module(model, layer_idx).register_forward_hook(hook)
    try:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            model(input_ids)
    finally:
        handle.remove()

    if "all" not in captured:
        raise RuntimeError(f"Hook on layer {layer_idx} never fired.")
    return captured["all"]


def run_with_range_patch(model, tokenizer, prompt: str, layer_idx: int,
                          patch_map: dict, device: str) -> torch.Tensor:
    """
    Runs one forward pass on `prompt`. patch_map is {position: [hidden_dim]
    tensor}; every position present gets its attention-layer output
    overwritten before the residual add. Positions beyond this prompt's own
    sequence length are silently skipped (can happen if donor and recipient
    differ slightly in length). Pass an empty dict for an unpatched
    baseline pass — still goes through the hook for identical behavior.

    Returns logits for the next token, shape [vocab_size].
    """
    def hook(module, inputs, output):
        if not patch_map:
            return output
        attn_output = output[0] if isinstance(output, tuple) else output
        attn_output = attn_output.clone()
        seq_len = attn_output.shape[1]
        for pos, vec in patch_map.items():
            if 0 <= pos < seq_len:
                attn_output[0, pos, :] = vec.to(attn_output.dtype)
        if isinstance(output, tuple):
            return (attn_output,) + tuple(output[1:])
        return attn_output

    handle = _get_attn_module(model, layer_idx).register_forward_hook(hook)
    try:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(input_ids)
        return out.logits[0, -1, :].detach()
    finally:
        handle.remove()


def build_post_plateau_patch_map(donor_outputs: torch.Tensor,
                                  start_pos: int, end_pos: int) -> dict:
    """Slices a captured [seq_len, hidden_dim] tensor into a
    {position: vector} map for positions [start_pos, end_pos)."""
    return {pos: donor_outputs[pos] for pos in range(start_pos, end_pos)}