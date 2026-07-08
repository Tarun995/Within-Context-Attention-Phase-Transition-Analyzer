"""
metrics.py — Attention metric computation for the Attention Phase Analyzer.

Computes three families of metrics from a GPT-2 (or any causal LM) attention
tensor extracted during a single forward pass:

    1. ENTROPY-BASED
       Shannon entropy of each token's attention distribution, averaged across
       heads. Captures how focused vs. diffuse attention is at each position.
       Real GPT-2 curves RISE from near-zero (few tokens in context) to a
       plateau as context accumulates — they do NOT decay, which is a common
       assumption that led to the original changepoint detector being broken.

    2. INDUCTION-BASED
       Per-position induction score: how much attention mass lands on the token
       immediately following the most recent earlier occurrence of the current
       token. Canonical induction-head signature (Olsson et al. 2022).
       The oscillation envelope (local amplitude of induction score fluctuations)
       captures whether induction heads are actively engaging.

    3. PLATEAU-BASED  ← primary metric family for Phase C1 and beyond
       Fitted to the actual shape of real GPT-2 entropy curves (rising, not
       falling). Three sub-metrics:
         plateau_onset    : where the entropy curve first reaches 90% of its max
         post_plateau_var : variance after onset (high = oscillatory/noisy)
         has_plateau      : whether the curve genuinely flattens before the end

NOTE on the legacy changepoint detector:
    detect_changepoint() is retained for historical reference but is NOT used
    as a primary metric. It was designed assuming entropy curves decay, and
    was found to snap to the search boundary (position 10) for every task
    when run on real GPT-2 curves, which rise. See inline docstring for full
    history. Do not use it as a signal without first verifying the curve shape.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Low-level curve builders
# ---------------------------------------------------------------------------

def attention_entropy(attn_layer, eps: float = 1e-12) -> np.ndarray:
    """
    Shannon entropy of each token's attention distribution, averaged across
    heads.

    attn_layer : tensor [heads, seq, seq] — one layer, one example,
                 post-softmax attention weights as returned by HuggingFace
                 (output.attentions[layer_idx][0] after squeezing batch dim)
    eps        : small constant for numerical stability

    Returns shape [seq] — one entropy value per token position.

    Interpretation: entropy at position i measures how spread the attention
    of token i is across all preceding tokens. Low = focused (few tokens
    attended to strongly). High = diffuse (attention spread broadly).
    """
    attn = attn_layer.detach().cpu().numpy()
    p = attn + eps
    p = p / p.sum(axis=-1, keepdims=True)
    ent = -(p * np.log(p)).sum(axis=-1)   # [heads, seq]
    return ent.mean(axis=0)               # [seq]


def induction_score(attn_layer, input_ids, eps: float = 1e-12) -> np.ndarray:
    """
    Per-position induction-head score averaged across heads.

    For each position i and head h, measures the attention mass placed on
    the token immediately following the most recent earlier occurrence of
    the token at position i. This is the canonical induction-head signature
    described in Olsson et al. (2022): if token B follows token A earlier
    in the sequence, an induction head at a new occurrence of A will attend
    strongly to that earlier B.

    attn_layer : tensor [heads, seq, seq]
    input_ids  : [seq] token ids for the single example

    Returns shape [seq].
    """
    attn = attn_layer.detach().cpu().numpy()
    ids = (input_ids.detach().cpu().numpy()
           if hasattr(input_ids, "detach") else np.array(input_ids))
    seq_len = attn.shape[-1]
    heads = attn.shape[0]

    scores = np.zeros((heads, seq_len))
    for i in range(1, seq_len):
        cur_tok = ids[i]
        earlier = np.where(ids[:i] == cur_tok)[0]
        if len(earlier) == 0:
            continue
        prev_occurrence = earlier[-1]
        target_pos = prev_occurrence + 1
        if target_pos >= i:
            continue
        scores[:, i] = attn[:, i, target_pos]
    return scores.mean(axis=0)   # [seq]


def oscillation_envelope(score_curve: np.ndarray,
                          window: int = 15) -> np.ndarray:
    """
    Local amplitude (max - min) of a metric curve in a sliding window.
    Captures whether induction-score fluctuations are growing or flat
    across the sequence.

    Returns array same length as input (edge-padded at boundaries).
    """
    n = len(score_curve)
    half = window // 2
    env = np.zeros(n)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        seg = score_curve[lo:hi]
        env[i] = seg.max() - seg.min()
    return env


def smooth(curve: np.ndarray, window: int = 9) -> np.ndarray:
    """Simple moving average with odd window, edge-padded."""
    if window < 2:
        return curve
    kernel = np.ones(window) / window
    pad = window // 2
    padded = np.pad(curve, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


# ---------------------------------------------------------------------------
# PRIMARY metrics: plateau-based
# ---------------------------------------------------------------------------

def plateau_onset(entropy_curve: np.ndarray,
                  threshold_frac: float = 0.90,
                  causal_mask_region: int = 10) -> tuple:
    """
    Position where the entropy curve first reaches threshold_frac of its
    eventual maximum, measured after the causal-mask warm-up region.

    Lower onset_fraction = model spreads attention faster = potentially
    more structured early processing (e.g. lookup's clean sigmoid rise).
    Higher onset_fraction = slower plateau (e.g. hard mod_arith that keeps
    rising through most of the context).

    Returns (onset_pos: int, onset_fraction: float) or (None, None) if the
    curve never reaches the threshold.
    """
    curve = entropy_curve[causal_mask_region:]
    peak = float(np.max(curve))
    if peak < 1e-9:
        return None, None
    threshold = threshold_frac * peak
    hits = np.where(curve >= threshold)[0]
    if len(hits) == 0:
        return None, None
    onset = int(hits[0]) + causal_mask_region
    return onset, onset / len(entropy_curve)


def post_plateau_variance(entropy_curve: np.ndarray,
                           onset_pos,
                           causal_mask_region: int = 10) -> float | None:
    """
    Variance of the entropy curve in the region after plateau onset.

    High variance = oscillatory / sawtooth pattern (seen in failed
    mod_arith tasks where the model keeps revising its attention).
    Low variance = stable flat plateau (seen in lookup and copy).

    Returns None if onset_pos is too close to the end of the sequence to
    measure reliably (fewer than 5 positions remaining).
    """
    if onset_pos is None:
        return None
    start = max(onset_pos, causal_mask_region)
    tail = entropy_curve[start:]
    if len(tail) < 5:
        return None
    return float(np.var(tail))


def has_plateau(entropy_curve: np.ndarray,
                onset_pos,
                tail_frac: float = 0.25,
                slope_threshold: float = 0.003) -> bool:
    """
    Returns True if the entropy curve genuinely flattens after onset_pos,
    False if it keeps rising through the end of context (like the sorting
    task, which never stabilizes in these experiments).

    Method: fit a linear slope to the final tail_frac of the sequence and
    check whether the absolute slope per token is below slope_threshold.

    slope_threshold=0.003 is calibrated for smoothed entropy curves in the
    0-3 nats range over sequences of 100-500 tokens. Adjust if working with
    significantly different sequence lengths or entropy scales.
    """
    if onset_pos is None:
        return False
    n = len(entropy_curve)
    tail_start = int(n * (1 - tail_frac))
    tail_start = max(tail_start, onset_pos + 5)
    if tail_start >= n - 5:
        return False
    tail = entropy_curve[tail_start:]
    xs = np.arange(len(tail), dtype=float)
    slope = float(np.polyfit(xs, tail, 1)[0])
    return abs(slope) < slope_threshold


def entropy_rise_rate(entropy_curve: np.ndarray,
                       causal_mask_region: int = 10,
                       rise_frac: float = 0.25) -> float | None:
    """
    Mean slope of the entropy curve during the early rising phase
    (first rise_frac of the sequence, after causal mask warm-up).

    Higher rise_rate = model quickly distributes attention across context.
    Lower or negative = attention spreads slowly or not at all early on.
    """
    n = len(entropy_curve)
    start = causal_mask_region
    end = max(start + 5, int(n * rise_frac))
    segment = entropy_curve[start:end]
    if len(segment) < 2:
        return None
    xs = np.arange(len(segment), dtype=float)
    return float(np.polyfit(xs, segment, 1)[0])


# ---------------------------------------------------------------------------
# LEGACY: changepoint detector (kept for reference, NOT a primary metric)
# ---------------------------------------------------------------------------

def detect_changepoint(curve: np.ndarray,
                        causal_mask_region: int = 10,
                        min_gap_from_edges: int = 10,
                        detrend_window: int = 51,
                        local_window: int = 20,
                        min_effect_size: float = 6.0) -> tuple:
    """
    Detects a localized structural jump in a position-indexed curve after
    removing the slow-moving baseline trend.

    IMPORTANT — WHY THIS IS LEGACY:
    This detector was designed assuming entropy curves DECAY (high early,
    falling as context accumulates). Real GPT-2 small entropy curves RISE
    from near-zero to a plateau. When run on real curves, the detrending
    removes the rise as "trend," the residuals have no meaningful jump
    structure, and the detector snaps to the search boundary (position 10)
    for every task regardless of solvedness.

    It is retained here because:
    (a) it may be valid for other metrics or model families where the curve
        shape genuinely is decaying
    (b) the bug history is instructive and worth preserving in the codebase

    DO NOT use this as a primary signal without first plotting the raw curve
    and confirming it has the expected decaying shape.

    Returns (inflection_index, effect_size, is_high_confidence).
    Always returns a position (never None for non-degenerate input) so that
    the caller can see the best candidate even when confidence is low.
    is_high_confidence = (effect_size >= min_effect_size), advisory only.
    """
    n = len(curve)
    pad = detrend_window // 2
    padded = np.pad(curve, (pad, pad), mode="edge")
    kernel = np.ones(detrend_window) / detrend_window
    trend = np.convolve(padded, kernel, mode="valid")
    residual = curve - trend

    lo = causal_mask_region
    hi = n - min_gap_from_edges
    if hi <= lo:
        return None, None, False

    resid_std = residual.std()
    if resid_std < 1e-9:
        return None, 0.0, False

    half = local_window // 2
    best_idx, best_score = None, -np.inf
    for t in range(lo, hi):
        left = residual[max(0, t - half):t]
        right = residual[t:min(n, t + half)]
        if len(left) < 3 or len(right) < 3:
            continue
        score = abs(right.mean() - left.mean())
        if score > best_score:
            best_score = score
            best_idx = t

    if best_idx is None:
        return None, None, False

    effect_size = best_score / resid_std
    return best_idx, effect_size, effect_size >= min_effect_size


def envelope_growth_pct(envelope_curve: np.ndarray,
                         causal_mask_region: int = 10,
                         tail_frac: float = 0.25) -> float:
    """
    Percentage growth from early envelope amplitude to tail amplitude.

    LEGACY: was deeply negative for all tasks in Phase 0.5 experiments
    (induction envelope decays after the first ~20 tokens for every task,
    solved or failed). Retained for backward compatibility with Phase 0
    results comparison only.
    """
    n = len(envelope_curve)
    early = envelope_curve[
        causal_mask_region: causal_mask_region + max(5, n // 10)]
    tail_start = int(n * (1 - tail_frac))
    tail = envelope_curve[tail_start:]
    if early.mean() <= 1e-9:
        return float("nan")
    return 100.0 * (tail.mean() - early.mean()) / early.mean()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_all_metrics(attentions, input_ids,
                         layer_idx: int = -1,
                         causal_mask_region: int = 10) -> dict:
    """
    Compute all metrics for a single task from a single forward pass.

    attentions  : tuple of per-layer tensors, each [batch, heads, seq, seq],
                  as returned by HuggingFace with output_attentions=True
    input_ids   : [seq] token ids (single example, no batch dimension)
    layer_idx   : which layer to analyze. -1 = last layer (default).
                  Pass an int in 0..n_layers-1 for a specific layer.
                  The layer sweep (runner.py) calls this in a loop over
                  all layers from the same attentions tuple.
    causal_mask_region : number of early positions to skip (the model's
                  attention is trivially structured here due to the causal
                  mask forcing attention onto very few tokens)

    Returns a dict with:

    PRIMARY (plateau-based, use these for Phase C1 analysis):
        plateau_onset_pos      : int   — absolute token position
        plateau_onset_fraction : float — onset_pos / seq_len
        post_plateau_var       : float — variance after onset
        has_plateau            : bool  — False if curve never flattens
        entropy_rise_rate      : float — slope during early rising phase

    LEGACY (changepoint-based, retain for comparison only):
        inflection_position, inflection_fraction,
        inflection_effect_size, inflection_high_confidence,
        envelope_growth_pct

    RAW CURVES (for plotting — pop these before saving to JSON):
        entropy_curve, entropy_smoothed, induction_curve, envelope_curve
        seq_len
    """
    attn_layer = attentions[layer_idx][0]   # [heads, seq, seq]
    seq_len = attn_layer.shape[-1]

    entropy_curve = attention_entropy(attn_layer)
    entropy_smoothed = smooth(entropy_curve)

    induction_curve = induction_score(attn_layer, input_ids)
    induction_smoothed = smooth(induction_curve)
    env_curve = oscillation_envelope(induction_smoothed)

    # --- primary ---
    onset_pos, onset_frac = plateau_onset(
        entropy_smoothed, causal_mask_region=causal_mask_region)
    ppv = post_plateau_variance(
        entropy_smoothed, onset_pos, causal_mask_region=causal_mask_region)
    plateaus = has_plateau(entropy_smoothed, onset_pos)
    rise_rate = entropy_rise_rate(
        entropy_smoothed, causal_mask_region=causal_mask_region)

    # --- legacy ---
    infl_idx, infl_score, infl_high_conf = detect_changepoint(
        entropy_smoothed, causal_mask_region=causal_mask_region)
    env_growth = envelope_growth_pct(
        env_curve, causal_mask_region=causal_mask_region)
    infl_frac = (infl_idx / seq_len) if infl_idx is not None else None

    return {
        # primary
        "plateau_onset_pos":      onset_pos,
        "plateau_onset_fraction": onset_frac,
        "post_plateau_var":       ppv,
        "has_plateau":            plateaus,
        "entropy_rise_rate":      rise_rate,
        # legacy
        "inflection_position":        infl_idx,
        "inflection_fraction":        infl_frac,
        "inflection_effect_size":     infl_score,
        "inflection_high_confidence": infl_high_conf,
        "envelope_growth_pct":        env_growth,
        # raw curves (pop before JSON save)
        "seq_len":          seq_len,
        "entropy_curve":    entropy_curve,
        "entropy_smoothed": entropy_smoothed,
        "induction_curve":  induction_curve,
        "envelope_curve":   env_curve,
    }