"""Probe-position check for the causal audit (Baseline 1).

Question: the Baseline 1 probe reads the hidden state at the END of
"Q: ...\nA: <chosen answer>", but the causal test patches the state at the
"A:" boundary, before any answer exists. Does the boundary state carry the
probe's signal at all?

This module is self-contained (needs only torch, transformers, sklearn, datasets).
For every question it
  1. scores each choice by mean per-token teacher-forced log-likelihood,
  2. takes the model's top choice, labels it correct/incorrect,
  3. captures post-block hidden states at two positions of
     "Q: {question}\nA: {top choice}":
        "boundary" = last token of the prefix "Q: ...\nA:"
        "end"      = last token of the whole sequence (the original probe's position)
  4. trains the same logistic-regression probe at each position over repeated
     stratified 70/30 splits, plus a permuted-label null and a surface-feature
     baseline (answer length / "no comment" phrasing).

Run:  python -m attn_phase.audit.probe_position --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import re

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

POSITIONS = ("boundary", "end")
NO_COMMENT_RE = re.compile(r"no comment|i have no", re.I)

# Thresholds on boundary-probe mean AUC, fixed BEFORE running (my choice, not a standard).
CARRIES_AUC = 0.65
PARTIAL_AUC = 0.55


# ----------------------------------------------------------------------------
# model plumbing
# ----------------------------------------------------------------------------
def get_blocks(model):
    """Transformer block list for GPT-2 and GPT-NeoX (Pythia) style models."""
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers
    raise ValueError("Unsupported architecture: need GPT-2 or GPT-NeoX style model")


def resolve_layers(model, layers):
    n = len(get_blocks(model))
    out = sorted({int(L) % n for L in layers})
    return out


def model_device(model):
    return next(model.parameters()).device


@torch.no_grad()
def choice_score(model, prefix_ids, cont_ids):
    """Mean per-token log-likelihood of cont_ids given prefix_ids."""
    dev = model_device(model)
    ids = torch.tensor([list(prefix_ids) + list(cont_ids)], device=dev)
    logits = model(ids).logits[0].float()
    start = len(prefix_ids)
    logp = torch.log_softmax(logits[start - 1:-1], dim=-1)
    tgt = ids[0, start:]
    return logp.gather(-1, tgt[:, None]).squeeze(-1).mean().item()


@torch.no_grad()
def capture(model, ids, positions, layers):
    """Post-block hidden states. positions: {name: token index}; layers: resolved
    non-negative block indices. Returns {layer: {name: np.ndarray[d_model]}}."""
    blocks = get_blocks(model)
    store, handles = {}, []
    for li in layers:
        def hook(mod, inp, out, li=li):
            h = out[0] if isinstance(out, tuple) else out
            store[li] = h[0].detach().float().cpu()
        handles.append(blocks[li].register_forward_hook(hook))
    try:
        model(torch.tensor([list(ids)], device=model_device(model)))
    finally:
        for h in handles:
            h.remove()
    return {li: {name: store[li][pos].numpy() for name, pos in positions.items()}
            for li in layers}


# ----------------------------------------------------------------------------
# data + feature extraction
# ----------------------------------------------------------------------------
def load_truthfulqa(n=None):
    from datasets import load_dataset  # import before touching CUDA (arrow.dll crash)
    ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice")["validation"]
    items = [{"question": r["question"],
              "choices": list(r["mc1_targets"]["choices"]),
              "labels": list(r["mc1_targets"]["labels"])} for r in ds]
    return items[:n] if n else items


def extract_features(model, tok, items, layers, log_every=100):
    layers = resolve_layers(model, layers)
    labels, answers, n_tok = [], [], []
    X = {(li, p): [] for li in layers for p in POSITIONS}
    for i, it in enumerate(items):
        prefix_ids = tok(f"Q: {it['question']}\nA:", add_special_tokens=False)["input_ids"]
        conts = [tok(" " + c, add_special_tokens=False)["input_ids"] for c in it["choices"]]
        scores = [choice_score(model, prefix_ids, c) for c in conts]
        top = int(np.argmax(scores))
        ids = list(prefix_ids) + list(conts[top])
        hs = capture(model, ids, {"boundary": len(prefix_ids) - 1, "end": len(ids) - 1}, layers)
        for li in layers:
            for p in POSITIONS:
                X[(li, p)].append(hs[li][p])
        labels.append(int(it["labels"][top] == 1))
        answers.append(it["choices"][top])
        n_tok.append(len(conts[top]))
        if log_every and (i + 1) % log_every == 0:
            print(f"  extracted {i + 1}/{len(items)}", flush=True)
    return {
        "layers": layers,
        "n_blocks": len(get_blocks(model)),
        "labels": np.array(labels),
        "answers": answers,
        "n_tok": np.array(n_tok),
        "X": {k: np.stack(v) for k, v in X.items()},
    }


def save_features(path, feats):
    arrs = {"layers": np.array(feats["layers"]), "n_blocks": np.array(feats["n_blocks"]),
            "labels": feats["labels"],
            "n_tok": feats["n_tok"], "answers": np.array(feats["answers"], dtype=object)}
    for (li, p), v in feats["X"].items():
        arrs[f"X_{li}_{p}"] = v
    np.savez_compressed(path, **arrs)


def load_features(path):
    z = np.load(path, allow_pickle=True)
    layers = [int(x) for x in z["layers"]]
    return {"layers": layers, "n_blocks": int(z["n_blocks"]),
            "labels": z["labels"], "n_tok": z["n_tok"],
            "answers": list(z["answers"]),
            "X": {(li, p): z[f"X_{li}_{p}"] for li in layers for p in POSITIONS}}


# ----------------------------------------------------------------------------
# probes
# ----------------------------------------------------------------------------
def eval_probe(X, y, n_splits=20, C=1.0, seed=0):
    """Mean/std held-out AUC over repeated stratified 70/30 splits."""
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.3, random_state=seed)
    aucs = []
    for tr, te in sss.split(X, y):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
        clf.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.decision_function(X[te])))
    return float(np.mean(aucs)), float(np.std(aucs))


def surface_features(answers, n_tok):
    return np.array([[t, len(a), len(a.split()), int(bool(NO_COMMENT_RE.search(a)))]
                     for a, t in zip(answers, n_tok)], dtype=float)


def verdict(boundary_auc):
    if boundary_auc >= CARRIES_AUC:
        return "CARRIES: the 'A:' state carries the signal, so a donor swap there does change what the probe reads; the Baseline 1 null keeps its (power-limited) meaning"
    if boundary_auc >= PARTIAL_AUC:
        return "PARTIAL: the 'A:' state carries some signal; the donor swap moved only part of what the probe reads"
    return "NOT DETECTED: the 'A:' state does not carry the signal; the donor swap never moved what the probe reads, so the Baseline 1 null says much less"


def run_probes(feats, n_splits=20, C=1.0, seed=0):
    y = feats["labels"]
    rng = np.random.default_rng(seed)
    y_null = rng.permutation(y)
    rows = []
    for li in feats["layers"]:
        for p in POSITIONS:
            X = feats["X"][(li, p)]
            m, s = eval_probe(X, y, n_splits, C, seed)
            nm, _ = eval_probe(X, y_null, n_splits, C, seed)
            rows.append({"layer": li, "position": p, "auc": m, "auc_sd": s, "null_auc": nm})
    sm, ss = eval_probe(surface_features(feats["answers"], feats["n_tok"]), y, n_splits, C, seed)
    return rows, {"auc": sm, "auc_sd": ss}


def perm_test(X, y, n_perm=100, n_splits=5, C=1.0, seed=0):
    """Permutation test for a probe. Observed statistic and every null draw use the
    same procedure (mean held-out AUC over n_splits stratified 70/30 splits), each null
    draw with a fresh label permutation. Returns observed, null mean/sd, one-sided p."""
    obs, _ = eval_probe(X, y, n_splits, C, seed)
    rng = np.random.default_rng(seed + 1)
    null = np.array([eval_probe(X, rng.permutation(y), n_splits, C, seed)[0]
                     for _ in range(n_perm)])
    p = (1 + int((null >= obs).sum())) / (1 + n_perm)
    return {"observed": obs, "null_mean": float(null.mean()), "null_sd": float(null.std()),
            "null_max": float(null.max()), "p": p, "n_perm": n_perm}


def report_perm(res_by_pos, focus_layer):
    print(f"\nPermutation test at layer {focus_layer} (fresh label permutation per draw)")
    for pos, r in res_by_pos.items():
        print(f"  {pos:>8}: observed {r['observed']:.3f} | null {r['null_mean']:.3f} +/- {r['null_sd']:.3f} "
              f"(max {r['null_max']:.3f}) | one-sided p = {r['p']:.4f} (n_perm={r['n_perm']})")


def report(rows, surface, base_acc, n, focus_layer):
    print(f"\nn={n}, model base accuracy = {base_acc:.1%} (Baseline 1 reported 27.7%)")
    print("\nHeld-out AUC, mean +/- sd over repeated 70/30 splits (null = permuted labels)")
    print(f"{'layer':>5} | {'end-of-answer':>16} {'null':>6} | {'A: boundary':>16} {'null':>6}")
    by = {(r["layer"], r["position"]): r for r in rows}
    for li in sorted({r["layer"] for r in rows}):
        e, b = by[(li, "end")], by[(li, "boundary")]
        print(f"{li:>5} | {e['auc']:.3f} +/- {e['auc_sd']:.3f} {e['null_auc']:.3f} | "
              f"{b['auc']:.3f} +/- {b['auc_sd']:.3f} {b['null_auc']:.3f}")
    print(f"\nSurface-feature baseline (answer length + 'no comment' flag): "
          f"AUC {surface['auc']:.3f} +/- {surface['auc_sd']:.3f}")
    b = by[(focus_layer, "boundary")]
    print(f"\nVERDICT at layer {focus_layer} (boundary AUC {b['auc']:.3f}): {verdict(b['auc'])}")
    print(f"(thresholds fixed before running: >= {CARRIES_AUC} carries, "
          f"{PARTIAL_AUC}-{CARRIES_AUC} partial, < {PARTIAL_AUC} not detected)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n", type=int, default=817)
    ap.add_argument("--layers", type=int, nargs="+", default=[1, 2, 3, 4, 6, 9, -1])
    ap.add_argument("--focus-layer", type=int, default=3)
    ap.add_argument("--splits", type=int, default=20)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--perms", type=int, default=0, help="if >0, run a permutation test at the focus layer")
    ap.add_argument("--cache", default=None, help="npz path: load features if present, else save after extraction")
    ap.add_argument("--out", default=None, help="write results JSON here")
    a = ap.parse_args(argv)

    if a.cache and os.path.exists(a.cache):
        print(f"Loading cached features from {a.cache}")
        feats = load_features(a.cache)
    else:
        items = load_truthfulqa(a.n)           # dataset first, then CUDA
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.model)
        model = AutoModelForCausalLM.from_pretrained(a.model).to(a.device).eval()
        ex = items[0]
        pids = tok(f"Q: {ex['question']}\nA:", add_special_tokens=False)["input_ids"]
        print(f"Boundary token check: {tok.decode(pids[-1:])!r} (expected ':'); "
              f"prefix tail: {tok.decode(pids[-6:])!r}")
        feats = extract_features(model, tok, items, a.layers)
        if a.cache:
            save_features(a.cache, feats)
            print(f"Saved features to {a.cache}")

    focus = a.focus_layer % feats["n_blocks"]
    if focus not in feats["layers"]:
        raise SystemExit(f"focus layer {focus} not among extracted layers {feats['layers']}")
    rows, surface = run_probes(feats, a.splits, a.C, a.seed)
    report(rows, surface, float(feats["labels"].mean()), len(feats["labels"]), focus)
    perm = None
    if a.perms > 0:
        perm = {pos: perm_test(feats["X"][(focus, pos)], feats["labels"], a.perms, 5, a.C, a.seed)
                for pos in POSITIONS}
        report_perm(perm, focus)
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"rows": rows, "surface": surface, "perm": perm, "n": len(feats["labels"]),
                       "base_acc": float(feats["labels"].mean())}, f, indent=2)
        print(f"Wrote {a.out}")


if __name__ == "__main__":
    main()