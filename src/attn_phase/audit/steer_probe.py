"""Probe-direction steering test: is the probe's direction a lever on the model's choice?

Baseline 1 showed a hidden-state probe predicts whether GPT-2's top-scored TruthfulQA
answer is correct (AUC ~0.78 at the end of the answer), but patching the "A:" state did
nothing. That probe is correlational. This script asks the causal question directly:
if we push the residual stream along the probe's direction while the model scores each
answer choice, do its choices move toward correct answers more than they do for random
directions of the same size?

Design
  * Probe: StandardScaler + logistic regression on the END-of-answer state (layer L),
    exactly as probe_position.py. Direction u = unit vector of the probe weights mapped
    back to raw activation space (w / scaler.scale_). +u = "predicts correct".
  * Cross-fitting: 5 folds. The probe used on a question never saw that question, so
    steering is not evaluated on the data the direction was fit on.
  * Intervention: add sign * alpha * s * u to the post-block hidden state at layer L, at
    the ANSWER tokens of every choice (default) while the choice is scored, then recompute
    all choice scores. s = std of the training states' projection on u, so alpha is in
    "typical SDs along the probe direction". Adding at the last token alone would change
    nothing (no scored token follows it), which is why the vector goes on the answer tokens.
  * Controls (a "replicate" = the whole pipeline with u replaced by a random direction,
    redrawn independently in every fold, SAME L2 norm as the probe step):
      iso : isotropic random unit directions
      cov : directions drawn from the training states' own covariance (variance-matched;
            guards against "the probe direction just lives in high-variance dimensions")
      surf: directions from probes trained on SURFACE labels (answer says "no comment",
            answer is longer than the median). These are real feature directions, not
            correctness. Random directions cannot tell "a correctness lever" from "any
            feature direction the model happens to read" (GPT-2 ties its output embedding to
            its input embedding, so a direction that encodes the current token's identity
            raises that token's next-token logit). The probe must move the margin more
            than the surface directions do, or the effect is not specific to correctness.
  * Outcome: per-question margin = best correct score - best incorrect score (mean
    per-token log-likelihood, the same score as everywhere else). mc1 accuracy is
    mean(margin > 0). Margin is the continuous version, so it has more power.
  * Statistic: slope S = mean(margin at +alpha) - mean(margin at -alpha). Generic
    disruption hits both signs about equally and cancels; a lever shows up as S != 0.
    S > 0 means pushing toward "correct" raises the correct answers' margin.

Known limits: the probe is fit on the state at the LAST answer token but is added at every
answer token (a distribution shift); with where="answer" the state at "A:" is not steered,
so the first answer token's log-probability is unchanged.

Fixed BEFORE running (my choices, not standards); everything else is exploratory
  primary layer = 3 (the layer the patching experiments used), primary alpha = first value
  of --alphas (default 2), where = answer, statistic = margin slope S,
  claim a LEVER only if the probe's |S| beats the random |S| null in BOTH families at
  p <= 0.05 (p = (1 + #{null |S| >= probe |S|}) / (1 + n_random)).

Run:
  python -m attn_phase.audit.steer_probe --features probe_pos_gpt2.npz --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from attn_phase.audit import probe_position as pp

P_THRESH = 0.05
SIGNS = (+1, -1)


# ----------------------------------------------------------------------------
# scoring with an optional additive intervention
# ----------------------------------------------------------------------------
@torch.no_grad()
def score_choices(model, prefix_ids, conts, layer=None, vec=None, where="answer"):
    """Mean per-token log-likelihood of every choice, in one batch (right-padded; the
    causal mask means padding cannot affect earlier positions). If vec is given it is
    added to the post-block hidden state at `layer` on the chosen positions:
    where="answer" -> only the choice tokens; where="all" -> prefix and choice."""
    assert all(len(c) > 0 for c in conts), "empty choice"
    dev = pp.model_device(model)
    P = len(prefix_ids)
    lens = [P + len(c) for c in conts]
    B, T = len(conts), max(lens)
    ids = torch.zeros((B, T), dtype=torch.long)
    for i, c in enumerate(conts):
        ids[i, :lens[i]] = torch.tensor(list(prefix_ids) + list(c))
    ids = ids.to(dev)
    handle = None
    if vec is not None:
        start = P if where == "answer" else 0
        m = torch.zeros((B, T, 1), device=dev)
        for i, L in enumerate(lens):
            m[i, start:L] = 1.0
        delta = m * torch.as_tensor(np.asarray(vec), dtype=torch.float32, device=dev)

        def hook(mod, inp, out):
            if isinstance(out, tuple):
                return (out[0] + delta.to(out[0].dtype),) + tuple(out[1:])
            return out + delta.to(out.dtype)

        handle = pp.get_blocks(model)[layer].register_forward_hook(hook)
    try:
        logits = model(ids).logits.float()
    finally:
        if handle is not None:
            handle.remove()
    out = []
    for i, L in enumerate(lens):
        lp = torch.log_softmax(logits[i, P - 1:L - 1], dim=-1)
        out.append(lp.gather(-1, ids[i, P:L, None]).squeeze(-1).mean().item())
    return np.array(out)


@torch.no_grad()
def score_choices_multi(model, prefix_ids, conts, layer, vecs, where="answer", max_seqs=128):
    """Scores of every choice under EVERY steering vector at once.

    vecs: [V, d]. Returns [V, n_choices] (mean per-token log-likelihood, same score as
    score_choices). The V x n_choices sequences go through the model in chunks of about
    max_seqs, and the output head is applied only at the answer positions, so one call does
    the work of V*n_choices separate forward passes. This is what makes the full run cheap.
    """
    vecs = np.asarray(vecs, dtype=np.float32)
    V, B = len(vecs), len(conts)
    assert B > 0 and all(len(c) > 0 for c in conts), "empty choice"
    dev = pp.model_device(model)
    P = len(prefix_ids)
    lens = [P + len(c) for c in conts]
    T, A = max(lens), max(len(c) for c in conts)
    base = torch.zeros((B, T), dtype=torch.long)
    pos = torch.zeros((B, A), dtype=torch.long)      # position whose output predicts answer token j
    valid = torch.zeros((B, A))
    for i, c in enumerate(conts):
        base[i, :lens[i]] = torch.tensor(list(prefix_ids) + list(c))
        pos[i, :len(c)] = torch.arange(P - 1, P - 1 + len(c))
        valid[i, :len(c)] = 1.0
    base, pos, valid = base.to(dev), pos.to(dev), valid.to(dev)
    tgt = base.gather(1, (pos + 1).clamp(max=T - 1))                       # [B, A]
    start = P if where == "answer" else 0
    mask = torch.zeros((B, T, 1), device=dev)
    for i, L in enumerate(lens):
        mask[i, start:L] = 1.0
    head, block = model.get_output_embeddings(), pp.get_blocks(model)[layer]
    per = max(1, max_seqs // B)
    out = np.zeros((V, B))
    for v0 in range(0, V, per):
        vc = torch.as_tensor(vecs[v0:v0 + per], device=dev)               # [v, d]
        v = len(vc)
        delta = (mask[None] * vc[:, None, None, :]).reshape(v * B, T, -1)

        def hook(mod, inp, o):
            if isinstance(o, tuple):
                return (o[0] + delta.to(o[0].dtype),) + tuple(o[1:])
            return o + delta.to(o.dtype)

        handle = block.register_forward_hook(hook)
        try:
            hs = model.base_model(base.repeat(v, 1)).last_hidden_state     # [v*B, T, d]
        finally:
            handle.remove()
        hg = hs.gather(1, pos.repeat(v, 1)[..., None].expand(-1, -1, hs.shape[-1]))
        lp = torch.log_softmax(head(hg).float(), dim=-1)
        tl = lp.gather(-1, tgt.repeat(v, 1)[..., None]).squeeze(-1)        # [v*B, A]
        vr = valid.repeat(v, 1)
        sc = (tl * vr).sum(1) / vr.sum(1)
        out[v0:v0 + v] = sc.reshape(v, B).float().cpu().numpy()
    return out


def margin_top(scores, labels):
    """(margin, top-is-correct, top index). margin = best correct - best incorrect."""
    labels = np.asarray(labels)
    top = int(np.argmax(scores))
    m = float(scores[labels == 1].max() - scores[labels == 0].max())
    return m, bool(labels[top] == 1), top


# ----------------------------------------------------------------------------
# directions
# ----------------------------------------------------------------------------
def probe_direction(X, y, C=1.0):
    """Unit probe direction in RAW activation space (+ = predicts correct) and the std of
    the training states' projection on it."""
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000)).fit(X, y)
    w = clf[1].coef_[0] / clf[0].scale_
    u = w / np.linalg.norm(w)
    s = float(np.std((X - X.mean(0)) @ u))
    return u, s


def surface_labels(feats):
    """Binary surface labels per question; a label is dropped if either class has < 20 rows."""
    ans, n_tok = feats["answers"], np.asarray(feats["n_tok"])
    cand = {"surf_nocomment": np.array([int(bool(pp.NO_COMMENT_RE.search(a))) for a in ans]),
            "surf_length": (n_tok > np.median(n_tok)).astype(int)}
    return {k: v for k, v in cand.items() if min(v.sum(), len(v) - v.sum()) >= 20}


def iso_direction(d, rng):
    v = rng.standard_normal(d)
    return v / np.linalg.norm(v)


def cov_direction(X, rng):
    """Random direction with the covariance of the states: v = Xc^T z, z ~ N(0, I)."""
    v = (X - X.mean(0)).T @ rng.standard_normal(len(X))
    return v / np.linalg.norm(v)


# ----------------------------------------------------------------------------
# the experiment
# ----------------------------------------------------------------------------
def _meta(layer, N, K, alphas, folds, seed, n_random, where, C):
    return np.array([layer, N, K, len(alphas), folds, seed, n_random, int(where == "all"), C], float)


def run_steering(model, tok, items, feats, layer, alphas, n_random=30, where="answer",
                 folds=5, C=1.0, seed=0, log=print, ckpt=None, stop_after=None, max_seqs=128):
    """Run the experiment fold by fold. With ckpt set, progress is saved after every fold and
    a rerun resumes from it (identical results: directions depend only on seed and fold).
    stop_after=n runs at most n NEW folds, then returns None so you can continue later."""
    layer = pp.resolve_layers(model, [layer])[0]
    if (layer, "end") not in feats["X"]:
        raise SystemExit(f"layer {layer} not in the feature file (has {feats['layers']})")
    X, y = feats["X"][(layer, "end")], feats["labels"]
    N, d = X.shape
    if len(items) != N:
        raise SystemExit(f"{len(items)} items but {N} cached feature rows; use the same --n")

    prefixes, conts, labs = [], [], []
    for it in items:
        prefixes.append(tok(f"Q: {it['question']}\nA:", add_special_tokens=False)["input_ids"])
        conts.append([tok(" " + c, add_special_tokens=False)["input_ids"] for c in it["choices"]])
        labs.append(np.array(it["labels"]))

    base_margin, base_ok, base_top = np.zeros(N), np.zeros(N, bool), np.zeros(N, int)
    for i in range(N):
        base_margin[i], base_ok[i], base_top[i] = margin_top(
            score_choices(model, prefixes[i], conts[i]), labs[i])
    mism = int((base_ok != y.astype(bool)).sum())
    log(f"Unsteered: accuracy {base_ok.mean():.1%}, mean margin {base_margin.mean():+.4f}; "
        f"{mism} label mismatches vs the feature file")
    if mism > max(2, N // 200):
        raise SystemExit("the feature file does not match this model/data (label mismatch); "
                         "rebuild it with probe_position.py using the same --model and --n")

    surf = surface_labels(feats)
    names = (["probe"] + [f"iso{r}" for r in range(n_random)] + [f"cov{r}" for r in range(n_random)]
             + list(surf))
    A, K = len(alphas), len(names)
    margin = np.zeros((K, A, 2, N))
    ok = np.zeros((K, A, 2, N), bool)
    flip = np.zeros((K, A, 2, N), bool)
    s_fold, done = {}, set()
    state_norm = float(np.linalg.norm(X, axis=1).mean())
    meta = _meta(layer, N, K, alphas, folds, seed, n_random, where, C)

    if ckpt and os.path.exists(ckpt):
        z = np.load(ckpt, allow_pickle=False)
        if not (np.array_equal(z["meta"], meta) and np.array_equal(z["alphas"], np.array(alphas, float))):
            raise SystemExit(f"{ckpt} was made with different settings; delete it or pass the same "
                             f"--layer/--alphas/--random/--folds/--seed/--where/--C")
        margin, ok, flip = z["margin"], z["ok"], z["flip"]
        done = {int(f) for f in z["done"]}
        s_fold = {int(f): float(v) for f, v in zip(z["done"], z["s_done"])}
        log(f"Resuming from {ckpt}: folds done {sorted(done)} of {folds}")

    coef_a = np.array(alphas, float)[:, None] * np.array(SIGNS, float)[None, :]      # [A, 2]
    new_folds = 0
    skf = StratifiedKFold(folds, shuffle=True, random_state=seed)
    splits = list(skf.split(X, y))
    for f, (tr, te) in enumerate(splits):
        if f in done:
            continue
        if stop_after is not None and new_folds >= stop_after:
            break
        t0 = time.time()
        u_p, s = probe_direction(X[tr], y[tr], C)
        s_fold[f] = s
        rng = np.random.default_rng([seed, f])
        dirs = [u_p] + [iso_direction(d, rng) for _ in range(n_random)] \
                     + [cov_direction(X[tr], rng) for _ in range(n_random)]
        dirs += [probe_direction(X[tr], lab[tr], C)[0] for lab in surf.values()]
        dirs = np.stack(dirs)                                                         # [K, d]
        vecs = (dirs[:, None, None, :] * (s * coef_a)[None, :, :, None]).reshape(-1, d)
        for i in te:
            B = len(conts[i])
            sc = score_choices_multi(model, prefixes[i], conts[i], layer, vecs, where,
                                     max_seqs).reshape(K, A, 2, B)
            lab = labs[i]
            top = sc.argmax(-1)
            margin[:, :, :, i] = sc[..., lab == 1].max(-1) - sc[..., lab == 0].max(-1)
            ok[:, :, :, i] = lab[top] == 1
            flip[:, :, :, i] = top != base_top[i]
        done.add(f)
        new_folds += 1
        log(f"  fold {f + 1}/{folds} done in {time.time() - t0:.0f}s "
            f"(probe-projection SD s = {s:.3f})")
        if ckpt:
            order = sorted(done)
            np.savez_compressed(ckpt, meta=meta, alphas=np.array(alphas, float), margin=margin,
                                ok=ok, flip=flip, done=np.array(order),
                                s_done=np.array([s_fold[k] for k in order]))
    if len(done) < folds:
        log(f"Stopped after {new_folds} new fold(s); {len(done)}/{folds} done. "
            f"Run the same command again to continue.")
        covered = np.concatenate([splits[f][1] for f in sorted(done)])
        dose_summary(margin, flip, ok, base_margin, base_ok, covered, alphas, names, state_norm,
                     float(np.mean(list(s_fold.values()))), log)
        return None
    return {"names": np.array(names), "alphas": np.array(alphas, float), "where": where,
            "layer": layer, "margin": margin, "ok": ok, "flip": flip,
            "base_margin": base_margin, "base_ok": base_ok, "base_top": base_top,
            "s_fold": np.array([s_fold[k] for k in range(folds)]), "n_random": n_random,
            "state_norm": np.array(state_norm)}


def dose_summary(margin, flip, ok, base_margin, base_ok, covered, alphas, names, state_norm, s_mean, log=print):
    """Dose-response on the questions covered so far. Uses RANDOM directions to judge whether
    the push is big enough to change anything at all (flip rate = share of questions whose
    top-scored choice changes). Pick the dose from this, not from the probe's own slope."""
    names = [str(n) for n in names]
    rnd = [i for i, n in enumerate(names) if n.startswith(("iso", "cov"))]
    ip = names.index("probe")
    log(f"\nDose-response on {len(covered)} questions (mean hidden-state norm {state_norm:.1f}, "
        f"probe-projection SD {s_mean:.3f})")
    log(f"{'alpha':>7} {'push norm':>10} {'% of state':>11} | {'random flip':>12} {'random acc':>11} "
        f"{'random |d margin|':>18} | {'probe flip':>11} {'probe slope':>12}")
    for ai, a in enumerate(alphas):
        push = a * s_mean
        rf = flip[rnd, ai][:, :, covered].mean()
        racc = ok[rnd, ai][:, :, covered].mean()
        rdm = np.abs(margin[rnd, ai][:, :, covered] - base_margin[covered]).mean()
        pf = flip[ip, ai][:, covered].mean()
        sl = (margin[ip, ai, 0, covered] - margin[ip, ai, 1, covered]).mean()
        log(f"{a:>7g} {push:>10.2f} {100 * push / state_norm:>10.1f}% | {rf:>12.1%} {racc:>11.1%} "
            f"{rdm:>18.4f} | {pf:>11.1%} {sl:>+12.4f}")
    log(f"(unsteered accuracy on these questions: {base_ok[covered].mean():.1%}. Rule fixed in advance: "
        f"use the smallest alpha whose RANDOM-direction flip rate is >= 5%.)")


# ----------------------------------------------------------------------------
# analysis (works from the saved raw arrays, so it can be rerun without the model)
# ----------------------------------------------------------------------------
def p_two_sided(stat, null):
    null = np.asarray(null)
    return (1 + int((np.abs(null) >= abs(stat)).sum())) / (1 + len(null))


def verdict(slope, p_iso, p_cov, specific=None, p_thresh=P_THRESH):
    if p_iso <= p_thresh and p_cov <= p_thresh:
        if slope > 0 and specific is False:
            return ("LEVER, BUT NOT SPECIFIC: the probe direction beats random directions, but a "
                    "surface-feature direction (length / 'no comment') moves the margin as much, "
                    "so this is not evidence about correctness itself")
        if slope > 0:
            return ("LEVER: pushing along the probe direction moves choices toward the correct "
                    "answers more than random directions of equal size do")
        return ("WRONG-SIGN LEVER: the probe direction moves choices more than random directions, "
                "but toward the wrong answers when pushed toward 'correct'")
    return ("NO SPECIAL EFFECT: the probe direction moves the correct-vs-incorrect margin no more "
            "than random directions of equal size; the probe reads a signal the model does not "
            "use as a lever here")


def analyze(raw, boot=2000, seed=0, p_thresh=P_THRESH):
    names = [str(n) for n in raw["names"]]
    alphas = [float(a) for a in raw["alphas"]]
    M, OK, FL = raw["margin"], raw["ok"], raw["flip"]
    bm, bok = raw["base_margin"], raw["base_ok"]
    N = len(bm)
    ip = names.index("probe")
    fams = {"iso": [i for i, n in enumerate(names) if n.startswith("iso")],
            "cov": [i for i, n in enumerate(names) if n.startswith("cov")]}
    surf_idx = [i for i, n in enumerate(names) if n.startswith("surf_")]
    rng = np.random.default_rng(seed)
    bidx = rng.integers(0, N, (boot, N))
    res = {"base_acc": float(bok.mean()), "base_margin": float(bm.mean()), "n": N, "p_thresh": p_thresh,
           "alphas": {}}
    for ai, a in enumerate(alphas):
        d_plus = (M[:, ai, 0, :] - bm).mean(1)
        d_minus = (M[:, ai, 1, :] - bm).mean(1)
        slope = (M[:, ai, 0, :] - M[:, ai, 1, :]).mean(1)
        acc_p, acc_m = OK[:, ai, 0, :].mean(1), OK[:, ai, 1, :].mean(1)
        flp = FL[:, ai].mean((1, 2))
        si = M[ip, ai, 0, :] - M[ip, ai, 1, :]
        bs = si[bidx].mean(1)
        row = {"probe": {"d_plus": float(d_plus[ip]), "d_minus": float(d_minus[ip]),
                         "slope": float(slope[ip]),
                         "slope_ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                         "acc_plus": float(acc_p[ip]), "acc_minus": float(acc_m[ip]),
                         "flip": float(flp[ip])}}
        for fam, idx in fams.items():
            ns = slope[idx]
            row[fam] = {"n": len(idx), "slope_mean": float(ns.mean()), "slope_sd": float(ns.std()),
                        "d_plus_mean": float(d_plus[idx].mean()), "d_minus_mean": float(d_minus[idx].mean()),
                        "acc_plus_mean": float(acc_p[idx].mean()), "acc_minus_mean": float(acc_m[idx].mean()),
                        "flip_mean": float(flp[idx].mean()),
                        "p": p_two_sided(slope[ip], ns)}
        specific = None
        if surf_idx:
            row["surf"] = {names[i]: float(slope[i]) for i in surf_idx}
            specific = bool(abs(slope[ip]) > max(abs(slope[i]) for i in surf_idx))
            row["specific"] = specific
        row["verdict"] = verdict(slope[ip], row["iso"]["p"], row["cov"]["p"], specific, p_thresh)
        res["alphas"][str(a)] = row
    return res


def report(res, raw):
    a0 = str(float(raw["alphas"][0]))
    print(f"\nProbe-direction steering | layer {raw['layer']} | vector added at: {raw['where']} tokens "
          f"| n={res['n']} | random replicates per family: {raw['n_random']}")
    print(f"Unsteered: accuracy {res['base_acc']:.1%}, mean margin {res['base_margin']:+.4f}")
    if "state_norm" in raw:
        sn, sm = float(raw["state_norm"]), float(np.mean(raw["s_fold"]))
        print("Push size: " + ", ".join(f"{float(a):g} SD = {float(a) * sm:.2f} ({100 * float(a) * sm / sn:.1f}% "
                                         f"of the mean hidden-state norm {sn:.1f})" for a in raw["alphas"]))
    print("Margin = best correct score - best incorrect score; slope = margin(+alpha) - margin(-alpha)")
    for a, r in res["alphas"].items():
        tag = "  <- PRIMARY (fixed before running)" if a == a0 else "  (exploratory)"
        print(f"\nalpha = {float(a):g} SD along the probe direction{tag}")
        print(f"{'':<14}{'d margin(+)':>12}{'d margin(-)':>13}{'slope':>10}{'acc(+)':>9}{'acc(-)':>9}{'flip':>8}")
        p = r["probe"]
        print(f"{'probe':<14}{p['d_plus']:>+12.4f}{p['d_minus']:>+13.4f}{p['slope']:>+10.4f}"
              f"{p['acc_plus']:>9.1%}{p['acc_minus']:>9.1%}{p['flip']:>8.1%}")
        for fam in ("iso", "cov"):
            q = r[fam]
            print(f"{fam + ' (null)':<14}{q['d_plus_mean']:>+12.4f}{q['d_minus_mean']:>+13.4f}"
                  f"{q['slope_mean']:>+10.4f}{q['acc_plus_mean']:>9.1%}{q['acc_minus_mean']:>9.1%}"
                  f"{q['flip_mean']:>8.1%}   slope sd {q['slope_sd']:.4f}, p = {q['p']:.3f}")
        for nm, sl in r.get("surf", {}).items():
            print(f"{nm:<14}{'':>12}{'':>13}{sl:>+10.4f}   (real feature direction, unrelated to correctness)")
        lo, hi = p["slope_ci95"]
        print(f"probe slope 95% CI over questions: [{lo:+.4f}, {hi:+.4f}]")
        print(f"VERDICT: {r['verdict']}")
    pt = res.get("p_thresh", P_THRESH)
    print(f"\n(claim a lever only if p <= {pt:g} against BOTH null families; p resolution is "
          f"1/{raw['n_random'] + 1}. Only the primary alpha is a pre-specified test.)")
    if 1 / (raw["n_random"] + 1) > pt:
        print(f"WARNING: with {raw['n_random']} random replicates the smallest possible p is "
              f"{1 / (raw['n_random'] + 1):.3f} > {pt:g}, so no LEVER verdict is reachable; "
              f"use --random {int(np.ceil(1 / pt))}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n", type=int, default=817)
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--features", default=None,
                    help="npz from probe_position.py (--cache); extracted and saved here if missing")
    ap.add_argument("--alphas", type=float, nargs="+", default=[2.0, 4.0])
    ap.add_argument("--random", type=int, default=30, help="random replicates per family")
    ap.add_argument("--where", choices=["answer", "all"], default="answer")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--p-thresh", type=float, default=P_THRESH,
                    help="p threshold for the LEVER verdict (default 0.05; 0.025 after a second look)")
    ap.add_argument("--ckpt", default="steer_ckpt.npz",
                    help="progress is saved here after every fold; rerunning resumes from it")
    ap.add_argument("--stop-after", type=int, default=None,
                    help="run at most this many NEW folds, then stop (run again to continue)")
    ap.add_argument("--max-seqs", type=int, default=128,
                    help="sequences per forward pass; lower = less GPU memory/heat, slower")
    ap.add_argument("--raw", default="steer_raw.npz",
                    help="raw results; if it exists the model is skipped and only the analysis reruns")
    ap.add_argument("--out", default=None, help="write the analysis JSON here")
    a = ap.parse_args(argv)

    if os.path.exists(a.raw):
        print(f"Loading raw results from {a.raw} (delete it to rerun the model)")
        z = np.load(a.raw, allow_pickle=False)
        raw = {k: z[k] for k in z.files}
        raw["where"], raw["layer"] = str(raw["where"]), int(raw["layer"])
        raw["n_random"] = int(raw["n_random"])
    else:
        items = pp.load_truthfulqa(a.n)            # dataset first, then CUDA
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.model)
        model = AutoModelForCausalLM.from_pretrained(a.model).to(a.device).eval()
        if a.features and os.path.exists(a.features):
            print(f"Loading probe features from {a.features}")
            feats = pp.load_features(a.features)
        else:
            feats = pp.extract_features(model, tok, items, [a.layer])
            if a.features:
                pp.save_features(a.features, feats)
        raw = run_steering(model, tok, items, feats, a.layer, a.alphas, a.random, a.where,
                           a.folds, a.C, a.seed, ckpt=a.ckpt, stop_after=a.stop_after,
                           max_seqs=a.max_seqs)
        if raw is None:
            return
        np.savez_compressed(a.raw, **raw)
        print(f"Saved raw results to {a.raw}")
    res = analyze(raw, seed=a.seed, p_thresh=a.p_thresh)
    report(res, raw)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"Wrote {a.out}")


if __name__ == "__main__":
    main()