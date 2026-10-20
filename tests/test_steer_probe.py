import numpy as np
import pytest
import torch

from attn_phase.audit import probe_position as pp
from attn_phase.audit import steer_probe as sp

# Tiny synthetic models/tokenizer, inlined so this file does not depend on test_probe_position.py.
WORDS = [f"w{i}" for i in range(60)] + ["zz"]


def make_tok():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    vocab = {"<unk>": 0}
    for w in ["Q", ":", "A"] + WORDS:
        vocab[w] = len(vocab)
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>"), len(vocab)


def make_model(kind, vocab, n_layer=4, seed=0):
    from transformers import GPT2Config, GPT2LMHeadModel, GPTNeoXConfig, GPTNeoXForCausalLM
    torch.manual_seed(seed)
    if kind == "gpt2":
        cfg = GPT2Config(vocab_size=vocab, n_positions=128, n_embd=32, n_layer=n_layer, n_head=2)
        return GPT2LMHeadModel(cfg).eval()
    cfg = GPTNeoXConfig(vocab_size=vocab, hidden_size=32, num_hidden_layers=n_layer,
                        num_attention_heads=2, intermediate_size=64, max_position_embeddings=128)
    return GPTNeoXForCausalLM(cfg).eval()


def make_items(n, seed=0):
    rng = np.random.default_rng(seed)
    items = []
    for _ in range(n):
        q = " ".join(rng.choice(WORDS[:60], size=rng.integers(2, 6)))
        choices = [" ".join(rng.choice(WORDS[:60], size=rng.integers(1, 5))) for _ in range(4)]
        labels = [0] * 4
        labels[int(rng.integers(4))] = 1
        items.append({"question": q, "choices": choices, "labels": labels})
    return items


@pytest.fixture(scope="module")
def tok_vocab():
    return make_tok()


def prefix_conts(tok, it):
    prefix = tok(f"Q: {it['question']}\nA:", add_special_tokens=False)["input_ids"]
    conts = [tok(" " + c, add_special_tokens=False)["input_ids"] for c in it["choices"]]
    return prefix, conts


# ---------------------------------------------------------------- scoring plumbing
@pytest.mark.parametrize("kind", ["gpt2", "neox"])
def test_batched_scores_match_unbatched(kind, tok_vocab):
    tok, v = tok_vocab
    m = make_model(kind, v)
    for it in make_items(6, seed=3):
        prefix, conts = prefix_conts(tok, it)
        batched = sp.score_choices(m, prefix, conts)
        single = np.array([pp.choice_score(m, prefix, c) for c in conts])
        assert np.allclose(batched, single, atol=1e-5)


@pytest.mark.parametrize("kind", ["gpt2", "neox"])
def test_zero_vector_is_identity_and_nonzero_changes(kind, tok_vocab):
    tok, v = tok_vocab
    m = make_model(kind, v)
    prefix, conts = prefix_conts(tok, make_items(1, seed=5)[0])
    base = sp.score_choices(m, prefix, conts)
    assert np.allclose(sp.score_choices(m, prefix, conts, 1, np.zeros(32)), base, atol=1e-6)
    # (not an all-ones vector: LayerNorm subtracts the mean, so a constant shift is invisible)
    vec = 3.0 * np.random.default_rng(0).standard_normal(32)
    assert not np.allclose(sp.score_choices(m, prefix, conts, 1, vec), base, atol=1e-3)


@pytest.mark.parametrize("kind", ["gpt2", "neox"])
@pytest.mark.parametrize("where", ["answer", "all"])
def test_hook_adds_vector_at_the_right_positions(kind, where, tok_vocab):
    """Independent implementation: add the vector by slicing one sequence at a time."""
    tok, v = tok_vocab
    m = make_model(kind, v)
    rng = np.random.default_rng(0)
    vec = rng.standard_normal(32).astype(np.float32) * 2
    prefix, conts = prefix_conts(tok, make_items(1, seed=7)[0])
    got = sp.score_choices(m, prefix, conts, 2, vec, where)
    start = len(prefix) if where == "answer" else 0
    tv = torch.tensor(vec)
    want = []
    for c in conts:
        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h = h.clone()
            h[:, start:, :] += tv
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        hd = pp.get_blocks(m)[2].register_forward_hook(hook)
        try:
            want.append(pp.choice_score(m, prefix, c))
        finally:
            hd.remove()
    assert np.allclose(got, want, atol=1e-5)


@pytest.mark.parametrize("kind", ["gpt2", "neox"])
def test_last_token_only_steering_cannot_change_scores(kind, tok_vocab):
    """Why the vector goes on the answer tokens: nothing is scored after the last token."""
    tok, v = tok_vocab
    m = make_model(kind, v)
    prefix, conts = prefix_conts(tok, make_items(1, seed=9)[0])
    c = conts[0]
    base = pp.choice_score(m, prefix, c)

    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        h = h.clone()
        h[:, -1, :] += 50.0
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
    hd = pp.get_blocks(m)[2].register_forward_hook(hook)
    try:
        steered = pp.choice_score(m, prefix, c)
    finally:
        hd.remove()
    assert abs(base - steered) < 1e-6


def test_margin_top():
    sc = np.array([-1.0, -0.2, -0.5, -3.0])
    m, ok, top = sp.margin_top(sc, [0, 1, 0, 0])
    assert top == 1 and ok and m == pytest.approx(0.3)
    m, ok, top = sp.margin_top(sc, [0, 0, 1, 0])
    assert top == 1 and not ok and m == pytest.approx(-0.3)


# ---------------------------------------------------------------- directions
def test_probe_direction_recovers_planted_axis_in_raw_space():
    rng = np.random.default_rng(0)
    d, n = 16, 600
    y = rng.integers(0, 2, n)
    scales = np.linspace(0.2, 5.0, d)               # very different per-dimension scales
    X = rng.standard_normal((n, d)) * scales
    X[:, 3] += (2 * y - 1) * 1.5 * scales[3]         # signal on dimension 3
    u, s = sp.probe_direction(X, y)
    assert np.linalg.norm(u) == pytest.approx(1.0)
    assert u[3] > 0.9 and s > 0
    u2, _ = sp.probe_direction(X, 1 - y)             # flipped labels flip the sign
    assert u2[3] < -0.9


def test_cov_direction_follows_state_variance_and_iso_does_not():
    rng = np.random.default_rng(1)
    d = 32
    X = rng.standard_normal((400, d))
    X[:, 0] *= 30.0                                   # one huge-variance dimension
    cov = np.array([abs(sp.cov_direction(X, rng)[0]) for _ in range(40)])
    iso = np.array([abs(sp.iso_direction(d, rng)[0]) for _ in range(40)])
    assert cov.mean() > 0.8 and iso.mean() < 0.3


# ---------------------------------------------------------------- analysis
def synth_raw(probe_slope, n_random=30, N=400, seed=0):
    rng = np.random.default_rng(seed)
    names = ["probe"] + [f"iso{r}" for r in range(n_random)] + [f"cov{r}" for r in range(n_random)]
    K = len(names)
    margin = rng.normal(0, 0.05, (K, 1, 2, N))
    margin[0, 0, 0] += probe_slope / 2
    margin[0, 0, 1] -= probe_slope / 2
    z = np.zeros((K, 1, 2, N), bool)
    return {"names": np.array(names), "alphas": np.array([2.0]), "where": "answer", "layer": 3,
            "margin": margin, "ok": z, "flip": z.copy(), "base_margin": np.zeros(N),
            "base_ok": np.zeros(N, bool), "base_top": np.zeros(N, int),
            "s_fold": np.ones(5), "n_random": n_random}


def test_analysis_flags_a_planted_lever_and_not_noise():
    res = sp.analyze(synth_raw(0.3))
    r = res["alphas"]["2.0"]
    assert r["verdict"].startswith("LEVER") and r["iso"]["p"] <= 0.05 and r["cov"]["p"] <= 0.05
    assert r["probe"]["slope"] == pytest.approx(0.3, abs=0.02)
    res = sp.analyze(synth_raw(-0.3))
    assert res["alphas"]["2.0"]["verdict"].startswith("WRONG-SIGN")
    res = sp.analyze(synth_raw(0.0, seed=4))
    assert res["alphas"]["2.0"]["verdict"].startswith("NO SPECIAL")


def test_p_two_sided_resolution():
    assert sp.p_two_sided(10.0, np.arange(30) * 0.01) == pytest.approx(1 / 31)
    assert sp.p_two_sided(0.0, np.arange(30) * 0.01) == 1.0


# ---------------------------------------------------------------- end to end
def zz_items(n, seed=0):
    """Four choices; the correct one always ends with the token 'zz'."""
    rng = np.random.default_rng(seed)
    items = []
    for _ in range(n):
        q = " ".join(rng.choice(WORDS[:60], size=rng.integers(2, 6)))
        choices = [" ".join(rng.choice(WORDS[:60], size=rng.integers(1, 4))) for _ in range(3)]
        correct = " ".join(rng.choice(WORDS[:60], size=rng.integers(1, 4))) + " zz"
        pos = int(rng.integers(4))
        choices.insert(pos, correct)
        labels = [0] * 4
        labels[pos] = 1
        items.append({"question": q, "choices": choices, "labels": labels})
    return items


def build(kind, tok, vocab, plant, kappa=10.0, n=160):
    items = zz_items(n)
    m = make_model(kind, vocab, n_layer=3)
    head = m.get_output_embeddings()
    # Untie AND re-randomise. A cloned head still equals the input embedding, and then a
    # direction encoding "the current token is zz" raises zz's logit with no planting at all.
    g = torch.Generator().manual_seed(123)
    head.weight = torch.nn.Parameter(0.02 * torch.randn(head.weight.shape, generator=g))
    if plant:
        f0 = pp.extract_features(m, tok, items, [-1], log_every=0)
        u, _ = sp.probe_direction(f0["X"][(2, "end")], f0["labels"])
        zz = tok.convert_tokens_to_ids("zz")
        with torch.no_grad():
            head.weight[zz] += kappa * torch.tensor(u, dtype=head.weight.dtype)
    feats = pp.extract_features(m, tok, items, [-1], log_every=0)
    return m, items, feats


@pytest.mark.parametrize("kind", ["gpt2", "neox"])
def test_end_to_end_detects_a_planted_lever(kind, tok_vocab):
    """Output head wired to the probe direction: steering along it must be flagged."""
    tok, v = tok_vocab
    m, items, feats = build(kind, tok, v, plant=True)
    assert 0 < feats["labels"].mean() < 1
    raw = sp.run_steering(m, tok, items, feats, -1, [2.0], n_random=30, folds=5, log=lambda *_: None)
    assert raw["margin"].shape == (61 + len(sp.surface_labels(feats)), 1, 2, len(items))
    res = sp.analyze(raw)["alphas"]["2.0"]
    assert res["probe"]["slope"] > 0
    assert res["verdict"].startswith("LEVER:"), res


@pytest.mark.parametrize("kind", ["gpt2", "neox"])
def test_end_to_end_no_lever_when_head_is_not_wired(kind, tok_vocab):
    tok, v = tok_vocab
    m, items, feats = build(kind, tok, v, plant=False)
    raw = sp.run_steering(m, tok, items, feats, -1, [2.0], n_random=30, folds=5, log=lambda *_: None)
    res = sp.analyze(raw)["alphas"]["2.0"]
    assert not res["verdict"].startswith("LEVER"), res


def test_label_mismatch_is_refused(tok_vocab):
    tok, v = tok_vocab
    m, items, feats = build("gpt2", tok, v, plant=False, n=60)
    feats["labels"] = 1 - feats["labels"]
    with pytest.raises(SystemExit):
        sp.run_steering(m, tok, items, feats, -1, [2.0], n_random=2, folds=3, log=lambda *_: None)


def test_cli_reanalyses_saved_raw(tmp_path, capsys):
    path = tmp_path / "raw.npz"
    np.savez_compressed(path, **synth_raw(0.3, n_random=30))   # p <= 0.05 needs >= 19 randoms
    sp.main(["--raw", str(path), "--out", str(tmp_path / "o.json")])
    out = capsys.readouterr().out
    assert "VERDICT: LEVER" in out and "PRIMARY" in out
    assert (tmp_path / "o.json").exists()


# ---------------------------------------------------------------- surface-direction control
def synth_raw_surf(probe_slope, surf_slope, n_random=30, N=400):
    raw = synth_raw(probe_slope, n_random, N)
    rng = np.random.default_rng(9)
    extra = rng.normal(0, 0.05, (1, 1, 2, N))
    extra[0, 0, 0] += surf_slope / 2
    extra[0, 0, 1] -= surf_slope / 2
    raw["margin"] = np.concatenate([raw["margin"], extra])
    z = np.zeros((1, 1, 2, N), bool)
    raw["ok"], raw["flip"] = np.concatenate([raw["ok"], z]), np.concatenate([raw["flip"], z])
    raw["names"] = np.array(list(raw["names"]) + ["surf_length"])
    return raw


def test_surface_direction_as_strong_as_probe_is_flagged_not_specific():
    r = sp.analyze(synth_raw_surf(0.3, 0.3))["alphas"]["2.0"]
    assert r["specific"] is False and r["verdict"].startswith("LEVER, BUT NOT SPECIFIC")
    r = sp.analyze(synth_raw_surf(0.3, 0.02))["alphas"]["2.0"]
    assert r["specific"] is True and r["verdict"].startswith("LEVER:")
    assert "surf_length" in r["surf"]


def test_surface_labels_drops_rare_classes():
    feats = {"answers": ["I have no comment"] * 5 + ["x"] * 95, "n_tok": np.r_[np.full(50, 2), np.full(50, 6)]}
    labs = sp.surface_labels(feats)
    assert "surf_nocomment" not in labs and "surf_length" in labs
    feats["answers"] = ["No comment."] * 40 + ["x"] * 60
    assert "surf_nocomment" in sp.surface_labels(feats)


# ---------------------------------------------------------------- batched scorer + resume
@pytest.mark.parametrize("kind", ["gpt2", "neox"])
@pytest.mark.parametrize("where", ["answer", "all"])
def test_multi_scorer_matches_single_scorer(kind, where, tok_vocab):
    tok, v = tok_vocab
    m = make_model(kind, v)
    rng = np.random.default_rng(0)
    vecs = rng.standard_normal((7, 32)).astype(np.float32) * 2
    vecs[0] = 0
    for it in make_items(4, seed=11):
        prefix, conts = prefix_conts(tok, it)
        for chunk in (1000, 9):                      # one chunk, and several chunks
            got = sp.score_choices_multi(m, prefix, conts, 2, vecs, where, max_seqs=chunk)
            want = np.stack([sp.score_choices(m, prefix, conts, 2, vec, where) for vec in vecs])
            assert got.shape == (7, len(conts)) and np.allclose(got, want, atol=1e-4)
        assert np.allclose(got[0], sp.score_choices(m, prefix, conts), atol=1e-4)   # zero vec = unsteered


def test_resume_matches_an_uninterrupted_run(tok_vocab, tmp_path):
    tok, v = tok_vocab
    m, items, feats = build("gpt2", tok, v, plant=False, n=60)
    kw = dict(n_random=3, folds=3, log=lambda *_: None)
    full = sp.run_steering(m, tok, items, feats, -1, [2.0], **kw)
    ck = str(tmp_path / "ck.npz")
    assert sp.run_steering(m, tok, items, feats, -1, [2.0], ckpt=ck, stop_after=1, **kw) is None
    assert sp.run_steering(m, tok, items, feats, -1, [2.0], ckpt=ck, stop_after=1, **kw) is None
    resumed = sp.run_steering(m, tok, items, feats, -1, [2.0], ckpt=ck, **kw)
    for k in ("margin", "ok", "flip", "s_fold"):
        assert np.allclose(full[k], resumed[k]), k


def test_checkpoint_with_other_settings_is_refused(tok_vocab, tmp_path):
    tok, v = tok_vocab
    m, items, feats = build("gpt2", tok, v, plant=False, n=60)
    ck = str(tmp_path / "ck.npz")
    kw = dict(folds=3, log=lambda *_: None, ckpt=ck)
    sp.run_steering(m, tok, items, feats, -1, [2.0], n_random=2, stop_after=1, **kw)
    with pytest.raises(SystemExit):
        sp.run_steering(m, tok, items, feats, -1, [2.0], n_random=3, **kw)


def test_dose_summary_reports_random_flip_rate(capsys):
    rng = np.random.default_rng(0)
    K, A, N = 5, 2, 40
    names = ["probe", "iso0", "iso1", "cov0", "surf_length"]
    flip = np.zeros((K, A, 2, N), bool)
    flip[1:4, 1, :, :20] = True                       # random directions flip half the covered questions at alpha[1]
    ok = np.zeros((K, A, 2, N), bool)
    margin = rng.normal(0, 0.01, (K, A, 2, N))
    sp.dose_summary(margin, flip, ok, np.zeros(N), np.zeros(N, bool), np.arange(20), [2.0, 8.0],
                    names, 60.0, 0.4)
    out = capsys.readouterr().out
    assert "Dose-response on 20 questions" in out
    lines = [l for l in out.splitlines() if l.strip().startswith(("2 ", "8 "))]
    assert len(lines) == 2 and "100.0%" in lines[1] and "0.0%" in lines[0]


def test_p_thresh_is_respected():
    raw = synth_raw(0.3, n_random=30)                # smallest possible p = 1/31 = 0.032
    assert sp.analyze(raw)["alphas"]["2.0"]["verdict"].startswith("LEVER")
    strict = sp.analyze(raw, p_thresh=0.025)
    assert strict["alphas"]["2.0"]["verdict"].startswith("NO SPECIAL") and strict["p_thresh"] == 0.025