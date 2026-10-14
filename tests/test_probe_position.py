import numpy as np
import pytest
import torch
import torch.nn.functional as F

from attn_phase.audit import probe_position as pp

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


def make_items(n, seed=0, flag_every=None):
    rng = np.random.default_rng(seed)
    items = []
    for i in range(n):
        q = " ".join(rng.choice(WORDS[:60], size=rng.integers(2, 6)))
        if flag_every and i % flag_every == 0:
            q = "zz " + q
        choices = [" ".join(rng.choice(WORDS[:60], size=rng.integers(1, 5))) for _ in range(4)]
        labels = [0] * 4
        labels[int(rng.integers(4))] = 1
        items.append({"question": q, "choices": choices, "labels": labels})
    return items


@pytest.fixture(scope="module")
def tok_vocab():
    return make_tok()


@pytest.mark.parametrize("kind", ["gpt2", "neox"])
def test_get_blocks_and_resolve(kind, tok_vocab):
    _, v = tok_vocab
    m = make_model(kind, v, n_layer=4)
    assert len(pp.get_blocks(m)) == 4
    assert pp.resolve_layers(m, [1, 3, -1, -4]) == [0, 1, 3]


@pytest.mark.parametrize("kind", ["gpt2", "neox"])
def test_capture_matches_hf_hidden_states(kind, tok_vocab):
    tok, v = tok_vocab
    m = make_model(kind, v)
    ids = tok("Q: w1 w2\nA: w3 w4", add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        hs = m(torch.tensor([ids]), output_hidden_states=True).hidden_states
    cap = pp.capture(m, ids, {"end": len(ids) - 1, "first": 0}, [0, 1, 2])
    for li in [0, 1, 2]:  # hidden_states[li+1] = output of block li (not the last block: ln applied)
        assert np.allclose(cap[li]["end"], hs[li + 1][0, -1].numpy(), atol=1e-5)
        assert np.allclose(cap[li]["first"], hs[li + 1][0, 0].numpy(), atol=1e-5)


@pytest.mark.parametrize("kind", ["gpt2", "neox"])
def test_boundary_state_is_causal_prefix_state(kind, tok_vocab):
    """The boundary state in the full sequence equals the last-token state of the prefix alone."""
    tok, v = tok_vocab
    m = make_model(kind, v)
    prefix = tok("Q: w5 w6 w7\nA:", add_special_tokens=False)["input_ids"]
    cont = tok(" w8 w9", add_special_tokens=False)["input_ids"]
    assert tok.decode(prefix[-1:]).strip() == ":"
    a = pp.capture(m, prefix, {"b": len(prefix) - 1}, [1, 3])
    b = pp.capture(m, prefix + cont, {"b": len(prefix) - 1}, [1, 3])
    for li in [1, 3]:
        assert np.allclose(a[li]["b"], b[li]["b"], atol=1e-5)


def test_choice_score_matches_independent_formula(tok_vocab):
    tok, v = tok_vocab
    m = make_model("gpt2", v)
    prefix = tok("Q: w1\nA:", add_special_tokens=False)["input_ids"]
    cont = tok(" w2 w3 w4", add_special_tokens=False)["input_ids"]
    ids = torch.tensor([prefix + cont])
    with torch.no_grad():
        logits = m(ids).logits[0]
    # independent: cross-entropy of positions that predict the continuation tokens
    ce = F.cross_entropy(logits[len(prefix) - 1:-1], ids[0, len(prefix):], reduction="mean")
    assert pp.choice_score(m, prefix, cont) == pytest.approx(-ce.item(), abs=1e-5)


def test_chosen_answer_invariant_to_choice_order(tok_vocab):
    tok, v = tok_vocab
    m = make_model("gpt2", v)
    items = make_items(10, seed=3)
    for it in items:
        prefix = tok(f"Q: {it['question']}\nA:", add_special_tokens=False)["input_ids"]
        def top(choices):
            sc = [pp.choice_score(m, prefix, tok(" " + c, add_special_tokens=False)["input_ids"]) for c in choices]
            return choices[int(np.argmax(sc))]
        rev = list(reversed(it["choices"]))
        assert top(it["choices"]) == top(rev)


@pytest.mark.parametrize("kind", ["gpt2", "neox"])
def test_extract_features_shapes_and_cache(kind, tok_vocab, tmp_path):
    tok, v = tok_vocab
    m = make_model(kind, v)
    feats = pp.extract_features(m, tok, make_items(30, seed=1), [1, 3, -1], log_every=0)
    assert feats["layers"] == [1, 2, 3] or feats["layers"] == [1, 3]
    n = 30
    assert feats["labels"].shape == (n,) and set(np.unique(feats["labels"])) <= {0, 1}
    for (li, p), X in feats["X"].items():
        assert X.shape == (n, 32) and np.isfinite(X).all()
    path = tmp_path / "f.npz"
    pp.save_features(path, feats)
    back = pp.load_features(path)
    assert back["layers"] == feats["layers"]
    assert np.array_equal(back["labels"], feats["labels"])
    for k in feats["X"]:
        assert np.allclose(back["X"][k], feats["X"][k])
    assert list(back["answers"]) == feats["answers"]


def test_eval_probe_detects_planted_signal_and_null_is_chance():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    X = rng.normal(size=(400, 32))
    X[:, 0] += 3.0 * y  # planted signal (ideal AUC about 0.98)
    auc, _ = pp.eval_probe(X, y, n_splits=10)
    assert auc > 0.85
    null, _ = pp.eval_probe(X, rng.permutation(y), n_splits=10)
    assert 0.40 < null < 0.60


def test_surface_features_and_verdict():
    f = pp.surface_features(["I have no comment", "Paris is the capital"], np.array([4, 5]))
    assert f.shape == (2, 4) and f[0, 3] == 1 and f[1, 3] == 0
    assert pp.verdict(0.80).startswith("CARRIES")
    assert pp.verdict(0.60).startswith("PARTIAL")
    assert pp.verdict(0.52).startswith("NOT DETECTED")


def relabel_with_signal(model, tok, items, flagged_correct):
    """Set labels so the model's own top choice is correct iff the question is flagged ('zz')."""
    out = []
    for it in items:
        prefix = tok(f"Q: {it['question']}\nA:", add_special_tokens=False)["input_ids"]
        sc = [pp.choice_score(model, prefix, tok(" " + c, add_special_tokens=False)["input_ids"]) for c in it["choices"]]
        top = int(np.argmax(sc))
        correct = top if it["question"].startswith("zz") == flagged_correct else (top + 1) % 4
        labels = [0] * 4
        labels[correct] = 1
        out.append({**it, "labels": labels})
    return out


@pytest.mark.parametrize("kind", ["gpt2", "neox"])
def test_end_to_end_finds_planted_boundary_signal_and_not_in_control(kind, tok_vocab):
    tok, v = tok_vocab
    m = make_model(kind, v, seed=1)
    base = make_items(240, seed=5, flag_every=2)
    planted = relabel_with_signal(m, tok, base, True)
    feats = pp.extract_features(m, tok, planted, [1], log_every=0)
    rows, surface = pp.run_probes(feats, n_splits=10)
    by = {(r["layer"], r["position"]): r for r in rows}
    assert by[(1, "boundary")]["auc"] > 0.9          # signal is in the prefix, so boundary sees it
    assert 0.35 < by[(1, "boundary")]["null_auc"] < 0.65
    # control: same prefixes, labels unrelated to the flag
    rng = np.random.default_rng(9)
    ctrl = [{**it, "labels": [int(j == rng.integers(4)) for j in range(4)]} for it in base]
    cfeats = pp.extract_features(m, tok, ctrl, [1], log_every=0)
    crows, _ = pp.run_probes(cfeats, n_splits=10)
    cby = {(r["layer"], r["position"]): r for r in crows}
    assert cby[(1, "boundary")]["auc"] < 0.65         # no planted signal -> near chance


def test_cli_from_cache_prints_table_and_verdict(tok_vocab, tmp_path, capsys):
    tok, v = tok_vocab
    m = make_model("gpt2", v, seed=1)
    planted = relabel_with_signal(m, tok, make_items(200, seed=5, flag_every=2), True)
    feats = pp.extract_features(m, tok, planted, [1, 2, -1], log_every=0)
    assert feats["n_blocks"] == 4
    path = tmp_path / "c.npz"
    pp.save_features(path, feats)
    out = tmp_path / "r.json"
    pp.main(["--cache", str(path), "--focus-layer", "-1", "--splits", "5", "--out", str(out)])
    text = capsys.readouterr().out
    assert "VERDICT at layer 3" in text and "Surface-feature baseline" in text
    assert out.exists()
    with pytest.raises(SystemExit):
        pp.main(["--cache", str(path), "--focus-layer", "0", "--splits", "5"])