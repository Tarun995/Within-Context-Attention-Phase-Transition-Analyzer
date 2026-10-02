"""
tests/test_model_adapter.py — checks the block lookup AND the capture/patch
properties the whole causal audit depends on, on BOTH GPT-2 and
Pythia/GPT-NeoX architectures (tiny random models, no network):
  - patching an early position changes LATER positions' logits (leverage)
  - patching does not change EARLIER positions (causal masking)
  - patching a position with its own captured vector is a no-op
"""
import pytest
import torch
from transformers import (GPT2Config, GPT2LMHeadModel,
                          GPTNeoXConfig, GPTNeoXForCausalLM)

from attn_phase.audit.model_adapter import (
    get_block_module, get_num_layers, auto_layers)
from attn_phase.audit.activations import (
    capture_hidden_state_from_ids, run_with_hidden_patch)

VOCAB = 1000


def _gpt2():
    return GPT2LMHeadModel(GPT2Config(
        n_layer=3, n_embd=32, n_head=4, n_positions=64,
        vocab_size=VOCAB)).eval()


def _neox():
    return GPTNeoXForCausalLM(GPTNeoXConfig(
        num_hidden_layers=3, hidden_size=32, num_attention_heads=4,
        intermediate_size=64, vocab_size=VOCAB,
        max_position_embeddings=64)).eval()


BUILDERS = [pytest.param(_gpt2, id="gpt2"), pytest.param(_neox, id="pythia-neox")]


def _ids(n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, VOCAB, (1, n), generator=g)


@pytest.mark.parametrize("build", BUILDERS)
def test_block_lookup_and_layer_count(build):
    model = build()
    assert get_num_layers(model) == 3
    assert get_block_module(model, 0) is not get_block_module(model, 2)
    assert get_block_module(model, -1) is get_block_module(model, 2)


def test_unsupported_model_raises_instead_of_guessing():
    with pytest.raises(ValueError, match="Don't know where transformer blocks"):
        get_block_module(torch.nn.Linear(2, 2), 0)


def test_auto_layers():
    assert auto_layers(12) == [0, 3, 6, 9, 11]   # gpt2-small, includes 3
    assert auto_layers(6) == [0, 1, 3, 4, 5]     # pythia-70m
    assert auto_layers(24) == [0, 6, 12, 18, 23]  # gpt2-medium


@pytest.mark.parametrize("build", BUILDERS)
def test_capture_shape(build):
    model = build()
    vec = capture_hidden_state_from_ids(model, _ids(6, 0), 1)
    assert vec.shape == (32,)


@pytest.mark.parametrize("build", BUILDERS)
def test_patch_changes_later_positions_and_spares_earlier(build):
    model = build()
    prefix, choice = _ids(6, 0), _ids(4, 1)
    full = torch.cat([prefix, choice], dim=1)
    pos = prefix.shape[1] - 1
    real = capture_hidden_state_from_ids(model, prefix, 0)
    noise = torch.randn_like(real) * real.std() * 5

    base = run_with_hidden_patch(model, full, 0, None, "cpu")
    patched = run_with_hidden_patch(model, full, 0, noise, "cpu", position=pos)

    assert not torch.allclose(base[pos + 1:], patched[pos + 1:], atol=1e-4), (
        "Patching did not change later positions: no causal leverage on "
        "this architecture, do NOT trust results from it.")
    assert torch.allclose(base[:pos], patched[:pos], atol=1e-4), (
        "Patching changed EARLIER positions: violates causal masking.")


@pytest.mark.parametrize("build", BUILDERS)
def test_patching_own_vector_is_noop(build):
    model = build()
    prefix, choice = _ids(6, 0), _ids(4, 1)
    full = torch.cat([prefix, choice], dim=1)
    pos = prefix.shape[1] - 1
    own = capture_hidden_state_from_ids(model, prefix, 1)

    base = run_with_hidden_patch(model, full, 1, None, "cpu")
    same = run_with_hidden_patch(model, full, 1, own, "cpu", position=pos)
    assert torch.allclose(base, same, atol=1e-3)