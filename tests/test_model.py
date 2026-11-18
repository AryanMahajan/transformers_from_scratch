import numpy as np
import pytest

from src.nn.loss import cross_entropy, cross_entropy_backward
from src.transformer.model import GPT, GPTConfig
from src.utils.numerical import check_module


def _cfg(**kw):
    base = dict(vocab_size=11, d_model=8, n_heads=2, n_layers=2, max_len=16)
    base.update(kw)
    return GPTConfig(**base)


def _model(seed=0, **kw):
    return GPT(_cfg(**kw), np.random.default_rng(seed)).astype(np.float64)


def _ids(rng, B=2, T=6, V=11):
    return rng.integers(0, V, size=(B, T))


def _scale_weights(model, factor):
    """O(1) gradients for the grad check (see docs/05 §2 gotcha)."""
    for p in model.parameters():
        if p.data.ndim == 2:
            p.data *= factor
    return model


def expected_params(V, D, N, max_len, pos, tie):
    n = V * D + N * (12 * D * D + 12 * D) + 2 * D                # emb + blocks + ln_f
    n += max_len * D if pos == "learned" else 0
    n += 0 if tie else V * D
    return n


@pytest.mark.parametrize("pos", ["sinusoidal", "learned"])
@pytest.mark.parametrize("tie", [True, False])
def test_logits_shape_and_param_count(pos, tie):
    m = _model(pos=pos, tie_weights=tie)
    logits = m(_ids(np.random.default_rng(1)))
    assert logits.shape == (2, 6, 11)
    assert m.num_parameters() == expected_params(11, 8, 2, 16, pos, tie)


@pytest.mark.parametrize("pos", ["sinusoidal", "learned"])
@pytest.mark.parametrize("tie", [True, False])
def test_full_model_gradient_check(pos, tie):
    """Phase 5 exit criterion: every parameter of a 2-block, D=8 model."""
    rng = np.random.default_rng(2)
    m = _scale_weights(_model(pos=pos, tie_weights=tie), 20.0)
    errs = check_module(m, _ids(rng, B=2, T=5), rng)
    assert "input" not in errs                                   # ids are integers
    assert len(errs) == len(list(m.named_parameters()))
    assert max(errs.values()) < 1e-8, max(errs, key=errs.get)


@pytest.mark.parametrize("pos", ["sinusoidal", "learned"])
def test_full_model_causality(pos):
    """Phase 5 exit criterion: changing tokens after t can't change logits <= t."""
    rng = np.random.default_rng(3)
    m = _model(pos=pos)
    ids = _ids(rng, B=1, T=12)
    logits = m(ids)
    for t in range(11):
        ids2 = ids.copy()
        ids2[0, t + 1:] = (ids2[0, t + 1:] + 1 + rng.integers(0, 10, size=11 - t)) % 11
        logits2 = m(ids2)
        np.testing.assert_array_equal(logits2[0, : t + 1], logits[0, : t + 1])
        assert not np.allclose(logits2[0, t + 1:], logits[0, t + 1:])


def test_tied_gradient_is_sum_of_both_uses():
    """Tied E.grad == embedding-lookup grad + lm-head grad (from an untied twin)."""
    rng = np.random.default_rng(4)
    tied = _model(seed=5)
    untied = _model(seed=5, tie_weights=False)
    untied.load_state_dict({**tied.state_dict(), "lm_head.W": tied.tok_emb.E.data.T.copy()})
    ids = _ids(rng)
    np.testing.assert_allclose(tied(ids), untied(ids), rtol=1e-12)     # same function
    g = rng.standard_normal((2, 6, 11))
    tied.zero_grad(); tied.backward(g)
    untied.zero_grad(); untied.backward(g)
    both = untied.tok_emb.E.grad + untied.lm_head.W.grad.T
    np.testing.assert_allclose(tied.tok_emb.E.grad, both, rtol=1e-10)
    assert np.abs(untied.lm_head.W.grad).max() > 1e-3 and np.abs(untied.tok_emb.E.grad).max() > 1e-3


def test_initial_loss_is_close_to_log_vocab():
    """Small init -> logits ~ 0 -> uniform prediction -> loss ~ ln V (Phase 1)."""
    V = 66
    m = GPT(_cfg(vocab_size=V, d_model=32, n_layers=2, max_len=32), np.random.default_rng(6))
    ids = np.random.default_rng(7).integers(0, V, size=(4, 32))
    loss, _ = cross_entropy(m(ids)[:, :-1], ids[:, 1:])
    assert abs(loss - np.log(V)) < 0.05


def test_one_step_of_gradient_descent_lowers_loss():
    """End-to-end sanity check of forward + loss + backward together."""
    rng = np.random.default_rng(8)
    m = _model()
    ids = _ids(rng, B=4, T=10)
    x, y = ids[:, :-1], ids[:, 1:]
    loss0, probs = cross_entropy(m(x), y)
    m.zero_grad()
    m.backward(cross_entropy_backward(probs, y).reshape(4, 9, 11))
    for p in m.parameters():
        p.data -= 0.5 * p.grad
    loss1, _ = cross_entropy(m(x), y)
    assert loss1 < loss0


def test_residual_std_used_for_output_projections():
    m = GPT(_cfg(d_model=256, n_layers=8), np.random.default_rng(9))
    std = 0.02 / np.sqrt(16)
    for b in m.blocks:
        assert abs(b.attn.o_proj.W.data.std() - std) < 0.1 * std
        assert abs(b.ffn.fc2.W.data.std() - std) < 0.1 * std
        assert abs(b.attn.q_proj.W.data.std() - 0.02) < 0.002


def test_too_long_sequence_and_bad_pos_rejected():
    with pytest.raises(ValueError):
        _model()(np.zeros((1, 17), dtype=int))
    with pytest.raises(ValueError):
        _model(pos="rope")


@pytest.mark.parametrize("pos", ["sinusoidal", "learned"])
@pytest.mark.parametrize("tie", [True, False])
def test_scale_emb_gradient_check(pos, tie):
    """sqrt(D) token scaling: backward must multiply the lookup gradient by sqrt(D).

    The embedding gradient (the only one the scaling changes) is checked tightly. Every
    parameter gets a looser 1e-7 bound: with weights x20 the attention q/k gradients sit at a
    finite-difference noise floor of ~1e-8 to 6e-8 whether or not scale_emb is on (measured).
    Forgetting the sqrt(D) in backward gives a tok_emb.E error of 0.17-0.48 (measured)."""
    rng = np.random.default_rng(4)
    m = _scale_weights(_model(pos=pos, tie_weights=tie, scale_emb=True), 20.0)
    errs = check_module(m, _ids(rng, B=2, T=5), rng)
    assert errs["tok_emb.E"] < 1e-9
    assert max(errs.values()) < 1e-7, max(errs, key=errs.get)


def test_scale_emb_forward_scales_only_the_token_part():
    a, b = _model(), _model(scale_emb=True)                      # same seed -> same weights
    ids = np.array([[1, 2, 3]])
    e = a.tok_emb(ids)
    np.testing.assert_allclose(b.pos_enc(b.tok_emb(ids) * np.sqrt(8)), a.pos_enc(e * np.sqrt(8)))
    assert not np.allclose(a(ids), b(ids))
    assert GPTConfig(vocab_size=3).scale_emb is False             # old configs / checkpoints unchanged


def test_scale_emb_balances_tokens_and_sinusoidal_positions():
    """The Phase 8 finding: without scaling, position rows are ~35x the token rows at init."""
    for scale, lo, hi in ((False, 20, 50), (True, 2, 8)):
        m = GPT(GPTConfig(vocab_size=512, d_model=64, n_heads=4, max_len=128, scale_emb=scale),
                np.random.default_rng(0))
        tok = np.linalg.norm(m.tok_emb(np.arange(512)) * (8.0 if scale else 1.0), axis=1).mean()
        pos = np.linalg.norm(m.pos_enc.pe, axis=1).mean()
        assert lo < pos / tok < hi
