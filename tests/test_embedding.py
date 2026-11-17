import numpy as np
import pytest

from src.nn.embedding import Embedding
from src.utils.numerical import check_module


def _emb(V=10, D=4, seed=0):
    return Embedding(V, D, np.random.default_rng(seed)).astype(np.float64)


def test_forward_is_lookup_and_equals_onehot_matmul():
    emb = _emb()
    ids = np.array([[1, 3, 3], [0, 9, 1]])                 # [B, T]
    out = emb(ids)
    assert out.shape == (2, 3, 4)
    np.testing.assert_array_equal(out[0, 1], emb.E.data[3])
    onehot = np.eye(10)[ids]                               # [B, T, V]
    np.testing.assert_allclose(out, onehot @ emb.E.data, rtol=1e-12)


def test_gradient_check_with_repeated_ids():
    rng = np.random.default_rng(1)
    ids = np.array([[2, 2, 5, 2], [5, 0, 0, 7]])           # lots of repeats
    errs = check_module(_emb(), ids, rng)
    assert set(errs) == {"E"}                              # no input grad for ints
    assert errs["E"] < 1e-8


def test_backward_is_scatter_add_and_returns_none():
    emb = _emb()
    ids = np.array([[4, 4, 4]])
    g = np.arange(12, dtype=float).reshape(1, 3, 4)
    emb(ids)
    assert emb.backward(g) is None
    np.testing.assert_allclose(emb.E.grad[4], g[0].sum(axis=0))   # all three summed
    unused = np.setdiff1d(np.arange(10), [4])
    assert not emb.E.grad[unused].any()                    # unseen tokens: zero grad


def test_naive_fancy_index_add_drops_duplicates():
    """Documents WHY np.add.at is needed."""
    grad = np.zeros((10, 4))
    ids = np.array([4, 4, 4])
    g = np.ones((3, 4))
    grad[ids] += g                                         # buffered: last write wins
    np.testing.assert_allclose(grad[4], 1.0)               # should be 3.0!
    fixed = np.zeros((10, 4))
    np.add.at(fixed, ids, g)
    np.testing.assert_allclose(fixed[4], 3.0)


def test_rejects_negative_out_of_range_and_float_ids():
    emb = _emb()
    with pytest.raises(IndexError):
        emb(np.array([[0, -1]]))                           # would silently be row 9
    with pytest.raises(IndexError):
        emb(np.array([[10]]))
    with pytest.raises(TypeError):
        emb(np.array([[1.0, 2.0]]))


def test_positional_embedding_usage():
    """Learned positions = same class, looked up with arange(T), broadcast over B."""
    rng = np.random.default_rng(2)
    pos = _emb(V=8, D=4)                                   # max context 8
    tok = rng.standard_normal((2, 5, 4))                   # pretend token embeddings
    x = tok + pos(np.arange(5))                            # [B,T,D] + [T,D]
    assert x.shape == (2, 5, 4)
    np.testing.assert_allclose(x[1, 3] - tok[1, 3], pos.E.data[3])
