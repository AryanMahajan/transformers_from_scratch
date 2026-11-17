import numpy as np
import pytest

from src.utils.numerical import check_vjp, numerical_gradient, relative_error


def test_known_gradient_of_quadratic():
    x = np.array([1.0, -2.0, 3.0])
    num = numerical_gradient(lambda: float(np.sum(x**2)), x)
    assert relative_error(num, 2 * x) < 1e-10


def test_x_is_restored_after_check():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((3, 4))
    before = x.copy()
    numerical_gradient(lambda: float(np.sum(np.sin(x))), x)
    np.testing.assert_array_equal(x, before)


def test_rejects_float32():
    x = np.ones(3, dtype=np.float32)
    with pytest.raises(TypeError):
        numerical_gradient(lambda: float(x.sum()), x)


def test_relative_error_handles_zeros():
    assert relative_error(np.zeros(3), np.zeros(3)) == 0.0
    a = np.array([0.0, 1.0])
    b = np.array([1e-12, 1.0])  # elementwise rel. error would be 1.0 here
    assert relative_error(a, b) < 1e-11


def test_central_beats_forward_difference():
    """Central is O(h^2), forward is O(h): orders of magnitude more accurate."""
    x0, h = 0.7, 1e-4
    exact = np.cos(x0)
    forward = (np.sin(x0 + h) - np.sin(x0)) / h
    central = (np.sin(x0 + h) - np.sin(x0 - h)) / (2 * h)
    assert abs(central - exact) < abs(forward - exact) / 1000


def test_check_vjp_catches_a_buggy_backward():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((2, 5))

    def correct(g):
        return g * np.cos(x)

    def buggy(g):          # forgot the chain rule factor
        return np.cos(x)

    assert check_vjp(np.sin, correct, x, rng) < 1e-8
    assert check_vjp(np.sin, buggy, x, rng) > 1e-2
