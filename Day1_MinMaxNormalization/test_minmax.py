import numpy as np
from problem import normalize, Normalizer

def test_basic():
    x = np.array([[1, 2, 3], [4, 5, 6]])
    out = normalize(x)
    assert np.allclose(out, [[0, 0, 0], [1, 1, 1]])

def test_constant_column():
    x = np.array([[2, 2], [2, 2]])
    out = normalize(x)
    assert np.allclose(out, np.zeros_like(x, dtype=float))

def test_with_nan():
    x = np.array([[1, np.nan], [5, 10]])
    out = normalize(x)
    assert np.allclose(out[0, 0], 0.0)
    assert np.allclose(out[1, 0], 1.0)

def test_empty():
    x = np.array([]).reshape(0, 2)
    out = normalize(x)
    assert out.shape == (0, 2)

def test_class_fit_transform():
    x = np.array([[1, 2], [3, 6]])
    norm = Normalizer().fit(x)
    out = norm.transform(x)
    assert np.allclose(out, [[0, 0], [1, 1]])

def test_inverse_transform():
    x = np.array([[1, 2], [3, 6]])
    norm = Normalizer().fit(x)
    out = norm.transform(x)
    inv = norm.inverse_transform(out)
    assert np.allclose(inv, x)
