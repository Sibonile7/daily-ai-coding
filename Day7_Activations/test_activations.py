import numpy as np
from Activation_problem import activation, activation_grad


def test_relu_grad():
    x = np.array([-1., 0., 2.])
    y = activation(x, "relu")
    dy = activation_grad(x, "relu")
    assert np.allclose(y, [0., 0., 2.])
    assert np.allclose(dy, [0., 0., 1.])


def test_sigmoid_range():
    x = np.linspace(-10, 10, 100)
    y = activation(x, "sigmoid")
    assert np.all(y >= 0) and np.all(y <= 1)


def test_tanh_range():
    x = np.linspace(-5, 5, 50)
    y = activation(x, "tanh")
    assert np.all(y >= -1) and np.all(y <= 1)


def test_swish_behavior():
    x = np.array([-5., 0., 5.])
    y = activation(x, "swish")
    assert y[0] < 0
    assert y[1] == 0
    assert y[2] > 0


def test_gelu_shape():
    x = np.random.randn(20)
    y = activation(x, "gelu")
    dy = activation_grad(x, "gelu")
    assert y.shape == x.shape
    assert dy.shape == x.shape
