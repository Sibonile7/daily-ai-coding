import numpy as np
# robust import that works locally and in CI
try:
    from .problem_perceptron import PerceptronPLA   # when collected as a package
except ImportError:
    from problem_perceptron import PerceptronPLA    # when run directly


def test_perceptron_converges_on_separable_data():
    rng = np.random.default_rng(0)
    X1 = rng.normal([2, 2], 0.5, (50, 2))
    X2 = rng.normal([-2, -2], 0.5, (50, 2))
    X = np.vstack([X1, X2])
    y = np.hstack([np.ones(50), -np.ones(50)])

    model = PerceptronPLA(lr=1.0, max_epochs=1000).fit(X, y)
    y_pred = model.predict(X)
    acc = np.mean(y_pred == y)

    print("Accuracy:", acc)
    assert acc == 1.0
    assert model.converged_
