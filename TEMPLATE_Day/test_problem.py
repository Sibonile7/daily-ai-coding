import numpy as np
from problem import solve

def test_placeholder():
    with np.testing.assert_raises(NotImplementedError):
        solve(np.array([1]))
