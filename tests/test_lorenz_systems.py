import numpy as np
import pytest
from lorenz.lorenz_systems import LorenzSystems


def test_generate_trajectory_l63():
    dt = 0.01
    n_steps = 100
    x0 = [1.0, 1.0, 1.0]
    trajectory = LorenzSystems.generate_trajectory('63', x0, dt, n_steps)

    assert trajectory.shape == (n_steps, 3)
    assert np.allclose(trajectory[0], x0)


def test_generate_trajectory_invalid_system():
    with pytest.raises(ValueError):
        LorenzSystems.generate_trajectory('invalid', [1.0, 1.0, 1.0], 0.01, 10)


def test_generate_trajectory_l96_wrong_dim():
    x0 = np.ones(20)  # DAPyr L96 requires N=40
    with pytest.raises(ValueError, match="N=40"):
        LorenzSystems.generate_trajectory('96', x0, 0.05, 10)
