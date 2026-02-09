import numpy as np
import pytest
from lorenz.lorenz_systems import LorenzSystems

def test_lorenz63_shape():
    x = np.array([1.0, 1.0, 1.0])
    dx = LorenzSystems.lorenz63(x)
    assert dx.shape == (3,)

def test_lorenz63_origin():
    # At the origin, derivatives should be zero
    x = np.array([0.0, 0.0, 0.0])
    dx = LorenzSystems.lorenz63(x)
    assert np.allclose(dx, 0)

def test_lorenz96_shape():
    nx = 40
    x = np.random.rand(nx)
    dx = LorenzSystems.lorenz96(x)
    assert dx.shape == (nx,)

def test_get_system():
    f63 = LorenzSystems.get_system('63')
    assert f63 == LorenzSystems.lorenz63
    
    f96 = LorenzSystems.get_system('96')
    assert f96 == LorenzSystems.lorenz96
    
    with pytest.raises(ValueError):
        LorenzSystems.get_system('invalid')

def test_generate_trajectory():
    dt = 0.01
    n_steps = 100
    x0 = [1.0, 1.0, 1.0]
    trajectory = LorenzSystems.generate_trajectory('63', x0, dt, n_steps)
    
    assert trajectory.shape == (n_steps, 3)
    assert np.allclose(trajectory[0], x0)
