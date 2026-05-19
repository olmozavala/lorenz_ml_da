"""Tests for probabilistic metrics."""

import numpy as np
import pytest

from metrics import energy_score


def test_energy_score_decomposition():
  """Total ES equals accuracy minus spread."""
  rng = np.random.default_rng(0)
  ens = rng.standard_normal((3, 30))
  truth = rng.standard_normal(3)
  es, acc, spr = energy_score(ens, truth)
  assert es == pytest.approx(acc - spr)
  assert acc > 0
  assert spr > 0


def test_energy_score_nan_ensemble():
  ens = np.ones((3, 10))
  ens[0, 0] = np.nan
  es, acc, spr = energy_score(ens, np.zeros(3))
  assert np.isnan(es) and np.isnan(acc) and np.isnan(spr)


def test_energy_score_collapsed_ensemble():
  """Identical members: spread term is zero."""
  truth = np.array([1.0, 2.0, 3.0])
  member = np.array([2.0, 3.0, 4.0])
  ens = np.tile(member[:, None], (1, 20))
  es, acc, spr = energy_score(ens, truth)
  assert spr == pytest.approx(0.0)
  assert es == pytest.approx(acc)
  assert acc == pytest.approx(np.linalg.norm(member - truth))
