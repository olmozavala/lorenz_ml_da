"""
metrics.py — probabilistic forecast metrics for ensemble data assimilation.

Provides a thin wrapper around :func:`scoringrules.energy_score` that adapts
to this codebase's ``(Nx, Ne)`` array convention and additionally returns
the two-term decomposition of the score:

.. math::

    \\mathrm{ES} = \\underbrace{\\frac{1}{N_e} \\sum_{m} \\| x_m - y \\|}_{\\text{accuracy}}
                 - \\underbrace{\\frac{1}{2 N_e^2} \\sum_{m,j} \\| x_m - x_j \\|}_{\\text{spread}}

The decomposition is the diagnostic of interest when studying ML-surrogate
variance collapse: a surrogate that under-disperses will show a near-zero
spread term, even when its accuracy term looks fine — a failure mode that
RMSE alone cannot detect.

Notes
-----
``scoringrules`` expects the ensemble axis at ``-2`` and the variable axis
at ``-1`` (i.e. shape ``(..., Ne, Nx)``). The convention used everywhere
else in this codebase is ``(Nx, Ne)``, so we transpose at the boundary.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
import scoringrules as sr


def energy_score(ensemble, truth):
    """Compute the Energy Score and its accuracy/spread decomposition.

    Parameters
    ----------
    ensemble : np.ndarray, shape (Nx, Ne)
        Ensemble of forecasts. Variable-first / member-second, matching the
        convention used throughout this codebase (e.g. ``Xf_k``, ``Xa_k``).
    truth : np.ndarray, shape (Nx,)
        Verifying observation / true state.

    Returns
    -------
    es : float
        Total Energy Score (lower is better).
    acc : float
        Accuracy term, :math:`\\frac{1}{N_e} \\sum_m \\| x_m - y \\|`.
        Sensitive to forecast bias.
    spr : float
        Spread term, :math:`\\frac{1}{2 N_e^2} \\sum_{m,j} \\| x_m - x_j \\|`.
        Sensitive to ensemble dispersion. Collapses to ~0 when members
        coincide — the variance-collapse signature.

    Notes
    -----
    The total ES is computed via ``scoringrules.energy_score`` for the
    canonical, well-tested implementation. The decomposition is computed
    here directly with :func:`scipy.spatial.distance.cdist` so callers can
    attribute changes in ES to bias vs spread independently.

    All particles are weighted uniformly (1/Ne). For weighted ensembles
    (e.g. a particle filter prior to resampling), this measures the
    *cloud* properties, not the weighted distributional estimate.
    """
    ensemble = np.asarray(ensemble)
    truth = np.asarray(truth)

    if ensemble.ndim != 2:
        raise ValueError(
            f"ensemble must be 2-D (Nx, Ne); got shape {ensemble.shape}"
        )
    if truth.ndim != 1:
        raise ValueError(f"truth must be 1-D (Nx,); got shape {truth.shape}")
    if truth.shape[0] != ensemble.shape[0]:
        raise ValueError(
            f"variable dimension mismatch: ensemble has Nx={ensemble.shape[0]}, "
            f"truth has Nx={truth.shape[0]}"
        )

    # Guard: if any particle has NaN/Inf, the ES is ill-defined.
    if not np.all(np.isfinite(ensemble)):
        return np.nan, np.nan, np.nan

    # scoringrules layout: (..., Ne, Nx)
    ens_sr = ensemble.T  # (Ne, Nx)

    # Total ES via scoringrules (canonical implementation)
    es = float(sr.energy_score(truth, ens_sr))

    # Decomposition computed directly
    Ne = ensemble.shape[1]
    diffs = ens_sr - truth[None, :]                          # (Ne, Nx)
    acc = float(np.mean(np.linalg.norm(diffs, axis=1)))
    pdist = cdist(ens_sr, ens_sr)                            # (Ne, Ne)
    spr = float(pdist.sum() / (2.0 * Ne * Ne))

    return es, acc, spr
