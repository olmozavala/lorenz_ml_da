"""Tests for multi-node IC slicing (mirrors ms_denkf._slice_initial_conditions)."""

import numpy as np


def _slice_initial_conditions(master_seed, task_id, ntasks, ic_per_node):
    """Copy of ms_denkf._slice_initial_conditions — keep in sync."""
    global_n_ic = ntasks * ic_per_node
    rng = np.random.default_rng(master_seed)
    all_ics = rng.standard_normal((global_n_ic, 3))
    start = task_id * ic_per_node
    end = start + ic_per_node
    return all_ics[start:end], global_n_ic


def test_slice_covers_all_ics_without_overlap():
    master_seed = 132
    ntasks = 3
    ic_per_node = 2
    slices = [
        _slice_initial_conditions(master_seed, task_id, ntasks, ic_per_node)[0]
        for task_id in range(ntasks)
    ]
    stacked = np.vstack(slices)
    single, global_n = _slice_initial_conditions(master_seed, 0, 1, ntasks * ic_per_node)
    assert global_n == ntasks * ic_per_node
    np.testing.assert_array_equal(stacked, single)
    assert len({tuple(row) for row in stacked}) == global_n


def test_single_process_matches_multi_task_union():
    master_seed = 42
    ntasks = 10
    ic_per_node = 100
    all_single, _ = _slice_initial_conditions(master_seed, 0, 1, ntasks * ic_per_node)
    parts = [
        _slice_initial_conditions(master_seed, t, ntasks, ic_per_node)[0]
        for t in range(ntasks)
    ]
    np.testing.assert_array_equal(np.vstack(parts), all_single)


def test_different_tasks_get_different_slices():
    a, _ = _slice_initial_conditions(7, 0, 3, 5)
    b, _ = _slice_initial_conditions(7, 1, 3, 5)
    assert not np.allclose(a, b)
