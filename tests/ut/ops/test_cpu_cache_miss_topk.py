import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "benchmarks"
    / "ops"
    / "bench_cpu_cache_miss_topk.py"
)
SPEC = importlib.util.spec_from_file_location("bench_cpu_cache_miss_topk",
                                              MODULE_PATH)
bench_cpu = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bench_cpu
assert SPEC.loader is not None
SPEC.loader.exec_module(bench_cpu)


def test_reference_matches_spec_example():
    old = np.array([[3, 7, 15, 22, -1, -1]], dtype=np.int32)
    new = np.array([[5, 7, 18, 22, 30, 33]], dtype=np.int32)

    out = bench_cpu.reference_update_topk_indices(old, new)

    np.testing.assert_array_equal(out,
                                  np.array([[5, -1, 18, -1, 30, 33]],
                                           dtype=np.int32))
    np.testing.assert_array_equal(old,
                                  np.array([[5, 7, 18, 22, 30, 33]],
                                           dtype=np.int32))
    assert out is new


def test_reference_deduplicates_repeated_new_tokens():
    old = np.array([[1, 2, 3, -1, -1, -1]], dtype=np.int32)
    new = np.array([[2, 4, 4, 5, 5, 6]], dtype=np.int32)

    out = bench_cpu.reference_update_topk_indices(old, new)

    assert sorted(out[out >= 0].tolist()) == [4, 5, 6]
    assert sorted(old[old >= 0].tolist()) == [2, 4, 5, 6]


@pytest.mark.parametrize("num_reqs", [1, 2, 4])
@pytest.mark.parametrize("seq_len", [32768, 131072])
@pytest.mark.parametrize("hit_rate", [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
def test_generate_case_exact_overlap(num_reqs, seq_len, hit_rate):
    topk = 2048

    old, new = bench_cpu.generate_case(
        num_reqs=num_reqs,
        seq_len=seq_len,
        topk=topk,
        hit_rate=hit_rate,
        seed=123,
    )

    assert old.shape == (num_reqs, topk)
    assert new.shape == (num_reqs, topk)
    assert old.dtype == np.int32
    assert new.dtype == np.int32
    assert old.flags.c_contiguous
    assert new.flags.c_contiguous
    assert int(old.min()) >= 0
    assert int(new.min()) >= 0
    assert int(old.max()) < seq_len
    assert int(new.max()) < seq_len

    expected_overlap = int(round(topk * hit_rate))
    for row in range(num_reqs):
        overlap = len(set(old[row].tolist()) & set(new[row].tolist()))
        assert overlap == expected_overlap


def test_numpy_baseline_matches_reference():
    old, new = bench_cpu.generate_case(
        num_reqs=4,
        seq_len=32768,
        topk=256,
        hit_rate=0.3,
        seed=7,
    )
    ref_old = old.copy()
    ref_new = new.copy()
    np_old = old.copy()
    np_new = new.copy()

    expected = bench_cpu.reference_update_topk_indices(ref_old, ref_new)
    actual = bench_cpu.numpy_isin_update_topk_indices(np_old, np_new)

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(np_old, ref_old)
    assert actual is np_new


def test_numba_stamp_matches_reference_when_numba_available():
    if bench_cpu.NUMBA_AVAILABLE is False:
        pytest.skip("numba is not installed in this environment")

    old, new = bench_cpu.generate_case(
        num_reqs=4,
        seq_len=32768,
        topk=512,
        hit_rate=0.7,
        seed=11,
    )
    ref_old = old.copy()
    ref_new = new.copy()
    nb_old = old.copy()
    nb_new = new.copy()

    expected = bench_cpu.reference_update_topk_indices(ref_old, ref_new)
    workspace = bench_cpu.make_topk_workspace(topk=512, max_token=32768)
    actual = bench_cpu.update_topk_indices_cpu(nb_old, nb_new, workspace)

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(nb_old, ref_old)
    assert actual is nb_new
