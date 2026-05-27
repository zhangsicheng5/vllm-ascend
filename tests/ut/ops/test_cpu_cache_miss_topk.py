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


def _offset_old(req_ids, old_raw):
    offsets = req_ids[:, None] * bench_cpu.REQ_ID_OFFSET_STRIDE
    return np.where(old_raw >= 0, old_raw + offsets, -1).astype(np.int64)


def test_reference_matches_spec_example_with_req_id_offset():
    req_ids = np.array([3], dtype=np.int64)
    old_raw = np.array([[3, 7, 15, 22, -1, -1]], dtype=np.int32)
    old = _offset_old(req_ids, old_raw)
    new = np.array([[5, 7, 18, 22, 30, 33]], dtype=np.int32)

    out = bench_cpu.reference_update_topk_indices(req_ids, old, new)

    np.testing.assert_array_equal(out,
                                  np.array([[5, -1, 18, -1, 30, 33]],
                                           dtype=np.int32))
    expected_old_raw = np.array([[5, 7, 18, 22, 30, 33]], dtype=np.int32)
    np.testing.assert_array_equal(old, _offset_old(req_ids, expected_old_raw))
    assert out is new


def test_reference_keeps_same_raw_token_isolated_by_req_id():
    req_ids = np.array([3, 7], dtype=np.int64)
    old_raw = np.array([
        [5, 10, -1, -1],
        [5, 20, -1, -1],
    ], dtype=np.int32)
    new = np.array([
        [5, 30, 31, -1],
        [5, 40, 41, -1],
    ], dtype=np.int32)
    old = _offset_old(req_ids, old_raw)

    out = bench_cpu.reference_update_topk_indices(req_ids, old, new)

    np.testing.assert_array_equal(out,
                                  np.array([
                                      [-1, 30, 31, -1],
                                      [-1, 40, 41, -1],
                                  ], dtype=np.int32))
    expected_old_raw = np.array([
        [5, 30, 31, -1],
        [5, 40, 41, -1],
    ], dtype=np.int32)
    np.testing.assert_array_equal(old, _offset_old(req_ids, expected_old_raw))


def test_reference_deduplicates_repeated_new_tokens():
    req_ids = np.array([11], dtype=np.int64)
    old_raw = np.array([[1, 2, 3, -1, -1, -1]], dtype=np.int32)
    old = _offset_old(req_ids, old_raw)
    new = np.array([[2, 4, 4, 5, 5, 6]], dtype=np.int32)

    out = bench_cpu.reference_update_topk_indices(req_ids, old, new)

    assert sorted(out[out >= 0].tolist()) == [4, 5, 6]
    expected_old_raw = np.array([[4, 2, 5, 6, -1, -1]], dtype=np.int32)
    assert sorted((old[old >= 0] -
                   req_ids[0] * bench_cpu.REQ_ID_OFFSET_STRIDE).tolist()) == [
                       2, 4, 5, 6]
    assert old[0, 0] == (
        expected_old_raw[0, 0] +
        req_ids[0] * bench_cpu.REQ_ID_OFFSET_STRIDE)


@pytest.mark.parametrize("num_reqs", [1, 2, 4])
@pytest.mark.parametrize("seq_len", [32768, 131072])
@pytest.mark.parametrize("hit_rate", [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
def test_generate_case_exact_overlap(num_reqs, seq_len, hit_rate):
    topk = 2048

    req_ids, old, new = bench_cpu.generate_case(
        num_reqs=num_reqs,
        seq_len=seq_len,
        topk=topk,
        hit_rate=hit_rate,
        seed=123,
    )

    assert req_ids.shape == (num_reqs,)
    assert old.shape == (num_reqs, topk)
    assert new.shape == (num_reqs, topk)
    assert req_ids.dtype == np.int64
    assert old.dtype == np.int64
    assert new.dtype == np.int32
    assert old.flags.c_contiguous
    assert new.flags.c_contiguous
    assert int(new.min()) >= 0
    assert int(new.max()) < seq_len

    expected_overlap = int(round(topk * hit_rate))
    for row in range(num_reqs):
        offset = req_ids[row] * bench_cpu.REQ_ID_OFFSET_STRIDE
        old_raw = old[row] - offset
        assert int(old_raw.min()) >= 0
        assert int(old_raw.max()) < seq_len
        overlap = len(set(old_raw.tolist()) & set(new[row].tolist()))
        assert overlap == expected_overlap


def test_numpy_baseline_matches_reference_with_req_ids():
    req_ids, old, new = bench_cpu.generate_case(
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

    expected = bench_cpu.reference_update_topk_indices(req_ids, ref_old,
                                                       ref_new)
    actual = bench_cpu.numpy_isin_update_topk_indices(req_ids, np_old, np_new)

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(np_old, ref_old)
    assert actual is np_new


def test_numba_stamp_matches_reference_when_numba_available():
    if bench_cpu.NUMBA_AVAILABLE is False:
        pytest.skip("numba is not installed in this environment")

    req_ids, old, new = bench_cpu.generate_case(
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

    expected = bench_cpu.reference_update_topk_indices(req_ids, ref_old,
                                                       ref_new)
    workspace = bench_cpu.make_topk_workspace(topk=512, max_token=32768)
    actual = bench_cpu.update_topk_indices_cpu(req_ids, nb_old, nb_new,
                                               workspace)

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(nb_old, ref_old)
    assert actual is nb_new
