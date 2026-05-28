import numpy as np
import pytest

from vllm_ascend.attention.cpu_cache_miss_topk import (
    NUMBA_AVAILABLE,
    REQ_ID_OFFSET_STRIDE,
    make_cpu_cache_miss_topk_workspace,
    update_topk_indices_cpu,
)


def _offset_old(req_ids, old_raw):
    offsets = req_ids[:, None] * REQ_ID_OFFSET_STRIDE
    return np.where(old_raw >= 0, old_raw + offsets, -1).astype(np.int64)


def test_cpu_topk_matches_spec_example_with_req_id_offset():
    if not NUMBA_AVAILABLE:
        pytest.skip("numba is required for CPU cache-miss topk")

    req_ids = np.array([3], dtype=np.int64)
    old_raw = np.array([[3, 7, 15, 22, -1, -1]], dtype=np.int32)
    old = _offset_old(req_ids, old_raw)
    new = np.array([[5, 7, 18, 22, 30, 33]], dtype=np.int32)
    workspace = make_cpu_cache_miss_topk_workspace(topk=6, max_token=128)

    out = update_topk_indices_cpu(req_ids, old, new, workspace)

    np.testing.assert_array_equal(
        out, np.array([[5, -1, 18, -1, 30, 33]], dtype=np.int32))
    expected_old_raw = np.array([[5, 7, 18, 22, 30, 33]], dtype=np.int32)
    np.testing.assert_array_equal(old, _offset_old(req_ids, expected_old_raw))
    assert out is new


def test_cpu_topk_keeps_same_raw_token_isolated_by_req_id():
    if not NUMBA_AVAILABLE:
        pytest.skip("numba is required for CPU cache-miss topk")

    req_ids = np.array([3, 7], dtype=np.int64)
    old_raw = np.array([
        [5, 10, -1, -1],
        [5, 20, -1, -1],
    ],
                       dtype=np.int32)
    old = _offset_old(req_ids, old_raw)
    new = np.array([
        [5, 30, 31, -1],
        [5, 40, 41, -1],
    ],
                   dtype=np.int32)
    workspace = make_cpu_cache_miss_topk_workspace(topk=4, max_token=128)

    out = update_topk_indices_cpu(req_ids, old, new, workspace)

    np.testing.assert_array_equal(
        out,
        np.array([
            [-1, 30, 31, -1],
            [-1, 40, 41, -1],
        ],
                 dtype=np.int32))
    expected_old_raw = np.array([
        [5, 30, 31, -1],
        [5, 40, 41, -1],
    ],
                                dtype=np.int32)
    np.testing.assert_array_equal(old, _offset_old(req_ids, expected_old_raw))


def test_cpu_topk_rejects_wrong_old_dtype():
    if not NUMBA_AVAILABLE:
        pytest.skip("numba is required for CPU cache-miss topk")

    req_ids = np.array([0], dtype=np.int64)
    old = np.array([[1, 2]], dtype=np.int32)
    new = np.array([[2, 3]], dtype=np.int32)
    workspace = make_cpu_cache_miss_topk_workspace(topk=2, max_token=16)

    with pytest.raises(TypeError, match="topk_indices_old must be int64"):
        update_topk_indices_cpu(req_ids, old, new, workspace)
