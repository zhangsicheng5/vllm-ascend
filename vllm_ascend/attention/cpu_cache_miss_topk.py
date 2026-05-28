from dataclasses import dataclass

import numpy as np

try:
    from numba import get_num_threads, get_thread_id, njit, prange

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    get_num_threads = None
    get_thread_id = None
    njit = None
    prange = range


OLD_FLAG = 1
NEW_FLAG = 2
BOTH_FLAG = 3
REQ_ID_OFFSET_STRIDE = 1 << 16
NUMBA_PARALLEL = True


@dataclass
class CPUCacheMissTopKWorkspace:
    mark_workspace: np.ndarray
    miss_workspace: np.ndarray
    epochs: np.ndarray
    topk: int
    max_token: int


if NUMBA_AVAILABLE:

    @njit(inline="always")
    def _has_old(mark, token, base):
        marker = mark[token]
        return marker == base + OLD_FLAG or marker == base + BOTH_FLAG

    @njit(inline="always")
    def _has_new(mark, token, base):
        marker = mark[token]
        return marker == base + NEW_FLAG or marker == base + BOTH_FLAG

    @njit(inline="always")
    def _set_old(mark, token, base):
        marker = mark[token]
        if marker == base + NEW_FLAG:
            mark[token] = base + BOTH_FLAG
        elif marker != base + OLD_FLAG and marker != base + BOTH_FLAG:
            mark[token] = base + OLD_FLAG

    @njit(inline="always")
    def _set_new(mark, token, base):
        marker = mark[token]
        if marker == base + OLD_FLAG:
            mark[token] = base + BOTH_FLAG
        elif marker != base + NEW_FLAG and marker != base + BOTH_FLAG:
            mark[token] = base + NEW_FLAG

    @njit(nogil=True, parallel=NUMBA_PARALLEL, boundscheck=False)
    def _update_topk_cache_miss_inplace_numba(
        req_ids_tensor,
        topk_indices_old,
        topk_indices_new,
        mark_workspace,
        miss_workspace,
        epochs,
    ):
        bs, topk = topk_indices_old.shape
        max_token = mark_workspace.shape[1]

        for row in prange(bs):
            tid = get_thread_id() if NUMBA_PARALLEL else 0
            base = epochs[tid] + 4

            if base >= 2_147_483_000:
                for token in range(max_token):
                    mark_workspace[tid, token] = 0
                base = 4

            epochs[tid] = base
            mark = mark_workspace[tid]
            miss_buffer = miss_workspace[tid]
            offset = req_ids_tensor[row] * REQ_ID_OFFSET_STRIDE

            for slot in range(topk):
                value = topk_indices_old[row, slot]
                if value >= 0:
                    raw_value = value - offset
                    if raw_value >= 0 and raw_value < max_token:
                        _set_old(mark, raw_value, base)

            miss_count = 0
            for slot in range(topk):
                value = topk_indices_new[row, slot]
                if value >= 0:
                    _set_new(mark, value, base)
                    if not _has_old(mark, value, base):
                        miss_buffer[miss_count] = value
                        miss_count += 1
                        _set_old(mark, value, base)

            miss_pos = 0
            for slot in range(topk):
                topk_indices_new[row, slot] = -1
                old_value = topk_indices_old[row, slot]
                old_raw = old_value - offset if old_value >= 0 else -1
                if old_raw >= 0 and not _has_new(mark, old_raw, base):
                    if miss_pos < miss_count:
                        replacement = miss_buffer[miss_pos]
                        miss_pos += 1
                        topk_indices_old[row, slot] = replacement + offset
                        topk_indices_new[row, slot] = replacement
                    else:
                        topk_indices_old[row, slot] = -1

            if miss_pos < miss_count:
                for slot in range(topk):
                    if topk_indices_old[row, slot] == -1:
                        replacement = miss_buffer[miss_pos]
                        miss_pos += 1
                        topk_indices_old[row, slot] = replacement + offset
                        topk_indices_new[row, slot] = replacement
                        if miss_pos >= miss_count:
                            break

        return topk_indices_new


def make_cpu_cache_miss_topk_workspace(
    topk: int,
    max_token: int,
) -> CPUCacheMissTopKWorkspace:
    if NUMBA_AVAILABLE is False:
        raise RuntimeError("numba is required for CPU cache-miss topk")

    num_threads = get_num_threads() if NUMBA_PARALLEL else 1
    return CPUCacheMissTopKWorkspace(
        mark_workspace=np.zeros((num_threads, max_token), dtype=np.int32),
        miss_workspace=np.empty((num_threads, topk), dtype=np.int32),
        epochs=np.zeros((num_threads, ), dtype=np.int32),
        topk=topk,
        max_token=max_token,
    )


def update_topk_indices_cpu(
    req_ids_tensor: np.ndarray,
    topk_indices_old: np.ndarray,
    topk_indices_new: np.ndarray,
    workspace: CPUCacheMissTopKWorkspace,
) -> np.ndarray:
    if NUMBA_AVAILABLE is False:
        raise RuntimeError("numba is required for CPU cache-miss topk")
    if req_ids_tensor.dtype != np.int64:
        raise TypeError("req_ids_tensor must be int64")
    if topk_indices_old.dtype != np.int64:
        raise TypeError("topk_indices_old must be int64")
    if topk_indices_new.dtype != np.int32:
        raise TypeError("topk_indices_new must be int32")
    if topk_indices_old.shape != topk_indices_new.shape:
        raise ValueError("old/new shape mismatch")
    if req_ids_tensor.shape[0] != topk_indices_old.shape[0]:
        raise ValueError("req_ids_tensor length must match batch size")
    if topk_indices_old.shape[1] != workspace.topk:
        raise ValueError("workspace topk mismatch")
    if not req_ids_tensor.flags.c_contiguous:
        raise ValueError("req_ids_tensor must be C-contiguous")
    if not topk_indices_old.flags.c_contiguous:
        raise ValueError("topk_indices_old must be C-contiguous")
    if not topk_indices_new.flags.c_contiguous:
        raise ValueError("topk_indices_new must be C-contiguous")

    return _update_topk_cache_miss_inplace_numba(
        req_ids_tensor,
        topk_indices_old,
        topk_indices_new,
        workspace.mark_workspace,
        workspace.miss_workspace,
        workspace.epochs,
    )
