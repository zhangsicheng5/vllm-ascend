import importlib.util
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.cpp_extension import load


def _load_cpu_sparse_attn():
    source = Path("vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/cpu_sparse_attn.cpp")
    return load(
        name="cpu_sparse_attn_lru_resident_test",
        sources=[str(source)],
        extra_cflags=["-O3", "-fopenmp"],
        extra_ldflags=["-fopenmp"],
        verbose=False,
    )


def _python_ref(req_ids, last_req_ids, topk_indices, slot_to_token, lru_slots, max_token):
    num_reqs, topk = topk_indices.shape
    capacity = slot_to_token.shape[1]
    current_slots = torch.full((num_reqs, topk), -1, dtype=torch.int32)
    miss_count = torch.zeros((num_reqs,), dtype=torch.int32)
    miss_tokens = torch.full((num_reqs, topk), -1, dtype=torch.int32)
    miss_slots = torch.full((num_reqs, topk), -1, dtype=torch.int32)
    for row in range(num_reqs):
        if last_req_ids[row].item() != req_ids[row].item():
            slot_to_token[row].fill_(-1)
            lru_slots[row].copy_(torch.arange(capacity, dtype=torch.int32))
            last_req_ids[row] = req_ids[row]
        wanted = {}
        for pos, token in enumerate(topk_indices[row].tolist()):
            if 0 <= token < max_token and token not in wanted:
                wanted[token] = pos
        hit_slots = []
        evictable = []
        for slot in lru_slots[row].tolist():
            token = slot_to_token[row, slot].item()
            if token in wanted:
                pos = wanted[token]
                current_slots[row, pos] = slot
                hit_slots.append(slot)
            else:
                evictable.append(slot)
        misses = []
        for pos, token in enumerate(topk_indices[row].tolist()):
            if 0 <= token < max_token and current_slots[row, pos].item() < 0:
                misses.append((pos, token))
        for idx, (pos, token) in enumerate(misses[:len(evictable)]):
            slot = evictable[idx]
            slot_to_token[row, slot] = token
            current_slots[row, pos] = slot
            miss_tokens[row, idx] = token
            miss_slots[row, idx] = slot
            miss_count[row] += 1
        kept_evictable = evictable[miss_count[row].item():]
        new_order = kept_evictable + miss_slots[row, :miss_count[row]].tolist() + hit_slots
        lru_slots[row].copy_(torch.tensor(new_order, dtype=torch.int32))
    return current_slots, miss_count, miss_tokens, miss_slots


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch required")
@pytest.mark.skipif(sys.platform == "darwin", reason="OpenMP extension test runs on Linux/NPU CI")
def test_lru_resident_compact_fixed_shape_and_state():
    ext = _load_cpu_sparse_attn()
    num_reqs = 2
    topk = 4
    capacity = 6
    max_token = 64
    req_ids = torch.tensor([101, 202], dtype=torch.int64)
    last_req_ids = torch.tensor([-1, 202], dtype=torch.int64)
    topk_indices = torch.tensor([[1, 2, 3, 4], [10, 11, 12, 13]], dtype=torch.int32)
    slot_to_token = torch.tensor(
        [[-1, -1, -1, -1, -1, -1], [10, 99, 11, -1, -1, -1]],
        dtype=torch.int32,
    )
    lru_slots = torch.tensor(
        [[0, 1, 2, 3, 4, 5], [1, 3, 4, 5, 0, 2]],
        dtype=torch.int32,
    )
    expected_last = last_req_ids.clone()
    expected_slot = slot_to_token.clone()
    expected_lru = lru_slots.clone()
    exp_current, exp_count, exp_tokens, exp_slots = _python_ref(
        req_ids,
        expected_last,
        topk_indices,
        expected_slot,
        expected_lru,
        max_token,
    )
    current_slots = torch.full((num_reqs, topk), -1, dtype=torch.int32)
    miss_count = torch.zeros((num_reqs,), dtype=torch.int32)
    miss_tokens = torch.full((num_reqs, topk), -1, dtype=torch.int32)
    miss_slots = torch.full((num_reqs, topk), -1, dtype=torch.int32)
    workspace = torch.zeros((4, max_token), dtype=torch.int32)
    pos_workspace = torch.full((4, max_token), -1, dtype=torch.int32)
    slot_workspace = torch.empty((4, capacity * 3), dtype=torch.int32)
    miss_pos_workspace = torch.empty((4, topk), dtype=torch.int32)
    epochs = torch.zeros((4,), dtype=torch.int32)
    ext.lru_resident_compact(
        req_ids.data_ptr(),
        last_req_ids.data_ptr(),
        topk_indices.data_ptr(),
        slot_to_token.data_ptr(),
        lru_slots.data_ptr(),
        current_slots.data_ptr(),
        miss_count.data_ptr(),
        miss_tokens.data_ptr(),
        miss_slots.data_ptr(),
        workspace.data_ptr(),
        pos_workspace.data_ptr(),
        slot_workspace.data_ptr(),
        miss_pos_workspace.data_ptr(),
        epochs.data_ptr(),
        num_reqs,
        topk,
        capacity,
        max_token,
        4,
        1,
    )
    assert current_slots.shape == (num_reqs, topk)
    assert miss_tokens.shape == (num_reqs, topk)
    assert miss_slots.shape == (num_reqs, topk)
    torch.testing.assert_close(current_slots, exp_current)
    torch.testing.assert_close(miss_count, exp_count)
    torch.testing.assert_close(miss_tokens, exp_tokens)
    torch.testing.assert_close(miss_slots, exp_slots)
    torch.testing.assert_close(slot_to_token, expected_slot)
    torch.testing.assert_close(lru_slots, expected_lru)
    torch.testing.assert_close(last_req_ids, expected_last)
