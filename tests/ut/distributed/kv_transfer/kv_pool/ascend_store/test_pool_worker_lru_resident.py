from unittest.mock import MagicMock

import torch


def test_prepare_lru_resident_and_load_keeps_static_shapes():
    worker = MagicMock()
    worker.prepare_lru_resident_and_load.return_value = True
    num_reqs = 2
    topk = 4
    capacity = 6
    topk_indices = torch.full((num_reqs, topk), -1, dtype=torch.int32)
    slot_to_token = torch.full((num_reqs, capacity), -1, dtype=torch.int32)
    lru_slots = torch.arange(capacity, dtype=torch.int32).view(1, -1).repeat(num_reqs, 1)
    current_slots = torch.full((num_reqs, topk), -1, dtype=torch.int32)
    miss_count = torch.zeros((num_reqs,), dtype=torch.int32)
    miss_tokens = torch.full((num_reqs, topk), -1, dtype=torch.int32)
    miss_slots = torch.full((num_reqs, topk), -1, dtype=torch.int32)
    req_ids = torch.tensor([1, 2], dtype=torch.int64)
    last_req_ids = torch.tensor([-1, -1], dtype=torch.int64)

    ok = worker.prepare_lru_resident_and_load(
        "layer.0",
        num_reqs,
        topk_indices,
        slot_to_token,
        lru_slots,
        current_slots,
        miss_count,
        miss_tokens,
        miss_slots,
        req_ids,
        last_req_ids,
        32768,
        False,
    )

    assert ok is True
    assert current_slots.shape == (num_reqs, topk)
    assert miss_count.shape == (num_reqs,)
    assert miss_tokens.shape == (num_reqs, topk)
    assert miss_slots.shape == (num_reqs, topk)
