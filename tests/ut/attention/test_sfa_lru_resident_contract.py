import pytest
import torch

from vllm_ascend.ascend_config import LRUResidentCacheConfig
from vllm_ascend.attention.sfa_v1 import (
    _build_cpu_sparse_indices_from_slots,
    _normalize_sfa_lse,
)


def test_lru_resident_cache_config_defaults_disabled():
    cfg = LRUResidentCacheConfig({})
    assert cfg.enabled is False
    assert cfg.buffer_size == 2048
    assert cfg.topk == 2048


def test_lru_resident_cache_config_accepts_4096_capacity():
    cfg = LRUResidentCacheConfig({
        "enabled": True,
        "buffer_size": 4096,
        "topk": 2048,
    })
    assert cfg.enabled is True
    assert cfg.buffer_size == 4096
    assert cfg.topk == 2048


def test_lru_resident_cache_config_rejects_capacity_smaller_than_topk():
    with pytest.raises(ValueError, match="buffer_size must be >= topk"):
        LRUResidentCacheConfig({
            "enabled": True,
            "buffer_size": 1024,
            "topk": 2048,
        })


def test_build_cpu_sparse_indices_from_slots_static_shape():
    current_slots = torch.tensor([[5, -1, 2, 9]], dtype=torch.int32)
    cpu_mask = torch.tensor([[True, False, True, False]])
    result = _build_cpu_sparse_indices_from_slots(current_slots, cpu_mask)
    assert result.shape == (1, 1, 4)
    torch.testing.assert_close(
        result,
        torch.tensor([[[5, -1, 2, -1]]], dtype=torch.int32),
    )


def test_normalize_sfa_lse_from_tnd_stats_to_token_head():
    softmax_max = torch.zeros((1, 2, 64), dtype=torch.float32)
    softmax_sum = torch.ones((1, 2, 64), dtype=torch.float32)
    lse = _normalize_sfa_lse(softmax_max, softmax_sum, num_tokens=2, num_heads=64)
    assert lse.shape == (2, 64)
    torch.testing.assert_close(lse, torch.zeros((2, 64), dtype=torch.float32))
