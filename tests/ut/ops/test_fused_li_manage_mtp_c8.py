"""Registered generalized MTP C8 operator Meta dispatch contract.

Mirrors ``test_fused_li_manage_mtp.py`` but with the C8 ABI: int8
query/index_key_cache and fp16 per-token dequant scales. The in-place
operator returns ``None`` and writes the seven caller-supplied buffers.
"""

import torch

from vllm_ascend.utils import enable_custom_op

enable_custom_op()


def tensor(*shape, dtype=torch.int32):
    return torch.empty(shape, dtype=dtype, device="meta")


def test_generalized_lim_c8_meta_accepts_int8_query_and_fp16_scales():
    batch, tokens, heads, blocks = 3, 11, 32, 128
    inputs = [
        tensor(tokens, heads, dtype=torch.bfloat16),  # index_weights
        tensor(tokens, heads, dtype=torch.float16),  # query_dequant_scale
        tensor(tokens, heads, 128, dtype=torch.int8),  # query
        tensor(blocks, 128, 1, dtype=torch.float16),  # index_key_dequant_scale
        tensor(blocks, 128, 1, 128, dtype=torch.int8),  # index_key_cache
        tensor(batch, blocks),  # index_block_table
        *[tensor(batch) for _ in range(6)],  # 6 [B] metadata vectors
        tensor(7, blocks * 128),  # cache_slots_pool
        tensor(tokens, 1, 2048),  # topk_src_ids
        tensor(tokens, 1, 2048),  # topk_dst_slots
        tensor(tokens),  # topk_miss_counts
        tensor(batch, 32768),  # miss_src_ids
        tensor(batch, 32768),  # miss_dst_slots
        tensor(batch),  # miss_counts
    ]
    op = torch.ops._C_ascend.npu_fused_li_manage_mtp_c8.default
    assert op(*inputs) is None
    writes = {arg.name for arg in op._schema.arguments if arg.alias_info is not None and arg.alias_info.is_write}
    assert writes == {
        "cache_slots_pool",
        "topk_src_ids",
        "topk_dst_slots",
        "topk_miss_counts",
        "miss_src_ids",
        "miss_dst_slots",
        "miss_counts",
    }
