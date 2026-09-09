# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Metadata and cache ownership for generalized nano Q1/MTP offload.

Registered host KV supplies sparse misses and bounded circular HBM tails.
Colocated execution may additionally retain a full device cache for prefill.
"""

from dataclasses import dataclass

import torch

TOPK = 2048
COPY_MISS_CAPACITY = 32768
LIM_MISS_CAPACITY = COPY_MISS_CAPACITY
INVALID_SLOT = -(1 << 31)
REQUEST_STATE_NON_OFFLOAD = -3
REQUEST_STATE_FIRST_OFFLOAD = -2
REQUEST_STATE_STEADY = -1


def prepare_copy_sfa_queries(query, query_rope):
    """Prepare contiguous queries for native 8/16/32/64/128-head tiles.

    MLA shares KV across query heads, and softmax is independent per head.
    Zero-filled extra query heads cannot affect the original heads; callers
    must return only the original head range from the kernel output.
    """
    heads = query.shape[1]
    if 1 <= heads < 8:
        padded_query = query.new_zeros((query.shape[0], 8, query.shape[2]))
        padded_rope = query_rope.new_zeros((query_rope.shape[0], 8, query_rope.shape[2]))
        padded_query[:, :heads].copy_(query)
        padded_rope[:, :heads].copy_(query_rope)
        return padded_query, padded_rope
    if heads not in (8, 16, 32, 64, 128):
        raise ValueError(
            f"Generalized copy-SFA serving requires 1–8, 16, 32, 64 or 128 query heads per rank, got {heads}"
        )
    return query.contiguous(), query_rope.contiguous()


@dataclass
class MtpBatch:
    query_ends: torch.Tensor
    seq_lens: torch.Tensor
    offload_lens: torch.Tensor
    cache_tokens: torch.Tensor
    pool_entries: torch.Tensor
    hbm_block_table: torch.Tensor
    source_block_table: torch.Tensor
    tail_sources: torch.Tensor
    tail_destinations: torch.Tensor
    pool_rows: list[int]
    prefix_lengths: list[int]
    cache_sizes: list[int]
    num_tokens: int
    graph_buffers: object | None = None


def make_mtp_batch(metadata, manager) -> MtpBatch | None:
    """Snapshot scheduled lengths, including rejection-adjusted drafts."""
    if metadata.num_prefills or not metadata.num_decodes:
        return None
    count = metadata.num_decodes
    # Builders reuse these buffers for target and draft steps. Keep an owned
    # snapshot: .to().contiguous() can retain a view whose values later change
    # while this batch's Python lengths and queued custom calls stay fixed.
    query_ends = (
        metadata.cum_query_lens[:count]
        .to(dtype=torch.int32)
        .clone(
            memory_format=torch.contiguous_format,
        )
    )
    ends = query_ends.cpu().tolist()
    widths = [end - start for start, end in zip([0] + ends[:-1], ends)]
    seq_lens = (
        metadata.seq_lens[:count]
        .to(dtype=torch.int32)
        .clone(
            memory_format=torch.contiguous_format,
        )
    )
    lengths = seq_lens.cpu().tolist()
    block_size = manager.block_size
    prefixes = [(length - width) // block_size * block_size for length, width in zip(lengths, widths)]
    if any(width < 1 or width > 7 for width in widths):
        return None
    if any(prefix < TOPK for prefix in prefixes):
        return None
    caches = [min(prefix, manager.topk_buffer_size) for prefix in prefixes]
    for width, prefix, cache in zip(widths, prefixes, caches):
        if (prefix <= width * TOPK and cache != prefix) or (
            prefix > width * TOPK and not width * TOPK <= cache <= 16256
        ):
            raise ValueError(f"Invalid generalized LIM budget: Q={width}, L={prefix}, C={cache}")
    pools = (
        metadata.req_topk_buffer_slots[:count]
        .to(dtype=torch.int32)
        .clone(
            memory_format=torch.contiguous_format,
        )
    )
    pool_rows = pools.cpu().tolist()
    if len(set(pool_rows)) != count or any(pool < 0 or pool >= manager.max_num_reqs for pool in pool_rows):
        raise ValueError("Generalized MTP requests must occupy distinct pool rows")
    device = seq_lens.device
    source_block_table = metadata.block_table[:count].clone(memory_format=torch.contiguous_format)
    stride_blocks = manager.topk_buffer_size // block_size + 2
    table = torch.zeros((count, stride_blocks), dtype=torch.int32)
    source_rows, source_positions, destinations = [], [], []
    for row, (pool, prefix, length, cache) in enumerate(zip(pool_rows, prefixes, lengths, caches)):
        base = pool * stride_blocks
        cache_blocks = cache // block_size
        table[row, :cache_blocks] = torch.arange(cache_blocks, dtype=torch.int32) + base
        tail_start = base + stride_blocks - 2
        for tail_block in range(2):
            table[row, cache_blocks + tail_block] = tail_start + (prefix // block_size + tail_block) % 2
        for position in range(prefix, length):
            source_rows.append(row)
            source_positions.append(position)
            destinations.append(
                pool * stride_blocks * block_size + manager.topk_buffer_size + position % (2 * block_size)
            )
    positions = torch.tensor(source_positions, dtype=torch.int64, device=device)
    rows = torch.tensor(source_rows, dtype=torch.int64, device=device)
    source_blocks = source_block_table[rows, positions // block_size].to(torch.int64)
    return MtpBatch(
        query_ends,
        seq_lens,
        torch.tensor(prefixes, dtype=torch.int32, device=device),
        torch.tensor(caches, dtype=torch.int32, device=device),
        pools,
        table.to(device),
        source_block_table,
        source_blocks * block_size + positions % block_size,
        torch.tensor(destinations, dtype=torch.int64, device=device),
        pool_rows,
        prefixes,
        caches,
        ends[-1],
    )


class GeneralizedMtpRuntime:
    def __init__(self, manager):
        self.manager = manager
        self.maps = {}
        self.residents = {}
        self.outputs = None
        self.output_batch = None

    def invalidate(self):
        # A prefill or mixed step may rewrite cached tokens or recycle rows.
        self.residents.clear()
        self.output_batch = None

    def prepare_lim(self, layer_name, batch, source_capacity, device):
        if batch.graph_buffers is not None:
            return batch.graph_buffers.lim_inputs(layer_name)
        manager = self.manager
        mapping = self.maps.get(layer_name)
        if mapping is None:
            mapping = torch.full(
                (getattr(manager, "max_num_topk_rows", manager.max_num_reqs), source_capacity),
                INVALID_SLOT,
                dtype=torch.int32,
                device=device,
            )
            self.maps[layer_name] = mapping
        if mapping.shape[1] != source_capacity:
            raise ValueError("Generalized LIM source capacity changed after cache allocation")
        owners = {slot: req for req, slot in manager.topk_buffer_slot_manager.req2slot.items()}
        resident = self.residents.setdefault(layer_name, {})
        states = []
        for pool, prefix, cache in zip(batch.pool_rows, batch.prefix_lengths, batch.cache_sizes):
            owner = (owners[pool], manager.nano_mtp_slot_generations.get(pool, 0))
            previous = resident.get(pool)
            ready = previous is not None and previous[:2] == (owner, cache) and previous[2] <= prefix
            states.append(REQUEST_STATE_STEADY if ready else REQUEST_STATE_FIRST_OFFLOAD)
            resident[pool] = (owner, cache, prefix)
        count, tokens = len(batch.pool_rows), batch.num_tokens

        def empty(shape):
            return torch.empty(shape, dtype=torch.int32, device=device)

        self.outputs = (
            empty((tokens, 1, TOPK)),
            empty((tokens, 1, TOPK)),
            empty((tokens,)),
            empty((count, LIM_MISS_CAPACITY)),
            empty((count, LIM_MISS_CAPACITY)),
            empty((count,)),
        )
        self.output_batch = batch
        return mapping, torch.tensor(states, dtype=torch.int32, device=device), self.outputs

    def require_outputs(self, batch):
        if batch.graph_buffers is not None:
            return batch.graph_buffers.outputs
        if self.output_batch is not batch or self.outputs is None:
            raise RuntimeError("Shared MTP attention ran before its indexer owner for this batch")
        return self.outputs

    def copy_metadata(self, batch):
        # Both kernels use contiguous int32[B, 32768]. Shared-indexer
        # consumers read the same valid miss-count prefixes without a bridge.
        return self.require_outputs(batch)
