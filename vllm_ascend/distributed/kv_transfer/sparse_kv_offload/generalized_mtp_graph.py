# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Persistent inputs for full-decode generalized MTP graph replay.

Request lifecycle decisions run before replay. The captured kernels consume
only the tensors owned here; a Python object captured in an earlier iteration
must never be the source of current request ownership or sequence lengths.
"""

from dataclasses import fields, replace
from types import SimpleNamespace

import torch

from vllm_ascend.attention.indexer import AscendSFAIndexerMetadata

from .generalized_mtp import (
    COPY_MISS_CAPACITY,
    INVALID_SLOT,
    LIM_MISS_CAPACITY,
    REQUEST_STATE_FIRST_OFFLOAD,
    REQUEST_STATE_NON_OFFLOAD,
    REQUEST_STATE_STEADY,
    TOPK,
    MtpBatch,
    make_mtp_batch,
)
from .nano_cache import KV_COMPONENTS, TAIL_BLOCKS, CopyDescriptors


class MtpGraphBuffers:
    """One fixed request/query capacity, owned by one captured draft step.

    Inactive rows have private hot-cache pool entries beyond the scheduler's
    request pool. They run valid dummy queries, never mutating a live request.
    Non-offload LIM produces zero request/query misses for these rows. Their
    HBM hot cache and tails are initialized at allocation, so copy-SFA can
    execute dummy attention without fetching host KV. The model's token
    padding is independent of this operator capacity.
    """

    def __init__(self, runtime, *, query_width, source_capacity, device, request_capacity=None):
        self.runtime = runtime
        self.manager = manager = runtime.manager
        self.requests = request_capacity if request_capacity is not None else manager.max_num_reqs
        self.query_width = query_width
        self.tokens = self.requests * query_width
        self.source_capacity = source_capacity
        self.device = device
        # The existing MTP allocation has Q_max hot rows per request. Reserve
        # one additional row per request for graph padding, outside req2slot.
        if manager.max_num_topk_rows < manager.max_num_reqs + self.requests:
            raise ValueError("MTP graph padding requires a private hot-cache row per request")
        self.layers = {}
        self.tail_copies = {}

        def zeros(shape, dtype=torch.int32):
            return torch.zeros(shape, dtype=dtype, device=device)

        stride = manager.topk_buffer_size // manager.block_size + 2
        self.batch = MtpBatch(
            query_ends=torch.arange(1, self.requests + 1, dtype=torch.int32, device=device) * query_width,
            seq_lens=zeros(self.requests),
            offload_lens=zeros(self.requests),
            cache_tokens=zeros(self.requests),
            pool_entries=zeros(self.requests),
            hbm_block_table=zeros((self.requests, stride)),
            source_block_table=zeros((self.requests, source_capacity // manager.block_size)),
            tail_sources=zeros(self.requests * 2 * manager.block_size, torch.int64),
            tail_destinations=zeros(self.requests * 2 * manager.block_size, torch.int64),
            pool_rows=[],
            prefix_lengths=[],
            cache_sizes=[],
            num_tokens=self.tokens,
            graph_buffers=self,
        )
        self.tail_valid = zeros(self.requests * 2 * manager.block_size, torch.bool)
        self.active_mask = zeros(self.requests, torch.bool)
        self.outputs = (
            zeros((self.tokens, 1, TOPK)),
            zeros((self.tokens, 1, TOPK)),
            zeros(self.tokens),
            zeros((self.requests, LIM_MISS_CAPACITY)),
            zeros((self.requests, LIM_MISS_CAPACITY)),
            zeros(self.requests),
        )
        self.copy_sources = torch.full(
            (self.requests, COPY_MISS_CAPACITY),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self.copy_destinations = torch.full_like(self.copy_sources, -1)
        self.active_requests = 0

    def accepts(self, batch):
        if batch is None or len(batch.pool_rows) > self.requests:
            return False
        count = len(batch.pool_rows)
        # FULL_DECODE_ONLY captures uniform decode. Variable-width batches
        # remain eager rather than reusing a graph with different segmentation.
        expected = [self.query_width * (row + 1) for row in range(count)]
        return batch.query_ends.cpu().tolist() == expected

    def update(self, batch, *, is_capture=False):
        """Refresh graph inputs outside capture, on the replay's input stream."""
        if torch.npu.is_current_stream_capturing():
            raise RuntimeError("MTP graph input preparation must run before capture/replay")
        if not self.accepts(batch):
            raise ValueError("MTP batch does not match the graph's uniform-query capacity")
        manager, graph = self.manager, self.batch
        # Startup capture has no live requests. Select private rows before
        # constructing tables, just as for padding during normal replay.
        count = 0 if is_capture else len(batch.pool_rows)
        self.active_requests = count
        self.active_mask.zero_()
        self.active_mask[:count].fill_(True)
        scratch = list(range(manager.max_num_reqs + count, manager.max_num_reqs + self.requests))
        pools = batch.pool_rows[:count] + scratch
        prefixes = batch.prefix_lengths[:count] + [manager.topk_buffer_size] * (self.requests - count)
        caches = batch.cache_sizes[:count] + [manager.topk_buffer_size] * (self.requests - count)
        lengths = batch.seq_lens[:count].cpu().tolist() + [manager.topk_buffer_size + self.query_width] * (
            self.requests - count
        )
        graph.pool_rows, graph.prefix_lengths, graph.cache_sizes = pools, prefixes, caches
        for dst, values in (
            (graph.seq_lens, lengths),
            (graph.offload_lens, prefixes),
            (graph.cache_tokens, caches),
            (graph.pool_entries, pools),
        ):
            dst.copy_(torch.tensor(values, dtype=dst.dtype, device=self.device))
        graph.source_block_table.zero_()
        graph.source_block_table[:count].copy_(batch.source_block_table[:count])
        block_size = manager.block_size
        stride_blocks = manager.topk_buffer_size // block_size + 2
        table = torch.zeros_like(graph.hbm_block_table, device="cpu")
        destinations = []
        for row, (pool, prefix, cache) in enumerate(zip(pools, prefixes, caches)):
            base = pool * stride_blocks
            cache_blocks = cache // block_size
            table[row, :cache_blocks] = torch.arange(cache_blocks, dtype=torch.int32) + base
            for tail_block in range(2):
                table[row, cache_blocks + tail_block] = (
                    base + stride_blocks - 2 + (prefix // block_size + tail_block) % 2
                )
            destinations.extend(
                pool * stride_blocks * block_size + manager.topk_buffer_size + position % (2 * block_size)
                for position in range(prefix, prefix + 2 * block_size)
            )
        graph.hbm_block_table.copy_(table.to(self.device))
        positions = graph.offload_lens.to(torch.int64)[:, None] + torch.arange(2 * block_size, device=self.device)
        valid = positions < graph.seq_lens[:, None]
        valid[count:] = False
        safe_positions = positions.clamp(max=self.source_capacity - 1)
        blocks = graph.source_block_table.gather(1, safe_positions // block_size).to(torch.int64)
        graph.tail_sources.copy_((blocks * block_size + safe_positions % block_size).flatten())
        graph.tail_destinations.copy_(torch.tensor(destinations, dtype=torch.int64, device=self.device))
        self.tail_valid.copy_(valid.flatten())

    def prepare_layer(self, layer_name):
        """Update lifecycle state before replay; allocate only before capture."""
        runtime, manager = self.runtime, self.manager
        mapping = runtime.maps.get(layer_name)
        if mapping is None:
            mapping = torch.full(
                (manager.max_num_topk_rows, self.source_capacity),
                INVALID_SLOT,
                dtype=torch.int32,
                device=self.device,
            )
            runtime.maps[layer_name] = mapping
        if mapping.shape != (manager.max_num_topk_rows, self.source_capacity):
            raise ValueError("MTP graph map capacity changed after allocation")
        if layer_name not in self.layers:
            self.layers[layer_name] = (mapping, torch.empty(self.requests, dtype=torch.int32, device=self.device))
            self.tail_copies[layer_name] = CopyDescriptors(
                self.requests * TAIL_BLOCKS * KV_COMPONENTS,
                self.device,
            )
        owners = {slot: req for req, slot in manager.topk_buffer_slot_manager.req2slot.items()}
        resident = runtime.residents.setdefault(layer_name, {})
        states = []
        for row, (pool, prefix, cache) in enumerate(
            zip(
                self.batch.pool_rows,
                self.batch.prefix_lengths,
                self.batch.cache_sizes,
            )
        ):
            if row >= self.active_requests:
                # Padding is not a new request: -2 would emit C misses on
                # every replay and force batch-wide first-fill H2D. -3 emits
                # zero request and per-query misses, using private HBM only.
                states.append(REQUEST_STATE_NON_OFFLOAD)
                continue
            owner = (owners[pool], manager.nano_mtp_slot_generations.get(pool, 0))
            previous = resident.get(pool)
            ready = previous is not None and previous[:2] == (owner, cache) and previous[2] <= prefix
            states.append(REQUEST_STATE_STEADY if ready else REQUEST_STATE_FIRST_OFFLOAD)
            resident[pool] = (owner, cache, prefix)
        self.layers[layer_name][1].copy_(torch.tensor(states, dtype=torch.int32, device=self.device))

    def lim_inputs(self, layer_name):
        if layer_name not in self.layers:
            raise RuntimeError("MTP graph layer inputs were not prepared before capture")
        return (*self.layers[layer_name], self.outputs)

    def copy_metadata(self):
        src, dst, route_misses, miss_src, miss_dst, misses = self.outputs
        # These copies execute inside the graph after LIM, on every replay.
        self.copy_sources[:, :LIM_MISS_CAPACITY].copy_(miss_src)
        self.copy_destinations[:, :LIM_MISS_CAPACITY].copy_(miss_dst)
        return src, dst, route_misses, self.copy_sources, self.copy_destinations, misses


def clone_graph_metadata(metadata):
    """Own the SFA inputs as well as the additional sparse metadata."""
    result = replace(metadata)
    for field in fields(metadata):
        value = getattr(metadata, field.name)
        if isinstance(value, torch.Tensor):
            setattr(result, field.name, value.clone())
    return result


def update_graph_metadata(destination, source, *, skip_fields=()):
    for field in fields(destination):
        if field.name in skip_fields:
            continue
        dst, src = getattr(destination, field.name), getattr(source, field.name)
        if isinstance(dst, torch.Tensor) and isinstance(src, torch.Tensor):
            if dst.shape != src.shape:
                if dst.ndim != src.ndim or dst.shape[1:] != src.shape[1:]:
                    raise ValueError(f"SFA graph metadata shape changed: {field.name}: {dst.shape} != {src.shape}")
                # FIA may append a different number of padding requests. The
                # sparse operator's real requests are checked separately.
                dst.fill_(-1 if field.name == "slot_mapping" else 0)
                rows = min(dst.shape[0], src.shape[0])
                dst[:rows].copy_(src[:rows])
            else:
                dst.copy_(src)


def make_capture_batch(metadata, manager, query_width):
    """Safe, long, uniform dummy input; used only during startup capture."""
    count = min(manager.max_num_reqs, max(1, metadata.num_input_tokens // query_width))
    device = metadata.seq_lens.device
    dummy = SimpleNamespace(
        num_prefills=0,
        num_decodes=count,
        cum_query_lens=torch.arange(1, count + 1, dtype=torch.int32, device=device) * query_width,
        seq_lens=torch.full((count,), manager.topk_buffer_size + query_width, dtype=torch.int32, device=device),
        req_topk_buffer_slots=torch.arange(count, dtype=torch.int32, device=device),
        block_table=torch.zeros((count, metadata.block_table.shape[1]), dtype=torch.int32, device=device),
    )
    return make_mtp_batch(dummy, manager)


class MtpGraphMetadataSet:
    """Stable SFA and offload inputs for all steps in one graph entry."""

    def __init__(self, runtime, metadata_steps):
        self.runtime = runtime
        self.replays = 0
        self.steps = []
        self.groups = []
        self.indexer_steps = []
        for step in metadata_steps:
            captured = {}
            seen = {}
            groups = []
            indexers = {}
            for layer_name, metadata in step.items():
                # A model can mix sparse SFA and ordinary attention layers in
                # one metadata dictionary. Only SFA offload metadata owns an
                # MTP batch; keep the other layers on their existing captured
                # metadata instead of rejecting the entire graph.
                if not hasattr(metadata, "mtp_batch"):
                    if isinstance(metadata, AscendSFAIndexerMetadata):
                        owned_indexer = clone_graph_metadata(metadata)
                        # Later replays can contain more requests than the
                        # first live batch. Keep all indexer table rows stable.
                        owned_indexer.block_table = metadata.block_table.new_zeros(
                            (runtime.manager.max_num_reqs, metadata.block_table.shape[1]),
                        )
                        rows = min(metadata.block_table.shape[0], runtime.manager.max_num_reqs)
                        owned_indexer.block_table[:rows].copy_(metadata.block_table[:rows])
                        indexers[layer_name] = owned_indexer
                        captured[layer_name] = indexers[layer_name]
                    else:
                        captured[layer_name] = metadata
                    continue
                if id(metadata) not in seen:
                    batch = metadata.mtp_batch
                    query_width = batch.num_tokens // len(batch.pool_rows)
                    buffers = MtpGraphBuffers(
                        runtime,
                        query_width=query_width,
                        source_capacity=batch.source_block_table.shape[1] * runtime.manager.block_size,
                        device=batch.seq_lens.device,
                        request_capacity=min(
                            runtime.manager.max_num_reqs,
                            metadata.num_input_tokens // query_width,
                        ),
                    )
                    owned = clone_graph_metadata(metadata)
                    owned.mtp_batch = buffers.batch
                    owned.cum_query_lens = buffers.batch.query_ends
                    owned.seq_lens = buffers.batch.seq_lens
                    owned.block_table = buffers.batch.source_block_table
                    owned.req_topk_buffer_slots = buffers.batch.pool_entries
                    owned.num_decodes = buffers.requests
                    owned.num_decode_tokens = buffers.tokens
                    owned.num_prefills = 0
                    seen[id(metadata)] = (owned, buffers, [])
                    groups.append(seen[id(metadata)])
                owned, _, names = seen[id(metadata)]
                names.append(layer_name)
                captured[layer_name] = owned
            self.steps.append(captured)
            self.groups.append(groups)
            self.indexer_steps.append(indexers)

    def accepts(self, metadata_steps):
        if len(metadata_steps) != len(self.steps):
            return False
        for groups, step in zip(self.groups, metadata_steps):
            for _, buffers, names in groups:
                if any(name not in step for name in names) or not buffers.accepts(step[names[0]].mtp_batch):
                    return False
        return True

    def update(self, metadata_steps):
        if not self.accepts(metadata_steps):
            raise ValueError("MTP graph metadata does not match its captured request/query capacity")
        for indexers, step in zip(self.indexer_steps, metadata_steps):
            for name, owned in indexers.items():
                update_graph_metadata(owned, step[name])
                if any(getattr(metadata, "mtp_graph_capture", False) for metadata in step.values()):
                    owned.slot_mapping.fill_(-1)
        for groups, step in zip(self.groups, metadata_steps):
            for owned, buffers, names in groups:
                incoming = step[names[0]]
                update_graph_metadata(
                    owned,
                    incoming,
                    skip_fields=("cum_query_lens", "seq_lens", "block_table", "req_topk_buffer_slots"),
                )
                buffers.update(incoming.mtp_batch, is_capture=incoming.mtp_graph_capture)
                # Slot -1 suppresses D2H for padded query rows; active_mask
                # independently suppresses their tail H2D descriptors.
                active_tokens = buffers.active_requests * buffers.query_width
                owned.slot_mapping[active_tokens:].fill_(-1)
                if owned.main_slot_mapping is not None:
                    owned.main_slot_mapping[active_tokens:].fill_(-1)
                for layer_name in names:
                    buffers.prepare_layer(layer_name)


def eligible_graph_steps(metadata_steps):
    """Reject eager fallback paths before selecting a captured sparse graph."""
    if not metadata_steps:
        return False
    seen = set()
    found_sparse_metadata = False
    for step in metadata_steps:
        if not isinstance(step, dict) or not step:
            return False
        for metadata in step.values():
            if not hasattr(metadata, "mtp_batch"):
                continue
            found_sparse_metadata = True
            if id(metadata) in seen:
                continue
            seen.add(id(metadata))
            batch = getattr(metadata, "mtp_batch", None)
            if batch is None:
                return False
            ends = batch.query_ends.cpu().tolist()
            widths = [end - start for start, end in zip([0] + ends[:-1], ends)]
            if not widths or len(set(widths)) != 1:
                return False
    return found_sparse_metadata
