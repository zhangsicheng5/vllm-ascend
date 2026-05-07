import torch
import triton
import triton.language as tl


@triton.jit
def mark_cache_tokens_kernel(
        req_ids_ptr,
        old_ptr,
        new_ptr,
        old_marker_ptr,
        new_marker_ptr,
        num_reqs,
        stamp_ptr,
        topk: tl.constexpr,
        TOKEN_LIMIT: tl.constexpr,
        BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= num_reqs:
        return

    req_id = tl.load(req_ids_ptr + pid).to(tl.int64)
    req_offset = req_id * TOKEN_LIMIT
    row_off = pid * topk
    marker_off = pid * TOKEN_LIMIT
    cols = tl.arange(0, BLOCK)
    mask = cols < topk

    old_with_offset = tl.load(old_ptr + row_off + cols, mask=mask, other=-1).to(tl.int64)
    old_token = old_with_offset - req_offset
    old_valid = mask & (old_with_offset >= 0) & (old_token >= 0) & (old_token < TOKEN_LIMIT)
    old_token_safe = tl.where(old_valid, old_token, 0)
    stamp = tl.load(stamp_ptr).to(tl.int32)
    stamp_vals = tl.full((BLOCK,), 0, tl.int32) + stamp
    tl.store(old_marker_ptr + marker_off + old_token_safe, stamp_vals, mask=old_valid)

    new_token = tl.load(new_ptr + row_off + cols, mask=mask, other=-1).to(tl.int64)
    new_valid = mask & (new_token >= 0) & (new_token < TOKEN_LIMIT)
    new_token_safe = tl.where(new_valid, new_token, 0)
    tl.store(new_marker_ptr + marker_off + new_token_safe, stamp_vals, mask=new_valid)


@triton.jit
def compact_cache_miss_slots_kernel(
        req_ids_ptr,
        old_ptr,
        new_ptr,
        old_marker_ptr,
        new_marker_ptr,
        slot_scratch_ptr,
        miss_scratch_ptr,
        miss_count_ptr,
        slot_count_ptr,
        num_reqs,
        stamp_ptr,
        topk: tl.constexpr,
        TOKEN_LIMIT: tl.constexpr,
        BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= num_reqs:
        return

    req_id = tl.load(req_ids_ptr + pid).to(tl.int64)
    req_offset = req_id * TOKEN_LIMIT
    row_off = pid * topk
    marker_off = pid * TOKEN_LIMIT
    cols = tl.arange(0, BLOCK)
    mask = cols < topk

    old_with_offset = tl.load(old_ptr + row_off + cols, mask=mask, other=-1).to(tl.int64)
    old_token = old_with_offset - req_offset
    old_valid = mask & (old_with_offset >= 0) & (old_token >= 0) & (old_token < TOKEN_LIMIT)
    old_token_safe = tl.where(old_valid, old_token, 0)

    new_token = tl.load(new_ptr + row_off + cols, mask=mask, other=-1).to(tl.int64)
    new_valid = mask & (new_token >= 0) & (new_token < TOKEN_LIMIT)
    new_token_safe = tl.where(new_valid, new_token, 0)
    new_with_offset = new_token + req_offset

    old_hit = tl.load(old_marker_ptr + marker_off + new_token_safe, mask=new_valid, other=0)
    new_hit = tl.load(new_marker_ptr + marker_off + old_token_safe, mask=old_valid, other=0)

    stamp = tl.load(stamp_ptr).to(tl.int32)
    miss_mask = new_valid & (old_hit != stamp)
    avail_mask = old_valid & (new_hit != stamp)

    num_miss = tl.sum(miss_mask.to(tl.int32), axis=0)
    num_avail = tl.sum(avail_mask.to(tl.int32), axis=0)
    num_shortage = num_miss - num_avail

    empty_mask = mask & (old_with_offset == -1)
    empty_cumsum = tl.cumsum(empty_mask.to(tl.int32), axis=0)
    selected_empty = (empty_cumsum <= num_shortage) & empty_mask
    avail_mask = avail_mask | selected_empty

    miss_rank = tl.cumsum(miss_mask.to(tl.int32), axis=0) - 1
    avail_rank = tl.cumsum(avail_mask.to(tl.int32), axis=0) - 1
    num_slots = tl.sum(avail_mask.to(tl.int32), axis=0)
    miss_rank_safe = tl.where(miss_mask, miss_rank, 0)
    avail_rank_safe = tl.where(avail_mask, avail_rank, 0)

    tl.store(slot_scratch_ptr + row_off + avail_rank_safe, cols, mask=avail_mask)
    tl.store(miss_scratch_ptr + row_off + miss_rank_safe, new_with_offset, mask=miss_mask)
    tl.store(miss_count_ptr + pid, num_miss)
    tl.store(slot_count_ptr + pid, num_slots)


@triton.jit
def apply_cache_miss_slots_kernel(
        req_ids_ptr,
        old_ptr,
        out_ptr,
        slot_scratch_ptr,
        miss_scratch_ptr,
        miss_count_ptr,
        slot_count_ptr,
        num_reqs,
        topk: tl.constexpr,
        TOKEN_LIMIT: tl.constexpr,
        BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= num_reqs:
        return

    req_id = tl.load(req_ids_ptr + pid).to(tl.int64)
    req_offset = req_id * TOKEN_LIMIT
    row_off = pid * topk
    cols = tl.arange(0, BLOCK)
    mask = cols < topk

    tl.store(out_ptr + row_off + cols, tl.full((BLOCK,), -1, tl.int32), mask=mask)

    num_miss = tl.load(miss_count_ptr + pid).to(tl.int32)
    num_slots = tl.load(slot_count_ptr + pid).to(tl.int32)
    update_mask = cols < num_slots
    miss_mask = cols < num_miss
    slots = tl.load(slot_scratch_ptr + row_off + cols, mask=update_mask, other=0).to(tl.int32)
    miss_with_offset = tl.load(miss_scratch_ptr + row_off + cols, mask=miss_mask, other=-1).to(tl.int64)
    miss_token = miss_with_offset - req_offset
    out_mask = miss_mask & update_mask

    tl.store(old_ptr + row_off + slots, miss_with_offset, mask=update_mask)
    tl.store(out_ptr + row_off + slots, miss_token.to(tl.int32), mask=out_mask)


@triton.jit
def get_cache_miss_topk_kernel(
        req_ids_ptr,
        old_ptr,
        new_ptr,
        out_ptr,
        num_reqs,
        topk: tl.constexpr,
        BLOCK: tl.constexpr,
        SUB_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= num_reqs:
        return

    req_id = tl.load(req_ids_ptr + pid).to(tl.int32)
    req_offset = req_id * 65536
    row_off = pid * topk
    cols = tl.arange(0, BLOCK)
    mask = cols < topk

    old = tl.load(old_ptr + row_off + cols, mask=mask, other=-1).to(tl.int32)
    new = tl.load(new_ptr + row_off + cols, mask=mask, other=-1).to(tl.int32)
    new_with_offset = tl.where(new >= 0, new + req_offset, -1)
    # ---- sub-blocked miss_mask: new not in old ----
    miss_count = tl.zeros((BLOCK,), tl.int32)
    for sb_start in range(0, BLOCK, SUB_BLOCK):
        sb_cols = sb_start + tl.arange(0, SUB_BLOCK)
        sb_mask = sb_cols < topk
        old_chunk = tl.load(
            old_ptr + row_off + sb_cols, mask=sb_mask, other=-1
        ).to(tl.int32)
        old_b = tl.broadcast_to(old_chunk[None, :], (BLOCK, SUB_BLOCK))
        new_b = tl.broadcast_to(new_with_offset[:, None], (BLOCK, SUB_BLOCK))
        cmp = new_b == old_b
        miss_count += tl.sum(cmp.to(tl.int32), axis=1)
    miss_mask = (miss_count == 0) & (new_with_offset >= 0)

    # ---- sub-blocked avail_mask: old not in new ----
    avail_count = tl.zeros((BLOCK,), tl.int32)
    for sb_start in range(0, BLOCK, SUB_BLOCK):
        sb_cols = sb_start + tl.arange(0, SUB_BLOCK)
        sb_mask = sb_cols < topk
        new_chunk = tl.load(
            new_ptr + row_off + sb_cols, mask=sb_mask, other=-1
        ).to(tl.int32)
        new_chunk_off = tl.where(new_chunk >= 0, new_chunk + req_offset, -1)
        old_b = tl.broadcast_to(old[:, None], (BLOCK, SUB_BLOCK))
        new_b = tl.broadcast_to(new_chunk_off[None, :], (BLOCK, SUB_BLOCK))
        cmp = old_b == new_b
        avail_count += tl.sum(cmp.to(tl.int32), axis=1)
    avail_mask = (avail_count == 0) & (old >= 0)

    # ---- shortage: fill empty slots ----
    num_tokens_to_load = tl.sum(miss_mask.to(tl.int32), axis=0)
    num_available_slot = tl.sum(avail_mask.to(tl.int32), axis=0)
    num_shortage_slot = num_tokens_to_load - num_available_slot

    empty_mask = old == -1
    empty_cumsum = tl.cumsum(empty_mask.to(tl.int32), axis=0)
    selected_empty = (empty_cumsum <= num_shortage_slot) & empty_mask
    avail_mask = avail_mask | selected_empty

    # ---- compact: scatter miss values into available slots ----
    miss_vals = tl.where(miss_mask, new_with_offset, 0)
    avail_rank = tl.cumsum(avail_mask.to(tl.int32), axis=0) - 1
    miss_rank = tl.cumsum(miss_mask.to(tl.int32), axis=0) - 1
    num_miss = tl.sum(miss_mask.to(tl.int32), axis=0)

    # Gather-then-scatter: split by SUB_BLOCK chunks of target rank
    # Phase 1 (gather): for each target rank r in [sb_start, sb_start+SUB_BLOCK),
    #            find miss_vals where miss_rank == r
    # Phase 2 (scatter): for each available slot where avail_rank == r,
    #            write the gathered value
    out_with_offset = tl.full((BLOCK,), -1, tl.int32)
    for sb_start in range(0, BLOCK, SUB_BLOCK):
        target_ranks = sb_start + tl.arange(0, SUB_BLOCK)

        # Phase 1: gather - [BLOCK, SUB_BLOCK]
        mr_b = tl.broadcast_to(miss_rank[:, None], (BLOCK, SUB_BLOCK))
        tr_b = tl.broadcast_to(target_ranks[None, :], (BLOCK, SUB_BLOCK))
        mv_b = tl.broadcast_to(miss_vals[:, None], (BLOCK, SUB_BLOCK))
        mm_b = tl.broadcast_to(miss_mask[:, None], (BLOCK, SUB_BLOCK))

        miss_match = (mr_b == tr_b) & mm_b
        gathered = tl.sum(
            tl.where(miss_match, mv_b, tl.zeros((BLOCK, SUB_BLOCK), tl.int32)),
            axis=0,
        )

        # Phase 2: scatter - [BLOCK, SUB_BLOCK]
        ar_b = tl.broadcast_to(avail_rank[:, None], (BLOCK, SUB_BLOCK))
        am_b = tl.broadcast_to(avail_mask[:, None], (BLOCK, SUB_BLOCK))
        valid_rank = tr_b < num_miss

        slot_match = (ar_b == tr_b) & am_b & valid_rank
        result = tl.sum(
            tl.where(
                slot_match,
                gathered[None, :],
                tl.zeros((BLOCK, SUB_BLOCK), tl.int32),
            ),
            axis=1,
        )
        has_match = tl.sum(slot_match.to(tl.int32), axis=1) > 0
        out_with_offset = tl.where(has_match, result, out_with_offset)

    # ---- update old in-place ----
    updated_old = tl.where(avail_mask, out_with_offset, old)
    tl.store(old_ptr + row_off + cols, updated_old, mask=mask)

    # ---- remove req offset and store ----
    out = tl.where(out_with_offset >= 0, out_with_offset - req_offset, tl.full((BLOCK,), -1, tl.int32))
    tl.store(out_ptr + row_off + cols, out.to(tl.int32), mask=mask)


def get_cache_miss_topk_indices_triton_bitmap(
    req_ids_tensor: torch.Tensor,
    topk_indices_old: torch.Tensor,
    topk_indices_new: torch.Tensor,
    token_limit: int = 65536,
    old_marker: torch.Tensor | None = None,
    new_marker: torch.Tensor | None = None,
    slot_scratch: torch.Tensor | None = None,
    miss_scratch: torch.Tensor | None = None,
    miss_count: torch.Tensor | None = None,
    slot_count: torch.Tensor | None = None,
    stamp_tensor: torch.Tensor | None = None,
):
    num_reqs, topk = topk_indices_new.shape
    assert topk == topk_indices_old.shape[1]

    out = torch.empty(
        topk_indices_new.shape,
        dtype=torch.int32,
        device=topk_indices_new.device,
    )
    if old_marker is None:
        old_marker = torch.zeros(
            (num_reqs, token_limit),
            dtype=torch.int32,
            device=topk_indices_new.device,
        )
    if new_marker is None:
        new_marker = torch.zeros(
            (num_reqs, token_limit),
            dtype=torch.int32,
            device=topk_indices_new.device,
        )
    if slot_scratch is None:
        slot_scratch = torch.empty(
            topk_indices_new.shape,
            dtype=torch.int32,
            device=topk_indices_new.device,
        )
    if miss_scratch is None:
        miss_scratch = torch.empty(
            topk_indices_old.shape,
            dtype=topk_indices_old.dtype,
            device=topk_indices_old.device,
        )
    if miss_count is None:
        miss_count = torch.empty((num_reqs,), dtype=torch.int32, device=topk_indices_new.device)
    if slot_count is None:
        slot_count = torch.empty((num_reqs,), dtype=torch.int32, device=topk_indices_new.device)
    if stamp_tensor is None:
        stamp_tensor = torch.ones((1,), dtype=torch.int32, device=topk_indices_new.device)

    grid = (num_reqs,)
    BLOCK = triton.next_power_of_2(topk)

    mark_cache_tokens_kernel[grid](
        req_ids_tensor,
        topk_indices_old,
        topk_indices_new,
        old_marker,
        new_marker,
        num_reqs,
        stamp_tensor,
        topk=topk,
        TOKEN_LIMIT=token_limit,
        BLOCK=BLOCK,
    )
    torch.npu.synchronize()

    compact_cache_miss_slots_kernel[grid](
        req_ids_tensor,
        topk_indices_old,
        topk_indices_new,
        old_marker,
        new_marker,
        slot_scratch,
        miss_scratch,
        miss_count,
        slot_count,
        num_reqs,
        stamp_tensor,
        topk=topk,
        TOKEN_LIMIT=token_limit,
        BLOCK=BLOCK,
    )
    torch.npu.synchronize()

    apply_cache_miss_slots_kernel[grid](
        req_ids_tensor,
        topk_indices_old,
        out,
        slot_scratch,
        miss_scratch,
        miss_count,
        slot_count,
        num_reqs,
        topk=topk,
        TOKEN_LIMIT=token_limit,
        BLOCK=BLOCK,
    )
    torch.npu.synchronize()

    return out


def get_cache_miss_topk_indices_triton_exact(
    req_ids_tensor: torch.Tensor,
    topk_indices_old: torch.Tensor,
    topk_indices_new: torch.Tensor,
):
    num_reqs, topk = topk_indices_new.shape
    assert topk == topk_indices_old.shape[1]

    out = torch.empty(
        topk_indices_new.shape,
        dtype=torch.int32,
        device=topk_indices_new.device,
    )

    grid = (num_reqs,)
    BLOCK = triton.next_power_of_2(topk)

    get_cache_miss_topk_kernel[grid](
        req_ids_tensor,
        topk_indices_old,
        topk_indices_new,
        out,
        num_reqs,
        topk=topk,
        BLOCK=BLOCK,
        SUB_BLOCK=1
    )
    return out


def get_cache_miss_topk_indices_triton(
    req_ids_tensor: torch.Tensor,
    topk_indices_old: torch.Tensor,
    topk_indices_new: torch.Tensor,
    **kwargs,
):
    return get_cache_miss_topk_indices_triton_bitmap(
        req_ids_tensor,
        topk_indices_old,
        topk_indices_new,
        **kwargs,
    )


class CacheMissTopKScratch:

    def __init__(self):
        self.token_limit = None
        self.device = None
        self.history_dtype = None
        self.marker_shape = (0, 0)
        self.scratch_shape = (0, 0)
        self.stamp = 0
        self.old_marker = None
        self.new_marker = None
        self.slot_scratch = None
        self.miss_scratch = None
        self.miss_count = None
        self.slot_count = None
        self.stamp_tensor = None

    def prepare(
        self,
        num_reqs: int,
        topk: int,
        device,
        token_limit: int = 65536,
        history_dtype: torch.dtype = torch.int64,
    ):
        marker_shape = (num_reqs, token_limit)
        scratch_shape = (num_reqs, topk)
        needs_alloc = (
            self.token_limit != token_limit
            or self.device != device
            or self.history_dtype != history_dtype
            or self.marker_shape[0] < num_reqs
            or self.scratch_shape[0] < num_reqs
            or self.scratch_shape[1] != topk
        )

        if needs_alloc:
            self.old_marker = torch.zeros(marker_shape, dtype=torch.int32, device=device)
            self.new_marker = torch.zeros(marker_shape, dtype=torch.int32, device=device)
            self.slot_scratch = torch.empty(scratch_shape, dtype=torch.int32, device=device)
            self.miss_scratch = torch.empty(scratch_shape, dtype=history_dtype, device=device)
            self.miss_count = torch.empty((num_reqs,), dtype=torch.int32, device=device)
            self.slot_count = torch.empty((num_reqs,), dtype=torch.int32, device=device)
            self.stamp_tensor = torch.empty((1,), dtype=torch.int32, device=device)
            self.token_limit = token_limit
            self.device = device
            self.history_dtype = history_dtype
            self.marker_shape = marker_shape
            self.scratch_shape = scratch_shape
            self.stamp = 0

        self.stamp += 1
        if self.stamp >= 2_000_000_000:
            self.old_marker.zero_()
            self.new_marker.zero_()
            self.stamp = 1
        self.stamp_tensor.fill_(self.stamp)

        return {
            "token_limit": token_limit,
            "stamp_tensor": self.stamp_tensor,
            "old_marker": self.old_marker[:num_reqs],
            "new_marker": self.new_marker[:num_reqs],
            "slot_scratch": self.slot_scratch[:num_reqs, :topk],
            "miss_scratch": self.miss_scratch[:num_reqs, :topk],
            "miss_count": self.miss_count[:num_reqs],
            "slot_count": self.slot_count[:num_reqs],
        }
