import torch
import triton
import triton.language as tl


@triton.jit
def compact_cache_miss_state_kernel(
        req_ids_ptr,
        new_ptr,
        req_state_ptr,
        token_to_slot_ptr,
        slot_to_token_ptr,
        slot_stamp_ptr,
        free_scratch_ptr,
        miss_scratch_ptr,
        free_count_ptr,
        miss_count_ptr,
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
    last_req_id = tl.load(req_state_ptr + pid).to(tl.int64)
    req_changed = req_id != last_req_id

    row_off = pid * topk
    token_map_off = pid * TOKEN_LIMIT
    cols = tl.arange(0, BLOCK)
    mask = cols < topk

    old_slot_token = tl.load(slot_to_token_ptr + row_off + cols, mask=mask, other=-1).to(tl.int64)
    old_slot_valid = mask & (old_slot_token >= 0) & (old_slot_token < TOKEN_LIMIT)
    old_slot_token_safe = tl.where(old_slot_valid, old_slot_token, 0)
    tl.store(token_to_slot_ptr + token_map_off + old_slot_token_safe, -1, mask=req_changed & old_slot_valid)
    tl.store(slot_to_token_ptr + row_off + cols, -1, mask=req_changed & mask)
    tl.store(slot_stamp_ptr + row_off + cols, 0, mask=req_changed & mask)
    tl.store(req_state_ptr + pid, req_id, mask=req_changed)

    current_slot_token = tl.where(req_changed, -1, old_slot_token)
    new_token = tl.load(new_ptr + row_off + cols, mask=mask, other=-1).to(tl.int64)
    new_valid = mask & (new_token >= 0) & (new_token < TOKEN_LIMIT)
    new_token_safe = tl.where(new_valid, new_token, 0)

    slot = tl.load(token_to_slot_ptr + token_map_off + new_token_safe, mask=new_valid, other=-1).to(tl.int32)
    slot_valid = new_valid & (slot >= 0) & (slot < topk)
    slot_safe = tl.where(slot_valid, slot, 0)
    slot_token = tl.load(slot_to_token_ptr + row_off + slot_safe, mask=slot_valid, other=-1).to(tl.int64)
    hit_mask = slot_valid & (slot_token == new_token)

    stamp = tl.load(stamp_ptr).to(tl.int32)
    tl.store(slot_stamp_ptr + row_off + slot_safe, stamp, mask=hit_mask)

    miss_mask = new_valid & ~hit_mask
    slot_stamp = tl.load(slot_stamp_ptr + row_off + cols, mask=mask, other=0).to(tl.int32)
    empty_slot_mask = mask & (current_slot_token < 0)
    stale_slot_mask = mask & (current_slot_token >= 0) & (slot_stamp != stamp)
    free_mask = empty_slot_mask | stale_slot_mask

    miss_rank = tl.cumsum(miss_mask.to(tl.int32), axis=0) - 1
    free_rank = tl.cumsum(free_mask.to(tl.int32), axis=0) - 1
    miss_rank_safe = tl.where(miss_mask, miss_rank, 0)
    free_rank_safe = tl.where(free_mask, free_rank, 0)
    num_miss = tl.sum(miss_mask.to(tl.int32), axis=0)
    num_free = tl.sum(free_mask.to(tl.int32), axis=0)

    tl.store(miss_scratch_ptr + row_off + miss_rank_safe, new_token, mask=miss_mask)
    tl.store(free_scratch_ptr + row_off + free_rank_safe, cols, mask=free_mask)
    tl.store(miss_count_ptr + pid, num_miss)
    tl.store(free_count_ptr + pid, num_free)


@triton.jit
def apply_cache_miss_state_kernel(
        req_ids_ptr,
        out_ptr,
        token_to_slot_ptr,
        slot_to_token_ptr,
        slot_stamp_ptr,
        free_scratch_ptr,
        miss_scratch_ptr,
        free_count_ptr,
        miss_count_ptr,
        num_reqs,
        stamp_ptr,
        topk: tl.constexpr,
        TOKEN_LIMIT: tl.constexpr,
        BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= num_reqs:
        return

    row_off = pid * topk
    token_map_off = pid * TOKEN_LIMIT
    cols = tl.arange(0, BLOCK)
    mask = cols < topk

    tl.store(out_ptr + row_off + cols, tl.full((BLOCK,), -1, tl.int32), mask=mask)

    num_miss = tl.load(miss_count_ptr + pid).to(tl.int32)
    num_free = tl.load(free_count_ptr + pid).to(tl.int32)
    update_mask = mask & (cols < num_free)
    miss_mask = mask & (cols < num_miss)

    free_slot = tl.load(free_scratch_ptr + row_off + cols, mask=update_mask, other=0).to(tl.int32)
    old_token = tl.load(slot_to_token_ptr + row_off + free_slot, mask=update_mask, other=-1).to(tl.int64)
    old_valid = update_mask & (old_token >= 0) & (old_token < TOKEN_LIMIT)
    old_token_safe = tl.where(old_valid, old_token, 0)
    tl.store(token_to_slot_ptr + token_map_off + old_token_safe, -1, mask=old_valid)

    new_token = tl.load(miss_scratch_ptr + row_off + cols, mask=miss_mask, other=-1).to(tl.int64)
    new_valid = miss_mask & (new_token >= 0) & (new_token < TOKEN_LIMIT)
    new_token_safe = tl.where(new_valid, new_token, 0)
    write_mask = update_mask & new_valid
    stamp = tl.load(stamp_ptr).to(tl.int32)

    tl.store(slot_to_token_ptr + row_off + free_slot, new_token, mask=write_mask)
    tl.store(slot_stamp_ptr + row_off + free_slot, stamp, mask=write_mask)
    tl.store(token_to_slot_ptr + token_map_off + new_token_safe, free_slot, mask=write_mask)
    tl.store(out_ptr + row_off + free_slot, new_token.to(tl.int32), mask=write_mask)


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
    out: torch.Tensor | None = None,
):
    num_reqs, topk = topk_indices_new.shape
    assert topk == topk_indices_old.shape[1]

    if out is None:
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

    return out


def get_cache_miss_topk_indices_triton_exact(
    req_ids_tensor: torch.Tensor,
    topk_indices_old: torch.Tensor,
    topk_indices_new: torch.Tensor,
    sub_block: int = 4,
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
        SUB_BLOCK=sub_block,
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


def get_cache_miss_topk_indices_triton_state(
    req_ids_tensor: torch.Tensor,
    topk_indices_new: torch.Tensor,
    token_limit: int = 65536,
    req_state: torch.Tensor | None = None,
    token_to_slot: torch.Tensor | None = None,
    slot_to_token: torch.Tensor | None = None,
    slot_stamp: torch.Tensor | None = None,
    free_scratch: torch.Tensor | None = None,
    miss_scratch: torch.Tensor | None = None,
    free_count: torch.Tensor | None = None,
    miss_count: torch.Tensor | None = None,
    stamp_tensor: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
):
    num_reqs, topk = topk_indices_new.shape
    device = topk_indices_new.device

    if req_state is None:
        req_state = torch.full((num_reqs,), -1, dtype=req_ids_tensor.dtype, device=device)
    if token_to_slot is None:
        token_to_slot = torch.full((num_reqs, token_limit), -1, dtype=torch.int32, device=device)
    if slot_to_token is None:
        slot_to_token = torch.full((num_reqs, topk), -1, dtype=torch.int64, device=device)
    if slot_stamp is None:
        slot_stamp = torch.zeros((num_reqs, topk), dtype=torch.int32, device=device)
    if free_scratch is None:
        free_scratch = torch.empty((num_reqs, topk), dtype=torch.int32, device=device)
    if miss_scratch is None:
        miss_scratch = torch.empty((num_reqs, topk), dtype=torch.int64, device=device)
    if free_count is None:
        free_count = torch.empty((num_reqs,), dtype=torch.int32, device=device)
    if miss_count is None:
        miss_count = torch.empty((num_reqs,), dtype=torch.int32, device=device)
    if stamp_tensor is None:
        stamp_tensor = torch.ones((1,), dtype=torch.int32, device=device)
    if out is None:
        out = torch.empty((num_reqs, topk), dtype=torch.int32, device=device)

    grid = (num_reqs,)
    BLOCK = triton.next_power_of_2(topk)

    compact_cache_miss_state_kernel[grid](
        req_ids_tensor,
        topk_indices_new,
        req_state,
        token_to_slot,
        slot_to_token,
        slot_stamp,
        free_scratch,
        miss_scratch,
        free_count,
        miss_count,
        num_reqs,
        stamp_tensor,
        topk=topk,
        TOKEN_LIMIT=token_limit,
        BLOCK=BLOCK,
    )

    apply_cache_miss_state_kernel[grid](
        req_ids_tensor,
        out,
        token_to_slot,
        slot_to_token,
        slot_stamp,
        free_scratch,
        miss_scratch,
        free_count,
        miss_count,
        num_reqs,
        stamp_tensor,
        topk=topk,
        TOKEN_LIMIT=token_limit,
        BLOCK=BLOCK,
    )

    return out


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
        self.out = None

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
            self.out = torch.empty(scratch_shape, dtype=torch.int32, device=device)
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
            "out": self.out[:num_reqs, :topk],
        }


class CacheMissTopKState:

    def __init__(self):
        self.token_limit = None
        self.device = None
        self.req_dtype = None
        self.shape = (0, 0)
        self.stamp = 0
        self.req_state = None
        self.token_to_slot = None
        self.slot_to_token = None
        self.slot_stamp = None
        self.free_scratch = None
        self.miss_scratch = None
        self.free_count = None
        self.miss_count = None
        self.stamp_tensor = None
        self.out = None

    def prepare(
        self,
        num_reqs: int,
        topk: int,
        device,
        req_dtype: torch.dtype,
        token_limit: int = 65536,
    ):
        needs_alloc = (
            self.token_limit != token_limit
            or self.device != device
            or self.req_dtype != req_dtype
            or self.shape[0] < num_reqs
            or self.shape[1] != topk
        )

        if needs_alloc:
            self.req_state = torch.full((num_reqs,), -1, dtype=req_dtype, device=device)
            self.token_to_slot = torch.full((num_reqs, token_limit), -1, dtype=torch.int32, device=device)
            self.slot_to_token = torch.full((num_reqs, topk), -1, dtype=torch.int64, device=device)
            self.slot_stamp = torch.zeros((num_reqs, topk), dtype=torch.int32, device=device)
            self.free_scratch = torch.empty((num_reqs, topk), dtype=torch.int32, device=device)
            self.miss_scratch = torch.empty((num_reqs, topk), dtype=torch.int64, device=device)
            self.free_count = torch.empty((num_reqs,), dtype=torch.int32, device=device)
            self.miss_count = torch.empty((num_reqs,), dtype=torch.int32, device=device)
            self.stamp_tensor = torch.empty((1,), dtype=torch.int32, device=device)
            self.out = torch.empty((num_reqs, topk), dtype=torch.int32, device=device)
            self.token_limit = token_limit
            self.device = device
            self.req_dtype = req_dtype
            self.shape = (num_reqs, topk)
            self.stamp = 0

        self.stamp += 1
        if self.stamp >= 2_000_000_000:
            self.slot_stamp.zero_()
            self.stamp = 1
        self.stamp_tensor.fill_(self.stamp)

        return {
            "token_limit": token_limit,
            "req_state": self.req_state[:num_reqs],
            "token_to_slot": self.token_to_slot[:num_reqs],
            "slot_to_token": self.slot_to_token[:num_reqs, :topk],
            "slot_stamp": self.slot_stamp[:num_reqs, :topk],
            "free_scratch": self.free_scratch[:num_reqs, :topk],
            "miss_scratch": self.miss_scratch[:num_reqs, :topk],
            "free_count": self.free_count[:num_reqs],
            "miss_count": self.miss_count[:num_reqs],
            "stamp_tensor": self.stamp_tensor,
            "out": self.out[:num_reqs, :topk],
        }
