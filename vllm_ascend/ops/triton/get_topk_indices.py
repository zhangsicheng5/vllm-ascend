import torch
import torch_npu

import triton
import triton.language as tl

TOKEN_LIMIT_PER_REQ = 65536
INVALID_TOKEN_ID = -1


class CacheMissTopKState:

    def __init__(
        self,
        max_num_reqs: int,
        topk: int,
        token_limit: int = TOKEN_LIMIT_PER_REQ,
        device: str = "npu",
    ):
        self.max_num_reqs = max_num_reqs
        self.topk = topk
        self.token_limit = token_limit
        self.device = device

        self.token_to_slot = torch.full(
            (max_num_reqs, token_limit), -1, dtype=torch.int32, device=device
        )
        self.slot_to_token = torch.full(
            (max_num_reqs, topk), -1, dtype=torch.int64, device=device
        )
        self.miss_tokens = torch.full(
            (max_num_reqs, topk), -1, dtype=torch.int32, device=device
        )
        self.miss_counts = torch.zeros(
            (max_num_reqs,), dtype=torch.int32, device=device
        )


@triton.jit
def _stateful_find_miss_kernel(
    req_ids_ptr,
    new_topk_ptr,
    token_to_slot_ptr,
    miss_tokens_ptr,
    miss_counts_ptr,
    num_reqs,
    topk: tl.constexpr,
    token_limit: tl.constexpr,
    BLOCK: tl.constexpr,
    SUB_BLOCK: tl.constexpr,
):
    """Kernel 1: lookup token_to_slot, compact miss tokens, write miss_counts."""
    pid = tl.program_id(0)
    if pid >= num_reqs:
        return

    req_id = tl.load(req_ids_ptr + pid)
    row_off = pid * topk
    cols = tl.arange(0, BLOCK)
    mask = cols < topk

    new_token = tl.load(new_topk_ptr + row_off + cols, mask=mask, other=-1).to(tl.int64)
    valid_new = (new_token >= 0) & (new_token < token_limit)
    safe_new = tl.where(valid_new, new_token, tl.zeros((BLOCK,), tl.int64))

    # token_to_slot[req_id, token] -> slot index or -1
    prev_slot = tl.load(
        token_to_slot_ptr + req_id * token_limit + safe_new,
        mask=valid_new,
        other=-1,
    )
    is_miss = (prev_slot < 0) & valid_new
    miss_rank = tl.cumsum(is_miss.to(tl.int64), axis=0) - 1
    num_miss = tl.sum(is_miss.to(tl.int64), axis=0)

    tl.store(miss_counts_ptr + pid, num_miss.to(tl.int32))

    # Scatter miss tokens into compact miss_tensors[row] by rank
    miss_row_off = pid * topk
    tl.store(miss_tokens_ptr + miss_row_off + cols, tl.full((BLOCK,), -1, tl.int32), mask=mask)

    for sb_start in range(0, BLOCK, SUB_BLOCK):
        target_ranks = sb_start + tl.arange(0, SUB_BLOCK)
        mr_b = tl.broadcast_to(miss_rank[:, None], (BLOCK, SUB_BLOCK))
        tr_b = tl.broadcast_to(target_ranks[None, :], (BLOCK, SUB_BLOCK))
        im_b = tl.broadcast_to(is_miss[:, None], (BLOCK, SUB_BLOCK))
        nt_b = tl.broadcast_to(new_token[:, None], (BLOCK, SUB_BLOCK))

        miss_match = (mr_b == tr_b) & im_b
        gathered = tl.sum(
            tl.where(miss_match, nt_b, tl.zeros((BLOCK, SUB_BLOCK), tl.int64)),
            axis=0,
        )
        write_mask = target_ranks < num_miss
        tl.store(
            miss_tokens_ptr + miss_row_off + target_ranks,
            gathered.to(tl.int32),
            mask=write_mask,
        )


@triton.jit
def _stateful_assign_slots_kernel(
    req_ids_ptr,
    new_topk_ptr,
    slot_to_token_ptr,
    token_to_slot_ptr,
    miss_tokens_ptr,
    miss_counts_ptr,
    out_ptr,
    num_reqs,
    topk: tl.constexpr,
    token_limit: tl.constexpr,
    BLOCK: tl.constexpr,
    SUB_BLOCK: tl.constexpr,
):
    """Kernel 2: scan slot_to_token, assign miss tokens to free slots, write output."""
    pid = tl.program_id(0)
    if pid >= num_reqs:
        return

    req_id = tl.load(req_ids_ptr + pid)
    row_off = pid * topk
    cols = tl.arange(0, BLOCK)
    mask = cols < topk

    new_token = tl.load(new_topk_ptr + row_off + cols, mask=mask, other=-1).to(tl.int64)
    old_token = tl.load(slot_to_token_ptr + req_id * topk + cols, mask=mask, other=-1).to(tl.int64)

    # Load miss info from kernel 1
    miss_count = tl.load(miss_counts_ptr + pid).to(tl.int64)
    miss_token = tl.load(miss_tokens_ptr + row_off + cols, mask=mask, other=-1).to(tl.int64)

    # Phase A: find free slots — old token not in new_topk
    avail_count = tl.zeros((BLOCK,), tl.int64)
    for sb_start in range(0, BLOCK, SUB_BLOCK):
        sb_cols = sb_start + tl.arange(0, SUB_BLOCK)
        sb_mask = sb_cols < topk
        new_chunk = tl.load(
            new_topk_ptr + row_off + sb_cols, mask=sb_mask, other=-1
        ).to(tl.int64)
        old_b = tl.broadcast_to(old_token[:, None], (BLOCK, SUB_BLOCK))
        new_b = tl.broadcast_to(new_chunk[None, :], (BLOCK, SUB_BLOCK))
        cmp = old_b == new_b
        avail_count += tl.sum(cmp.to(tl.int64), axis=1)

    stale_mask = (avail_count == 0) & (old_token >= 0)
    empty_mask = old_token == -1
    avail_mask = stale_mask | empty_mask

    # Handle shortage: fill empty slots when not enough stale slots
    num_avail = tl.sum(avail_mask.to(tl.int64), axis=0)
    num_shortage = miss_count - num_avail
    empty_cumsum = tl.cumsum(empty_mask.to(tl.int64), axis=0)
    selected_empty = (empty_cumsum <= num_shortage) & empty_mask
    avail_mask = avail_mask | selected_empty

    avail_rank = tl.cumsum(avail_mask.to(tl.int64), axis=0) - 1

    # Phase B: scatter miss tokens into free slots by rank
    out_vals = tl.full((BLOCK,), -1, tl.int64)
    for sb_start in range(0, BLOCK, SUB_BLOCK):
        target_ranks = sb_start + tl.arange(0, SUB_BLOCK)

        # Gather: find miss token with this rank
        mr_b = tl.broadcast_to(tl.arange(0, BLOCK)[:, None], (BLOCK, SUB_BLOCK))
        tr_b = tl.broadcast_to(target_ranks[None, :], (BLOCK, SUB_BLOCK))
        mt_b = tl.broadcast_to(miss_token[:, None], (BLOCK, SUB_BLOCK))

        # miss token at position `rank` (where miss_token was compacted)
        # The miss tokens are compacted at positions 0..miss_count-1
        pos_match = (mr_b == tr_b) & (mr_b < miss_count)
        gathered = tl.sum(
            tl.where(pos_match, mt_b, tl.zeros((BLOCK, SUB_BLOCK), tl.int64)),
            axis=0,
        )

        # Scatter: find free slot with this rank
        ar_b = tl.broadcast_to(avail_rank[:, None], (BLOCK, SUB_BLOCK))
        am_b = tl.broadcast_to(avail_mask[:, None], (BLOCK, SUB_BLOCK))
        valid_rank = tr_b < miss_count

        slot_match = (ar_b == tr_b) & am_b & valid_rank
        result = tl.sum(
            tl.where(
                slot_match,
                gathered[None, :],
                tl.zeros((BLOCK, SUB_BLOCK), tl.int64),
            ),
            axis=1,
        )
        has_match = tl.sum(slot_match.to(tl.int64), axis=1) > 0
        out_vals = tl.where(has_match, result, out_vals)

    # Phase C: update slot_to_token
    new_slot_token = tl.where(
        avail_mask,
        tl.where(out_vals >= 0, out_vals, tl.full((BLOCK,), -1, tl.int64)),
        old_token,
    )
    tl.store(
        slot_to_token_ptr + req_id * topk + cols, new_slot_token, mask=mask
    )

    # Phase D: update token_to_slot
    # Clear old stale token entries
    clear_mask = stale_mask & (old_token >= 0) & (old_token < token_limit)
    safe_old = tl.where(clear_mask, old_token, tl.zeros((BLOCK,), tl.int64))
    tl.store(
        token_to_slot_ptr + req_id * token_limit + safe_old,
        tl.full((BLOCK,), -1, tl.int32),
        mask=clear_mask,
    )
    # Set new token entries for assigned slots
    set_mask = (new_slot_token >= 0) & (new_slot_token < token_limit)
    safe_set = tl.where(set_mask, new_slot_token, tl.zeros((BLOCK,), tl.int64))
    tl.store(
        token_to_slot_ptr + req_id * token_limit + safe_set,
        cols.to(tl.int32),
        mask=set_mask,
    )

    # Phase E: write slot-aligned output
    out = tl.where(
        avail_mask & (out_vals >= 0),
        out_vals,
        tl.full((BLOCK,), -1, tl.int64),
    )
    tl.store(out_ptr + row_off + cols, out.to(tl.int32), mask=mask)


def get_cache_miss_topk_indices_triton_state(
    req_ids_tensor: torch.Tensor,
    state: CacheMissTopKState,
    topk_indices_new: torch.Tensor,
) -> torch.Tensor:
    num_reqs, topk = topk_indices_new.shape
    assert topk == state.topk, f"topk mismatch: {topk} vs state.topk {state.topk}"

    out = torch.empty_like(topk_indices_new, dtype=torch.int32)

    grid = (num_reqs,)
    BLOCK = triton.next_power_of_2(topk)

    # Kernel 1: find misses via token_to_slot lookup
    _stateful_find_miss_kernel[grid](
        req_ids_tensor,
        topk_indices_new,
        state.token_to_slot,
        state.miss_tokens,
        state.miss_counts,
        num_reqs,
        topk=topk,
        token_limit=state.token_limit,
        BLOCK=BLOCK,
        SUB_BLOCK=1,
    )

    # Kernel 2: assign misses to free slots, update state, write output
    _stateful_assign_slots_kernel[grid](
        req_ids_tensor,
        topk_indices_new,
        state.slot_to_token,
        state.token_to_slot,
        state.miss_tokens,
        state.miss_counts,
        out,
        num_reqs,
        topk=topk,
        token_limit=state.token_limit,
        BLOCK=BLOCK,
        SUB_BLOCK=1,
    )

    return out


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

    req_id = tl.load(req_ids_ptr + pid)
    req_offset = req_id * 65536
    row_off = pid * topk
    cols = tl.arange(0, BLOCK)
    mask = cols < topk

    old = tl.load(old_ptr + row_off + cols, mask=mask, other=-1).to(tl.int64)
    new = tl.load(new_ptr + row_off + cols, mask=mask, other=-1).to(tl.int64)
    new_with_offset = tl.where(new >= 0, new + req_offset, -1)
    # ---- sub-blocked miss_mask: new not in old ----
    miss_count = tl.zeros((BLOCK,), tl.int64)
    for sb_start in range(0, BLOCK, SUB_BLOCK):
        sb_cols = sb_start + tl.arange(0, SUB_BLOCK)
        sb_mask = sb_cols < topk
        old_chunk = tl.load(
            old_ptr + row_off + sb_cols, mask=sb_mask, other=-1
        ).to(tl.int64)
        old_b = tl.broadcast_to(old_chunk[None, :], (BLOCK, SUB_BLOCK))
        new_b = tl.broadcast_to(new_with_offset[:, None], (BLOCK, SUB_BLOCK))
        cmp = new_b == old_b
        miss_count += tl.sum(cmp.to(tl.int64), axis=1)
    miss_mask = (miss_count == 0) & (new_with_offset >= 0)

    # ---- sub-blocked avail_mask: old not in new ----
    avail_count = tl.zeros((BLOCK,), tl.int64)
    for sb_start in range(0, BLOCK, SUB_BLOCK):
        sb_cols = sb_start + tl.arange(0, SUB_BLOCK)
        sb_mask = sb_cols < topk
        new_chunk = tl.load(
            new_ptr + row_off + sb_cols, mask=sb_mask, other=-1
        ).to(tl.int64)
        new_chunk_off = tl.where(new_chunk >= 0, new_chunk + req_offset, -1)
        old_b = tl.broadcast_to(old[:, None], (BLOCK, SUB_BLOCK))
        new_b = tl.broadcast_to(new_chunk_off[None, :], (BLOCK, SUB_BLOCK))
        cmp = old_b == new_b
        avail_count += tl.sum(cmp.to(tl.int64), axis=1)
    avail_mask = (avail_count == 0) & (old >= 0)

    # ---- shortage: fill empty slots ----
    num_tokens_to_load = tl.sum(miss_mask.to(tl.int64), axis=0)
    num_available_slot = tl.sum(avail_mask.to(tl.int64), axis=0)
    num_shortage_slot = num_tokens_to_load - num_available_slot

    empty_mask = old == -1
    empty_cumsum = tl.cumsum(empty_mask.to(tl.int64), axis=0)
    selected_empty = (empty_cumsum <= num_shortage_slot) & empty_mask
    avail_mask = avail_mask | selected_empty

    # ---- compact: scatter miss values into available slots ----
    miss_vals = tl.where(miss_mask, new_with_offset, 0)
    avail_rank = tl.cumsum(avail_mask.to(tl.int64), axis=0) - 1
    miss_rank = tl.cumsum(miss_mask.to(tl.int64), axis=0) - 1
    num_miss = tl.sum(miss_mask.to(tl.int64), axis=0)

    # Gather-then-scatter: split by SUB_BLOCK chunks of target rank
    # Phase 1 (gather): for each target rank r in [sb_start, sb_start+SUB_BLOCK),
    #            find miss_vals where miss_rank == r
    # Phase 2 (scatter): for each available slot where avail_rank == r,
    #            write the gathered value
    out_with_offset = tl.full((BLOCK,), -1, tl.int64)
    for sb_start in range(0, BLOCK, SUB_BLOCK):
        target_ranks = sb_start + tl.arange(0, SUB_BLOCK)

        # Phase 1: gather - [BLOCK, SUB_BLOCK]
        mr_b = tl.broadcast_to(miss_rank[:, None], (BLOCK, SUB_BLOCK))
        tr_b = tl.broadcast_to(target_ranks[None, :], (BLOCK, SUB_BLOCK))
        mv_b = tl.broadcast_to(miss_vals[:, None], (BLOCK, SUB_BLOCK))
        mm_b = tl.broadcast_to(miss_mask[:, None], (BLOCK, SUB_BLOCK))

        miss_match = (mr_b == tr_b) & mm_b
        gathered = tl.sum(
            tl.where(miss_match, mv_b, tl.zeros((BLOCK, SUB_BLOCK), tl.int64)),
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
                tl.zeros((BLOCK, SUB_BLOCK), tl.int64),
            ),
            axis=1,
        )
        has_match = tl.sum(slot_match.to(tl.int64), axis=1) > 0
        out_with_offset = tl.where(has_match, result, out_with_offset)

    # ---- update old in-place ----
    updated_old = tl.where(avail_mask, out_with_offset, old)
    tl.store(old_ptr + row_off + cols, updated_old, mask=mask)

    # ---- remove req offset and store ----
    out = tl.where(out_with_offset >= 0, out_with_offset - req_offset, tl.full((BLOCK,), -1, tl.int64))
    tl.store(out_ptr + row_off + cols, out.to(tl.int32), mask=mask)


def get_cache_miss_topk_indices_triton(
    req_ids_tensor: torch.Tensor,
    topk_indices_old: torch.Tensor,
    topk_indices_new: torch.Tensor,
):
    num_reqs, topk = topk_indices_new.shape
    assert topk == topk_indices_old.shape[1]

    out = torch.empty_like(topk_indices_new, dtype=torch.int32)

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
