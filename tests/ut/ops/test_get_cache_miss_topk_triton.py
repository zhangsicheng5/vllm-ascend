import random
import time

import torch
import torch_npu

import triton
import triton.language as tl

def get_cache_miss_topk_indices(
    req_ids_tensor: torch.Tensor,   # [num_reqs], int64, used to distinguish topk_idx from different requests
    topk_indices_old: torch.Tensor, # [num_reqs, topk(2048)], int64
    topk_indices_new: torch.Tensor, # [num_reqs, topk(2048)], int32
):
    """
    remove the cache hit (already in topk_indices_old) idx from topk_indices_new,
    only keep the cache miss part for following npu/cpu loading.
    for example,
    old: [1, 2, 3, 4],
    new: [1, 3, 5, 7],
    ret: [-1, 5, -1, 7],
    """
    def get_set_diff_mask(a: torch.tensor, b: torch.tensor) -> torch.Tensor:
        # only consider a.shape == b.shape == [bs, topk]
        assert a.shape == b.shape
        assert a.ndim == 2
        comparison_mask = a.unsqueeze(-1) == b.unsqueeze(1) # [bs, topk, topk]
        intersect_mask = comparison_mask.any(-1) # [bs, topk]
        return ~intersect_mask

    # to distinguish tokens of different reqs, add a req_ids_offset
    # maybe betther to use torch.bitwise_left_shift, but seems not supported on npu
    req_ids_offset = (req_ids_tensor * (1 << 16)).unsqueeze(-1)
    topk_indices_new = torch.where(topk_indices_new >= 0, topk_indices_new + req_ids_offset, -1)

    # tokens in new but not in old, which is cache miss and need to load
    cache_miss_token_mask = get_set_diff_mask(topk_indices_new, topk_indices_old)
    # tokens in old but not in new, which is useless now
    available_slot_mask = get_set_diff_mask(topk_indices_old, topk_indices_new)

    num_tokens_to_load = cache_miss_token_mask.sum(dim=1)
    num_available_slot = available_slot_mask.sum(dim=1)
    num_shortage_slot = num_tokens_to_load - num_available_slot
    # this part is needed while seq_len < 2k, num_shortage_slot > 0,
    # so there are multiple empty slots (idx == -1) in old topk_idx,
    # we also pick these empty slots to store cache miss tokens.
    num_shortage_slot = num_shortage_slot.unsqueeze(1)
    empty_slot_mask = topk_indices_old == -1
    empty_slot_cumsum = torch.cumsum(empty_slot_mask, dim=1)
    selected_empty_slot_mask = (empty_slot_cumsum <= num_shortage_slot) & empty_slot_mask
    available_slot_mask = torch.where(selected_empty_slot_mask, True, available_slot_mask)

    topk_indices_to_load_flattened = topk_indices_new[cache_miss_token_mask]
    topk_indices_new.fill_(-1)
    topk_indices_new[available_slot_mask] = topk_indices_to_load_flattened

    # update history topk_indices for next step usage
    topk_indices_old[...] = torch.where(available_slot_mask, topk_indices_new, topk_indices_old)

    # recover topk_indices (remove req offset)
    topk_indices_new = torch.where(topk_indices_new >= 0, topk_indices_new - req_ids_offset, -1)

    return topk_indices_new.to(torch.int32)


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


device='npu'

req_ids_tensor = torch.tensor([1, 2, 3, 5], dtype=torch.int64, device=device)
topk_indices_old = torch.tensor(
    [[4 + 1 * 65536, 2 + 1 * 65536, 5 + 1 * 65536, 6 + 1 * 65536],
     [4 + 2 * 65536, 2 + 2 * 65536, 1 + 2 * 65536, -1],
     [2 + 3 * 65536, 7 + 3 * 65536, -1, -1],
     [1 + 4 * 65536, 2 + 4 * 65536, 3 + 4 * 65536, 4 + 4 * 65536]],
    dtype=torch.int64,
    device=device,
)
topk_indices_new = torch.tensor(
    [[1, 2, 3, 4],
     [1, 2, 3, 4],
     [1, 2, 3, -1],
     [1, 2, 3, 4]],
    dtype=torch.int32,
    device=device,
)
gold = torch.tensor(
    [[-1, -1,  1,  3],
     [-1, -1, -1,  3],
     [-1,  1,  3, -1],
     [ 1,  2,  3,  4]],
    dtype=torch.int32,
    device=device,
)

# req_ids_tensor = torch.tensor([0], dtype=torch.int64, device=device)
# token_indices = random.sample(list(range(8 * 1024)), 2048); token_indices.sort()
# topk_indices_old = torch.tensor(token_indices, dtype=torch.int64, device=device).unsqueeze(0)
# token_indices = random.sample(list(range(8 * 1024)), 2048); token_indices.sort()
# topk_indices_new = torch.tensor(token_indices, dtype=torch.int32, device=device).unsqueeze(0)

print(f'>>>>> topk_idx old = {topk_indices_old}')
print(f'>>>>> topk_idx new = {topk_indices_new}')
repeat = 100

gold = get_cache_miss_topk_indices(req_ids_tensor, topk_indices_old.clone(), topk_indices_new.clone())
print(f'>>>>> gold = {gold}')
ret = get_cache_miss_topk_indices_triton(req_ids_tensor, topk_indices_old.clone(), topk_indices_new.clone())
print(f'>>>>> ret = {ret}')
print(f'>>>>> equal = {torch.equal(ret, gold)}')

time_0 = time.time()
for _ in range(repeat):
    get_cache_miss_topk_indices(req_ids_tensor, topk_indices_old.clone(), topk_indices_new.clone())
time_1 = time.time()

time_2 = time.time()
for _ in range(repeat):
    get_cache_miss_topk_indices_triton(req_ids_tensor, topk_indices_old.clone(), topk_indices_new.clone())
time_3 = time.time()

print(f'>>>>> repeat = {repeat}, time torch = {(time_1 - time_0) * 1000 / repeat} ms, time triton = {(time_3 - time_2) * 1000 / repeat} ms, time_all = {time_3 - time_0} s')
