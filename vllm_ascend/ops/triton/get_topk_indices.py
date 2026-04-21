import torch
import torch_npu

import triton
import triton.language as tl

@triton.jit
def get_cache_miss_topk_kernel(                                                                                                                                                                                    
    req_ids_ptr,
    old_ptr,                                                                                                                                                                                                       
    new_ptr,                                                                                                                                                                                                       
    out_ptr,
    num_reqs,                                                                                                                                                                                                      
    topk: tl.constexpr,
    BLOCK: tl.constexpr,                                                                                                                                                                                           
):                                                                                                                                                                                                                 
    pid = tl.program_id(0)                                                                                                                                                                                         
    if pid >= num_reqs:                                                                                                                                                                                            
        return                                                                                                                                                                                                     

    req_id = tl.load(req_ids_ptr + pid)                                                                                                                                                                            
    req_offset = req_id * 65536
                                                                                                                                                                                                                    
    row_off = pid * topk                                                                                                                                                                                           
    cols = tl.arange(0, BLOCK)                                                                                                                                                                                     
    mask = cols < topk                                                                                                                                                                                             
                
    old = tl.load(old_ptr + row_off + cols, mask=mask, other=-1).to(tl.int32)                                                                                                                                      
    new = tl.load(new_ptr + row_off + cols, mask=mask, other=-1).to(tl.int32)
                                                                                                                                                                                                                    
    # add req offset to new, ignore -1                                                                                                                                                                             
    new_with_offset = tl.where(new >= 0, new + req_offset, -1)
                                                                                                                                                                                                                    
    # ---- cache miss mask: new not in old ----                                                                                                                                                                    
    # new_b[i, j] = new_with_offset[i], old_b[i, j] = old[j]                                                                                                                                                       
    new_b = tl.broadcast_to(new_with_offset[:, None], (BLOCK, BLOCK))                                                                                                                                              
    old_b = tl.broadcast_to(old[None, :],             (BLOCK, BLOCK))                                                                                                                                              
    cmp_new_old = (new_b == old_b)                          # [BLOCK, BLOCK]                                                                                                                                       
                                                                                                                                                                                                                    
    # replace tl.any(cmp_new_old, axis=1) → sum > 0                                                                                                                                                                
    has_match_in_old = tl.sum(cmp_new_old.to(tl.int32), axis=1) > 0  # [BLOCK]                                                                                                                                     
    miss_mask = (~has_match_in_old) & (new_with_offset >= 0)          # [BLOCK]                                                                                                                                    
                                                                                                                                                                                                                    
    # ---- available slot mask: old not in new ----                                                                                                                                                                
    old_b2 = tl.broadcast_to(old[:, None],             (BLOCK, BLOCK))                                                                                                                                             
    new_b2 = tl.broadcast_to(new_with_offset[None, :], (BLOCK, BLOCK))                                                                                                                                             
    cmp_old_new = (old_b2 == new_b2)                        # [BLOCK, BLOCK]                                                                                                                                       
                                                                                                                                                                                                                    
    # replace tl.any(cmp_old_new, axis=1) → sum > 0                                                                                                                                                                
    has_match_in_new = tl.sum(cmp_old_new.to(tl.int32), axis=1) > 0  # [BLOCK]                                                                                                                                     
    avail_mask = ~has_match_in_new                                     # [BLOCK]                                                                                                                                   

    # ---- shortage: fill empty slots in old ----                                                                                                                                                                  
    num_tokens_to_load = tl.sum(miss_mask.to(tl.int32),  axis=0)
    num_available_slot = tl.sum(avail_mask.to(tl.int32), axis=0)                                                                                                                                                   
    num_shortage_slot  = num_tokens_to_load - num_available_slot
                                                                                                                                                                                                                    
    empty_mask        = (old == -1)                                                                                                                                                                                
    empty_cumsum      = tl.cumsum(empty_mask.to(tl.int32), axis=0)
    selected_empty    = (empty_cumsum <= num_shortage_slot) & empty_mask                                                                                                                                           
    avail_mask        = avail_mask | selected_empty                                                                                                                                                                
                                                                                                                                                                                                                    
    # ---- compact miss tokens into available slots ----                                                                                                                                                           
    miss_vals = tl.where(miss_mask, new_with_offset, -1)   # [BLOCK]                                                                                                                                               
                                                                                                                                                                                                                    
    # rank of each available slot (0-based)                                                                                                                                                                        
    avail_rank = tl.cumsum(avail_mask.to(tl.int32), axis=0) - 1  # [BLOCK]                                                                                                                                         
    # rank of each miss token (0-based)                                                                                                                                                                            
    miss_rank  = tl.cumsum(miss_mask.to(tl.int32),  axis=0) - 1  # [BLOCK]
                                                                                                                                                                                                                    
    # for each available slot i, pick miss_vals[avail_rank[i]]                                                                                                                                                     
    # broadcast: avail_rank[:, None] == miss_rank[None, :]  -> [BLOCK, BLOCK]                                                                                                                                      
    avail_rank_b = tl.broadcast_to(avail_rank[:, None], (BLOCK, BLOCK))                                                                                                                                            
    miss_rank_b  = tl.broadcast_to(miss_rank[None, :],  (BLOCK, BLOCK))                                                                                                                                            
    miss_vals_b  = tl.broadcast_to(miss_vals[None, :],  (BLOCK, BLOCK))                                                                                                                                            
                                                                                                                                                                                                                    
    rank_match   = (avail_rank_b == miss_rank_b) & (miss_rank_b >= 0)  # [BLOCK, BLOCK]                                                                                                                            
    # for each row i, at most one column j matches → gather                                                                                                                                                        
    matched_vals = tl.sum(                                                                                                                                                                                         
        tl.where(rank_match, miss_vals_b, tl.zeros((BLOCK, BLOCK), tl.int32)),                                                                                                                                     
        axis=1                                                                                                                                                                                                     
    )  # [BLOCK]                                                                                                                                                                                                   
                                                                                                                                                                                                                    
    out = tl.where(avail_mask, matched_vals, tl.full((BLOCK,), -1, tl.int32))                                                                                                                                      
                
    # ---- remove req offset and store ----                                                                                                                                                                        
    out = tl.where(out >= 0, out - req_offset, tl.full((BLOCK,), -1, tl.int32))
    tl.store(out_ptr + row_off + cols, out.to(tl.int32), mask=mask)                                                                                                                                                
                
    # ---- update old in-place ----                                                                                                                                                                                
    # avail_mask 位置写入 new_with_offset（用 matched_vals + offset 反推不方便，
    # 直接用带 offset 的 out 版本）                                                                                                                                                                                
    new_val_with_offset = tl.where(out >= 0, out + req_offset, tl.full((BLOCK,), -1, tl.int32))                                                                                                                    
    updated_old = tl.where(avail_mask, new_val_with_offset, old)                                                                                                                                                   
    tl.store(old_ptr + row_off + cols, updated_old, mask=mask)


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
    )
    return out