# Stateful TopK Triton UB Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the stateful `token_to_slot` topk path so engine startup no longer fails with Ascend Triton `ub overflow` at `topk=2048`.

**Architecture:** Keep stateful topk as the default path in `sfa_v1.py`, but reduce per-kernel UB pressure in `get_topk_indices.py`. Use `int32` internal state and direct rank-based scatter/gather instead of materializing rank matching loops. Keep the existing AscendC/custom op fallback path intact for quick rollback and benchmark comparison.

**Tech Stack:** Python 3, PyTorch NPU tensors, Triton Ascend backend, vLLM Ascend SFA offload path.

---

### Task 1: Reduce Stateful Triton Kernel UB Usage

**Files:**
- Modify: `vllm_ascend/ops/triton/get_topk_indices.py:25-229`
- Test: `python3 -m py_compile vllm_ascend/ops/triton/get_topk_indices.py`

- [ ] **Step 1: Confirm the failure signature**

Read `examples/output.txt` and verify the root cause is Triton compile-time UB overflow:

```bash
rg -n "ub overflow|_get_topk_buffer|USE_STATEFUL|Triton|NPUModelRunner" examples/output.txt
```

Expected: output contains `_get_topk_buffer` before the failing op and repeated lines like:

```text
error: ub overflow, requires 1868288 bits while 1572864 bits available
```

- [ ] **Step 2: Change internal `slot_to_token` state to `int32`**

In `CacheMissTopKState.__init__`, change:

```python
self.slot_to_token = torch.full(
    (max_num_reqs, topk), -1, dtype=torch.int64, device=device
)
```

to:

```python
self.slot_to_token = torch.full(
    (max_num_reqs, topk), -1, dtype=torch.int32, device=device
)
```

Rationale: token ids are bounded by `TOKEN_LIMIT_PER_REQ = 65536`, so `int32` is sufficient and halves vector storage in the hot Triton kernel.

- [ ] **Step 3: Simplify `_stateful_find_miss_kernel`**

Replace the compact-miss loop that broadcasts `(BLOCK, SUB_BLOCK)` rank matches with direct scatter:

```python
new_token = tl.load(new_topk_ptr + row_off + cols, mask=mask, other=-1).to(tl.int32)
valid_new = (new_token >= 0) & (new_token < token_limit)
safe_new = tl.where(valid_new, new_token, tl.zeros((BLOCK,), tl.int32))
prev_slot = tl.load(
    token_to_slot_ptr + req_id * token_limit + safe_new,
    mask=valid_new,
    other=-1,
)
is_miss = (prev_slot < 0) & valid_new
miss_rank = tl.cumsum(is_miss.to(tl.int32), axis=0) - 1
num_miss = tl.sum(is_miss.to(tl.int32), axis=0)
tl.store(miss_counts_ptr + pid, num_miss.to(tl.int32))
tl.store(
    miss_tokens_ptr + miss_row_off + cols,
    tl.full((BLOCK,), -1, tl.int32),
    mask=mask,
)
tl.store(
    miss_tokens_ptr + miss_row_off + miss_rank,
    new_token,
    mask=is_miss,
)
```

Expected behavior: `miss_tokens[row, 0:miss_count]` contains exactly the new tokens not present in `token_to_slot`.

- [ ] **Step 4: Simplify `_stateful_assign_slots_kernel`**

Keep the existing stale-slot detection loop, but use `int32` intermediates and replace the second rank-matching loop with direct gather:

```python
new_token = tl.load(new_topk_ptr + row_off + cols, mask=mask, other=-1).to(tl.int32)
old_token = tl.load(slot_to_token_ptr + req_id * topk + cols, mask=mask, other=-1).to(tl.int32)
miss_count = tl.load(miss_counts_ptr + pid).to(tl.int32)
avail_count = tl.zeros((BLOCK,), tl.int32)
...
avail_rank = tl.cumsum(avail_mask.to(tl.int32), axis=0) - 1
take_miss = avail_mask & (avail_rank >= 0) & (avail_rank < miss_count)
safe_rank = tl.where(take_miss, avail_rank, tl.zeros((BLOCK,), tl.int32))
out_vals = tl.load(
    miss_tokens_ptr + row_off + safe_rank,
    mask=take_miss,
    other=-1,
).to(tl.int32)
```

Expected behavior: `out_vals[slot]` is loaded from the compact miss list at the same rank as the free slot rank.

- [ ] **Step 5: Run syntax check**

```bash
python3 -m py_compile vllm_ascend/ops/triton/get_topk_indices.py
```

Expected: no output and exit code 0.

### Task 2: Make Benchmark Fallback Registration Explicit

**Files:**
- Modify: `benchmarks/ops/ben_cache_miss_topk_state.py:1-12`
- Test: `python3 -m py_compile benchmarks/ops/ben_cache_miss_topk_state.py`

- [ ] **Step 1: Import `ascend_kernel` in benchmark**

Add this import near the other runtime registration imports:

```python
import ascend_kernel  # noqa: F401
```

Expected behavior: `torch.ops.npu.get_cache_miss_topk_indices` is registered before `hasattr(torch.ops.npu, "get_cache_miss_topk_indices")` runs.

- [ ] **Step 2: Run syntax check**

```bash
python3 -m py_compile benchmarks/ops/ben_cache_miss_topk_state.py
```

Expected: no output and exit code 0.

### Task 3: Verify The Patch Shape

**Files:**
- Modify: none
- Test: `git diff --stat` and targeted grep checks

- [ ] **Step 1: Confirm stateful remains default**

```bash
rg -n "USE_STATEFUL_TOPK_CACHE_MISS|USE_ASCENDC_CACHE_MISS_TOPK_FALLBACK" vllm_ascend/attention/sfa_v1.py
```

Expected:

```text
USE_STATEFUL_TOPK_CACHE_MISS = True
USE_ASCENDC_CACHE_MISS_TOPK_FALLBACK = False
```

- [ ] **Step 2: Confirm no rank-matching scatter loop remains in stateful kernels**

```bash
rg -n "target_ranks|broadcast_to\\(miss_rank|broadcast_to\\(avail_rank" vllm_ascend/ops/triton/get_topk_indices.py
```

Expected: no matches in `_stateful_find_miss_kernel` or `_stateful_assign_slots_kernel`; matches in the legacy non-stateful Triton kernel are acceptable.

- [ ] **Step 3: Review diff**

```bash
git diff -- vllm_ascend/ops/triton/get_topk_indices.py benchmarks/ops/ben_cache_miss_topk_state.py
```

Expected: diff only changes stateful kernel internals, `CacheMissTopKState.slot_to_token` dtype, and benchmark `ascend_kernel` import.

