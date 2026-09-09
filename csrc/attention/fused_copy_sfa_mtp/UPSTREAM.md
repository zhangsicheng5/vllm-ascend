# Generalized MTP operator import

The generalized LIM and copy-SFA entry points support eager and
`FULL_DECODE_ONLY` sparse offload, including PD decode with registered host KV
and explicit tail restoration. The separate MTP0 entry points remain unchanged.
See the offloading user guide for supported serving configurations. Validation
recorded below predates the latest kernel update unless explicitly dated otherwise.

## Pinned sources

| Operator | Repository branch | Exact commit |
| --- | --- | --- |
| LIM | `xwLearnsLLM/nanovllm-DSA-offload:ops_lim_standardization` | `012962af05f06bf1bdd089ca7e7e4357d021682f` |
| Copy-SFA | `xwLearnsLLM/nanovllm-DSA-offload:ops_copysfa_mtp_standardization` | `1518a90dd17592dc3aa96c16869cfde597a250b4` |

The repository is private. The import uses the pinned commits, independently
of subsequent branch updates. Original copyright notices are retained; the
upstream CANN license is included in the parent attention directory.

## Public contracts

| Property | LIM | Copy-SFA |
| --- | --- | --- |
| PyTorch entry point | `_C_ascend.npu_fused_li_manage_mtp` | `_C_ascend.npu_fused_copy_sfa_mtp` |
| Query rows per request | 1 through 14 | 1 through 16 |
| Query segmentation | cumulative ends, `int32[B]` | cumulative ends, `int32[B]` |
| Heads | 32 or 64 index heads | 8, 16, 32, 64 or 128 attention heads |
| TopK | 2048 per query | 2048 per query |
| Query miss counts | produces `int32[T]` | consumes `int32[T]` |
| Request miss lists | produces `int32[B,32768]` | consumes `int32[B,32768]` |
| Floating-point data | BF16 or FP16 | BF16 or FP16 |
| Memory device | all tensors on one NPU | one NPU; DRAM sources may be registered host GVA views |

Both entry points mutate caller-owned buffers and return `None`. Their schemas
replace the previous fixed-four-query MTP schemas. Existing MTP0 schemas remain
unchanged. Read the adjacent operator READMEs for complete argument definitions.

The native chained contract allows up to 14 queries per request. Serving
retains its existing limit of seven query rows and a 16256-token hot budget;
this update does not expand the advertised serving MTP range.
LIM's FP32 dequant scale arguments reserve the C8 ABI; this pinned kernel does
not use their values and does not implement quantized indexer computation.

LIM and copy-SFA share contiguous `int32[B,32768]` request miss buffers.
Only each request's `miss_counts` prefix is consumed. Eager and graph consumers
pass the LIM output buffers directly, including layers that share an indexer;
there are no miss-list bridge allocations or copies.

The original imports used LIM `86facf38362c0956d1b32c88faccf7dc26e87178` and
copy-SFA `98398fe4b5095b52cb2aa850c4d51defcfd8eed3`. The update applies the
source delta to the pins above while retaining local build and launcher
adaptations. LIM also changes workspace sizing, route handling, and its internal
slot/source codec. Copy-SFA changes the public adapter and tiler head checks;
it does not add native four-head support.

## Build adaptations

- CANN operator names and kernel entries use vLLM-Ascend names without the
  upstream `Nanovllm` prefix. LIM uses the repository's generated public ACLNN
  API instead of the upstream forwarding wrapper around an inner ACLNN API.
- The new copy-SFA attention headers are private to `fused_copy_sfa_mtp`.
  Imported host helper types are prefixed with `CopySfaMtp` to avoid symbol
  collisions with the existing sparse-flash-attention tiler.
- The conditional `FirstFillScatterCopy` dependency is built on A2/A3. A
  copy-SFA call enqueues this conditional copy followed by attention on the
  same stream. The copy is a device-side no-op for steady batches. If any
  request needs first fill, the request miss lists are copied before the
  whole batch uses HBM-only attention.
- Both launches use the existing `EXEC_NPU_CMD_ORDERED` tensor and workspace
  keepalive. The upstream environment-controlled workspace reuse optimization
  is not imported; no performance equivalence is claimed.
- Infer-shape sources follow the repository's `_infershape.cpp` discovery rule.
- Incremental builds regenerate the operator and binary registries so an
  existing build cache cannot omit newly added operator entries.

Use a clean `csrc/build` directory when upgrading these native contracts.
The CANN incremental build can reuse old device objects while recompiling the
host tiler. With the new LIM workspace sizing and miss-row stride, that mixture
produces untouched miss rows and invalid device-memory accesses. Preserve any
needed build evidence before clearing the generated cache, rebuild vLLM-Ascend,
and verify that generated, packaged, and installed kernel objects match.
Reinstalling vLLM is unnecessary.

## Eager colocated serving integration

`generalized_mtp.py` derives actual query prefix sums and a stable prefix
`L = floor((S-Q)/128)*128` from the current metadata, including adjusted draft
lengths after rejection. The cache budget is `C=min(L, topk_buffer_size)`.
Configuration requires a block-aligned budget in `[Q_max*2048,16256]`.

Each eager batch owns snapshots of its query ends, sequence lengths, pool
rows, and source block table. The draft metadata builder reuses its buffers;
retaining a view can change the query prefix after the batch has captured its
token count. LIM and copy-SFA consume the same batch snapshots.
When FlashComm padding is removed before drafting, the unpadded metadata also
preserves the remaining requests' offload pool-slot IDs.

For one through seven attention heads per rank, the serving adapter pads query
and query-rope heads to eight, then retains only the original output heads.
The native kernel accepts 8, 16, 32, 64 or 128 heads. GLM-5.2 TP16 uses
four logical heads per rank; the padding path has focused first-fill and
steady-state numerical coverage.

Each indexer owner maintains its own source-to-slot map, initialized with
`INT32_MIN`. Shared attention layers consume their owner's selection and miss
lists. First fill uses state `-2`, subsequent valid reuse uses `-1`. Changes in
request allocation generation, budget, or a decreasing stable prefix force
first fill again. Draft iterations recompute indexer metadata; they do not
reuse earlier iteration miss lists.

The runtime shares request miss buffers and restores the bounded dense tail
from registered host KV. It passes the logical length
`C + S - L` to copy-SFA. Sparse misses use the manager's registered CPU/GVA
views directly, following the MTP0 adapter convention. An ordinary unregistered
CPU tensor is not a valid device-readable DRAM allocation.

The initial eager colocated integration required `keep_device_kv_cache=true`.
Later changes added PD-only decode and target graph replay with an eager drafter.
Private padded graph rows use state `-3` and positive dummy cache budgets, with
zero miss counts and masked tail H2D/D2H. Real new/reset requests retain `-2`.
The kernel update preserves those lifecycle rules. LI C8 and SFA C8 remain
unsupported by this fused path.

## Validation

The changed tests use the actual registered operators, without substituting
local schemas when the extension is absent. NPU coverage includes heterogeneous
query counts and states, first-fill to steady transitions, causal copy-SFA
attention, exact persistent-cache writes, graph metadata changes, and the
LIM-to-copy-SFA shared storage.

Run on an Ascend environment built from this worktree:

```bash
pytest -q tests/ut/ops/test_fused_li_manage_mtp.py \
  tests/ut/ops/test_fused_copy_sfa_mtp.py
pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/test_fused_li_manage_mtp.py \
  tests/e2e/nightly/single_node/ops/singlecard_ops/test_fused_copy_sfa_mtp.py \
  tests/e2e/nightly/single_node/ops/singlecard_ops/test_generalized_mtp_metadata.py
```

On 2026-09-04, `quay.nju.edu.cn/ascend/vllm-ascend:nightly-main-a3`
(CANN 9.1.0, image vLLM 0.27.1 preserved) passed an Ascend-only package rebuild
and 11 focused schema, metadata, and numerical tests. Subsequent adapter
checks covered four-head padding, reused query-prefix buffers, and batch
metadata ownership. Eager GLM-5.2-w4a8 DP1/TP16/MTP3 serving completed three
prompts twice, including two prompts longer than 10K tokens. Runtime logs
showed generalized LIM and copy-SFA with CPU/GVA DRAM inputs and four logical
heads padded to eight, using a 64 GiB MemFabric 1.2.1 pool.

The initial comparison had stable long-prompt outputs but an unstable short
fallback response in both states. A subsequent baseline with FlashComm1,
shared-expert DP, and expert parallelism enabled produced all three expected
responses consistently. Offload with those settings exposed a missing
pool-slot field in `AscendCommonAttentionMetadata.unpadded()`. Preserving that
optional field passed five metadata checks, including removal of padded pool
rows and the no-offload case. The repaired offload run then matched all three
baseline texts exactly across two passes, with CPU/GVA copy-SFA execution
observed on all 16 TP ranks.

This is one repaired service launch and a focused sequential prompt set.
Model graph, PD-only, C8, performance, and memory-savings claims remain outside
this validation.
