# Nano Sparse Offload Optimization Plan

This document summarizes the measured performance issues and proposed fixes for
generalized nano LIM and copy-SFA in prefill/decode (PD) offload serving. The
highest-priority defect is repeated cold initialization of padded graph rows.
Full batches also incur tensor adaptation, metadata preparation, and host-gather
costs that can consume the benefit of fusion.

**Status:** the padded-row transfer fix and shared miss-buffer integration are
implemented. Native four-head support and the remaining optimizations are still
proposals. No speedup from the latest kernel update is claimed here. The
original profiling inspected vLLM-Ascend commit
`3aed3bf54fc4591a4a65b983635df9b5ecf1587e` with vLLM 0.27.1.

The nano operators were imported from
[`xwLearnsLLM/nanovllm-DSA-offload`](https://github.com/xwLearnsLLM/nanovllm-DSA-offload)
at these pinned reference commits:

| Operator | Upstream branch | Reference commit |
| --- | --- | --- |
| LIM | [`ops_lim_standardization`](https://github.com/xwLearnsLLM/nanovllm-DSA-offload/tree/ops_lim_standardization) | [86facf38362c0956d1b32c88faccf7dc26e87178](https://github.com/xwLearnsLLM/nanovllm-DSA-offload/commit/86facf38362c0956d1b32c88faccf7dc26e87178) |
| copy-SFA | [`ops_copysfa_mtp_standardization`](https://github.com/xwLearnsLLM/nanovllm-DSA-offload/tree/ops_copysfa_mtp_standardization) | [98398fe4b5095b52cb2aa850c4d51defcfd8eed3](https://github.com/xwLearnsLLM/nanovllm-DSA-offload/commit/98398fe4b5095b52cb2aa850c4d51defcfd8eed3) |

The head-count and miss-list contracts discussed below refer to these pinned
revisions and their vLLM-Ascend integration. Access to the private upstream
repository is required to open its links.

The current integration updates LIM to
[`012962af05f06bf1bdd089ca7e7e4357d021682f`](https://github.com/xwLearnsLLM/nanovllm-DSA-offload/commit/012962af05f06bf1bdd089ca7e7e4357d021682f)
and copy-SFA to
[`1518a90dd17592dc3aa96c16869cfde597a250b4`](https://github.com/xwLearnsLLM/nanovllm-DSA-offload/commit/1518a90dd17592dc3aa96c16869cfde597a250b4).
LIM now writes contiguous `int32[B,32768]` miss lists, shared directly with
copy-SFA in eager and graph execution. Copy-SFA accepts 8, 16, 32, 64 or 128
heads; four-head TP16 inputs still need padding and output compaction.
The `request_state=-3` padding fix is retained. The historical measurements and
cause analysis below describe the original pins, before these changes.
A controlled nano-disabled offload comparison with 128K input is planned;
new measurements must be recorded separately from the historical results.

See the [offloading guide](../../user_guide/feature_guide/layerwise_and_sparse_kv_cache_offloading.md)
for current serving support and the
[offloading design](../Design_Documents/layerwise_and_sparse_kv_cache_offloading.md)
for the existing data path.

## 1. Experimental context and findings

The comparison used GLM-5.2-w4a8 on A3, with BF16 attention/KV tensors, DP1/TP16,
FlashComm1, shared-expert DP, and expert parallelism. MTP3 was enabled on both
Prefill and Decode. Prefill used eager layerwise Memcache offload. Decode used
`FULL_DECODE_ONLY` target graphs with capture size `[16]`, an eager drafter,
`max_num_seqs=4`, an 8192-token hot cache, a 64 GiB host pool, and
`keep_device_kv_cache=false`. The KV block size was 128 tokens. The compared
backends were `fused_op_type="default"` and `fused_op_type="nano"`.

The image was `quay.nju.edu.cn/ascend/vllm-ascend:nightly-main-a3`, retaining its
installed vLLM 0.27.1. The environment used CANN 9.1.0, MemFabric 1.2.1, and
Memcache 1.2.0. Native changes in this plan require rebuilding vLLM-Ascend and
recapturing graphs at service startup; they do not require reinstalling vLLM.

### Unprofiled serving measurements

Three alternating A/B measurements per concurrency used the same 10,555-token
prompt and exact 82-token answer. Each run excluded four warmup requests and
measured eight requests. All 96 measured answers matched.

| Metric | Default, mean ± sample SD | Nano, mean ± sample SD |
| --- | --- | --- |
| C1 output throughput, tokens/s | 20.336 ± 0.119 | 6.848 ± 0.067 |
| C1 time per output token, ms | 32.105 ± 0.147 | 125.449 ± 0.635 |
| C4 output throughput, tokens/s | 59.961 ± 0.235 | 56.044 ± 4.880 |
| C4 time per output token, ms | 41.275 ± 0.423 | 41.669 ± 2.211 |
| C4 time to first token, ms | 1934.567 ± 27.117 | 2410.743 ± 283.110 |

C1 has a repeatable decode regression. C4 decode time per token is approximately
equal, while nano throughput and time to first token vary. Nano C4 throughput
was 59.171, 58.540, and 50.421 tokens/s; its time-to-first-token comparison is
inconclusive under the experiment's variance gate. These are full PD request
metrics. The mean throughput gap cannot all be assigned to steady attention.

Supplementary accepted/drafted-token counter deltas from rounds 2 and 3 were
95.24% versus 96.83% at C1, and 89.05% versus 93.49% at C4, for default versus
nano. Those counters include warmup traffic and export timing effects, so they
are not measured-request-only acceptance rates.

### Profiling evidence

Each C4 capture passed one serial smoke request, four warmups, and an eight-request
streaming wave: 26 exact answers across both captures. Profiling started after
four streams began generation, with a five-iteration delay and six active
iterations. All 16 ranks produced complete kernel and timeline outputs.
Client concurrency still did not guarantee four active requests in every replay.

The sixth nano replay had three real requests and a four-slot target graph.
On rank 0, the following target-stream kernel work increased:

| Work per target replay | First five replays, four requests | Sixth replay, three requests |
| --- | --- | --- |
| FirstFillScatterCopy, 78 calls | 0.671 ms average | 110.333 ms |
| FusedLiManageMtp, 21 calls | 2.066 ms average | 26.135 ms |

The difference is **133.73 ms of extra kernel work in one replay**. Index sharing
accounts for fewer LIM calls than attention layers. Three eager draft forwards
follow each target replay and are excluded from this table.

This behavior is visible across the TP group: nano first-fill duration sums over
the six replays range from 97.68 to 113.85 ms per rank, and LIM sums range from
41.55 to 41.82 ms. These sums are diagnostic evidence, not a prediction of the
end-to-end speedup from an unimplemented fix.

## 2. Prevent padded rows from triggering cold-fill H2D

### Cause in the profiled revision

`MtpGraphBuffers` assigns private hot-cache rows to graph padding. In
`prepare_layer` at the profiled revision, every inactive row receives
`request_state=-2` on every replay.
LIM interprets `-2` as first offload, reinitializes the mapping, and emits
`miss_count=cache_tokens`. Both the conditional copy helper and copy-SFA use a
batch-wide first-fill decision: any row with `miss_count >= cache_tokens` selects
the cold path.

Consequently, a dummy row repeatedly requests its entire 8192-token cache and
changes dispatch for the whole batch. The existing tail-copy `active_mask` does
not cover LIM or this first-fill path. At C1 in a four-request graph, the three
dummy rows request the following logical payload per layer, rank, and replay:

```text
3 * 8192 * (512 + 64) * 2 bytes = 27 MiB
```

This is requested payload, not a measured physical-bus byte count. The same
mechanism occurs during partial occupancy in a C4 workload.

### Preferred fix: explicit inactive-row handling

Retain fixed graph dimensions and extend the persistent active mask to the
operator interfaces. For example, a three-request replay has
`active_mask=[1, 1, 1, 0]`. Allocate the mask before capture, define a consistent
dtype, and update its contents on the ordered input stream before replay.

| Component | Required behavior for an inactive row |
| --- | --- |
| Generalized LIM | Skip indexer/cache-management work, mapping initialization, eviction, and miss-list generation; publish zero request and route miss counts and safe unused outputs. |
| FirstFillScatterCopy | Exclude the row from the batch cold-fill predicate and skip its transfers. |
| Generalized copy-SFA | Use the same cold-fill decision, skip inactive attention, and write zero outputs without reading dummy cache data. |
| Python graph metadata | Keep mask/storage addresses stable; preserve owner/generation tracking for real requests and mask propagation through index-sharing consumers. |

Conceptually, both native cold-fill predicates become:

```text
first_fill = any(active[i] && miss_count[i] >= cache_tokens[i])
```

This is a change to **two logical fused operator interfaces and three native
kernel components**, including copy-SFA's conditional first-fill helper. It also
requires updating schemas, adapter/tiler checks, fake/meta registrations where
applicable, and tests. Internal synchronization must remain consistent when
inactive work is skipped; a partial early return must not strand cooperating
Cube/Vector work or barriers.

Real new or reset requests still initialize their cache. Existing valid requests
retain the warm state. Padding performs no cache work. Updating mask values does
not change captured tensor shapes or require a new graph for each occupancy.

### Implemented first step and remaining work

The Python integration now assigns LIM's existing `-3` non-offload state only to
private dummy rows. Live new/reset requests retain `-2`, and live requests with
valid resident state retain `-1`. This makes both `miss_counts[B]` and
`topk_miss_counts[T]` zero for padding, while keeping its cache budget positive.
Thus padding triggers neither first-fill copying nor source-aware host gathers.

The allocator initializes private hot-cache rows and their tails once. Startup
capture also uses private rows, without marking live requests resident. Graph
metadata explicitly masks padded D2H slots with `-1`; the existing active mask
keeps their tail H2D descriptor lengths zero. Shapes and storage addresses stay
fixed across replays. Native operator interfaces and kernels are unchanged.

The focused regressions cover Q1/Q4, capture, changing occupancy, new/reset versus
warm requests, registered-host transfers, and live attention against a direct
logical-host reference. They poison padding's host source block to expose
accidental copies and check that its private HBM remains zero. Validation results
for a particular run must be recorded separately; the measurements above precede
this change and do not establish its throughput benefit.

This first step still executes dummy LI, identity-map writes, and attention.
The explicit inactive-row kernel handling proposed above remains future work.

Changing `-2` to `-1` alone can use an uninitialized resident map. Setting the dummy
cache size to zero also fails the current `0 >= 0` first-fill test. Masking only
tail transfers does not fix this defect.

FlashComm1 rounds model tokens to a multiple of TP size before graph dispatch.
Under TP16/MTP3, one request can therefore still use 16 model-token slots. Smaller
capture sizes alone do not remove the operator-padding problem in that setup.

## 3. Remove query-head padding and output compaction

The originally profiled copy-SFA interface accepts **8 or 128 query heads per
rank**, rather than arbitrary multiples of eight. Its Python adapter accepts
1–7 heads by padding to eight; other unsupported counts are rejected. These are
attention heads, distinct from LIM's indexer-head contract.

GLM-5.2 has 64 attention heads. With TP16, each rank has four, so the current
adapter performs:

```text
query:       [T, 4, 512] -> zero-filled [T, 8, 512]
query_rope:  [T, 4,  64] -> zero-filled [T, 8,  64]
output:      [T, 8, 512] -> out[:, :4].contiguous()
```

The first two steps allocate/fill buffers and copy the real heads. The final
slice generally needs a copy to compact the useful heads. These operations run
inside graph replay as well as eager execution.

Two approaches were discussed:

- **Native four-head support:** extend and validate the copy-SFA adapter, tiler,
  and kernel for four heads, then remove the padding and output compaction. The
  internal tiler already includes group-four handling, but public checks still
  restrict the interface. Internal support is not sufficient evidence that the
  complete operator is correct for four heads. Cube-M alignment to 16 also means
  removing logical head padding does not imply a twofold compute speedup.
- **DP2/TP8 configuration:** TP8 gives eight heads per rank and avoids this
  particular padding. With a contiguous output, the full-head slice's
  `.contiguous()` becomes a no-op. Input `.contiguous()` calls can still copy if
  their strides require it. DP2 changes replica count; TP8 is what changes local
  head count. This topology was not benchmarked in the investigation. Compare
  per-replica occupancy, routing, graph sizes, memory, and communication as well
  as global concurrency. It does not remove miss-list conversion or implement
  inactive-row handling.

## 4. Eliminate the miss-list bridge with shared storage

LIM produces contiguous `int32[B, 16384]` source and destination miss lists.
Copy-SFA consumes contiguous `int32[B, 32768]` lists. Here `B` denotes request
capacity, not KV block size. The current bridge copies both lists into the first
16384 columns of larger buffers on each attention invocation. At `B=4`, these two
copies move 512 KiB of metadata per invocation; actual miss counts are unchanged.

### Why preallocation alone does not remove the copies

The graph path **already preallocates** both larger buffers. A view such as
`large_buffer[:, :16384]` retains a 32768-element row stride. For `B > 1`, it is
not a contiguous `[B, 16384]` tensor. LIM rejects it and its kernel currently uses
hard-coded 16384-element offsets for miss-output rows. Conversely, a packed
16384-stride view of the same storage would put later rows at addresses that
copy-SFA does not expect.

### Implemented shared-buffer change

1. Preallocate source and destination buffers as `[B, 32768]`.
2. Extend LIM's output contract and indexing to write directly with that physical
   row stride. Keep its existing logical miss-count limit separate from storage
   stride; do not increase unrelated union/workspace capacities accidentally.
3. Pass those same buffers directly to copy-SFA, removing both bridge copies.
4. Initialize unused storage as required by the contract and consume only the
   valid `miss_counts` prefixes. Avoid clearing entire oversized buffers on every
   invocation when the unused suffix is never read.

This requires C++ adapter validation, tiling/infer-shape updates, and changes to
every kernel path that writes request miss rows, followed by a rebuild. Merely
changing Python allocation or relaxing the contiguity assertion is insufficient.
An alternative is to make copy-SFA and its conditional copy helper accept LIM's
smaller stride, while retaining larger-capacity support for existing users.

As an interim optimization, convert lists once after each LIM call and reuse
them through that indexer's shared-attention consumers. The observed target has
21 LIM calls and 78 attention calls. Ownership must prevent reuse after another
LIM invocation, draft step, or batch update overwrites the shared outputs.

### Meaning of the measured layout-copy overhead

All 16 nano ranks have 1944 view-copy calls over six replays; the baseline has no
calls with that operator name. Additional view-copy plus transpose-copy kernel
work averages **22.80 ms per rank over six replays, or 3.80 ms per replay**.
Head adaptation, output compaction, and miss-list bridging are the identified
source-level targets. Their individual shares require separate ablations.

These are copies within NPU memory. Host-to-device and device-to-host KV
transfers are separate. The 3.80 ms value is accumulated kernel work across
layers, not a per-layer/per-token cost or an additive end-to-end latency penalty.

## 5. Reduce tail descriptor preparation and repeated tail H2D

The RD2H connector populates the registered host main cache, while copy-SFA also
needs nano's rank-local HBM tail. The current correct path restores that bounded
tail before attention on every invocation, after the current KV's D2H and the
existing TP ordering fence.

### First stage: prepare geometry once

`restore_nano_tail` currently calls `prepare_tail_copy` per layer. Lengths, masks,
block lookup, circular offsets, and most descriptor geometry depend on the batch;
the main per-layer differences are fixed K/V base addresses.

Prepare relative geometry once per batch update and populate a persistent
descriptor bank indexed by layer, component, request, and span. Give each layer
stable views for graph replay. Preserve two spans per component/request, zero
lengths for inactive rows, the source block lookup, and the current fence.
Different host blocks need not be physically contiguous, and the circular tail
can wrap, so a single large contiguous copy is not generally equivalent.

### Second stage: import once, then append locally

After validating descriptor reuse, consider restoring the imported tail only
when a real request arrives or its cache state resets. Subsequent locally
computed KV can be written directly into each rank's circular HBM tail, while
TP0 retains D2H writes for host history.

This requires explicit per-request/per-layer initialization state and readiness
ordering. Handle partial prefill blocks, tail wrap, slot generations, MTP
acceptance/rejection, overwritten draft positions, and fallback transitions.
One-shot flags must advance only after the associated ordered work completes.
Keep captured addresses stable. This is a larger lifecycle change than descriptor
reuse and must not remove required KV-transfer readiness or host-write fences.
The default path also broadcasts; all broadcast time cannot be labeled extra
nano overhead.

## 6. Reduce CPU/NPU metadata round trips

`make_mtp_batch` owns snapshots because target/draft builders reuse their tensors.
It then reads query ends, sequence lengths, and pool entries through
`.cpu().tolist()`. Graph preparation performs further copies, and `prepare_layer`
constructs and copies lifecycle-state tensors even when their values are unchanged.

Proposed changes are to retain owned CPU scheduler values when available, batch
H2D metadata updates into persistent buffers, and skip state-buffer writes when
the state tuple is unchanged. Continue updating request ownership, generations,
resident-prefix bookkeeping, and the active mask. Do not replace owned snapshots
with aliases to mutable draft metadata or freeze request state at capture time.

The rank-0 traces contain 774 stream-synchronize calls for nano versus 126 for
default. However, inclusive CPU event durations include waits for device work:
waiting moves from baseline `mla_forward` into nano tensor conversions. The
approximately 742 ms recorded under nano `aten::to` is not 742 ms of additional
CPU computation. Attribute CPU overhead using ordered timelines and ablations,
without adding nested waits or overlapping stream durations twice.

## 7. Tune the host-miss gather path

In five full-batch nano target replays, attention totals 9.44–22.67 ms. Average
target copy-SFA duration ranges from about 121–127 us in the faster replays to
291 us in the slowest. Average matrix-multiply time stays near 8.6–8.8 us, while
average AIV memory-read time rises from about 45–47 us to 94 us. This supports
investigating gathering and memory stalls; it does not measure the miss rate or
prove that all latency variation comes from one pipeline stage.

After the padding and adaptation fixes, compare:

- Current source-aware attention, which gathers misses from registered host
  memory while processing attention.
- Request-union miss prefetch into HBM, followed by HBM-only attention. The
  existing ordered conditional copy launch provides a starting point for an
  explicit prefetch mode, with matching copy-SFA dispatch.

Measure the crossover over actual query widths, head counts, occupancy, and miss
volume before considering device-side adaptive selection. Both alternatives must
use the same LIM map and attend the same TopK prefix plus dense causal tail.
Compare total attention/onload work and the critical path: baseline attention
alone excludes work that nano's source-aware kernel performs internally.

The default and nano attention contracts also differ in their explicit handling
of the dense tail. Identical output strings do not establish identical
intermediate attention tensors. Performance tuning must preserve the intended
nano contract rather than remove required tail attention.

## 8. Implementation order and validation

| Priority | Change | Completion evidence |
| --- | --- | --- |
| 1 | Inactive-row handling across LIM, first-fill copy, and SFA | Dummy rows generate no cold fills, H2D, cache mutation, or attention; real cold requests still initialize correctly. |
| 2 | Shared miss-list storage; query/output adaptation | Bridge kernels disappear; native four-head support or TP8 removes the specific head-padding copies where tensors are contiguous. |
| 3 | Tail descriptor reuse and metadata-update batching | Fewer descriptor kernels and small transfers, with identical descriptor values and request lifecycle behavior. |
| 4 | Host-gather/prefetch alternatives | Measured improvement on matched miss patterns and preserved numerical/cache results. |
| 5 | One-time imported-tail restore and local KV append | Correct admission, rejection, wrap, reuse, and fallback behavior, followed by a throughput improvement. |

Keep the changes independently reviewable and measure them one at a time, even
if interface changes share a native rebuild. DP2/TP8 is a separate topology
experiment, not evidence that a TP16 code optimization improved throughput.

The minimum validation matrix should cover:

- Eager and target `FULL_DECODE_ONLY`, with eager MTP3 drafting retained.
- B4 to B3 to B1 to B4, capture-time dummy inputs, real cold/warm rows mixed with
  padding, request-slot reuse, and changed request generations.
- MTP rejection, decreasing stable prefixes, block boundaries, circular-tail
  wrap, short-prompt fallback, and prompts exceeding the 8192-token resident
  budget. Include the existing approximately 10K-token prompt.
- Existing operator-supported query widths/dtypes and native head counts after
  any ABI change; add four-head cases before removing its padding. Test `B > 1`
  explicitly for the shared miss-list row stride.
- Exact miss counts, mapping/cache contents, inactive-row isolation, descriptor
  contents, numerical operator references, and end-to-end output correctness.
- Unprofiled alternating A/B runs at C1 and C4 with fixed topology, warmup
  exclusion, actual output-token counts, acceptance counters scoped to measured
  traffic, and variance reporting.

Use profiling to confirm removal of the targeted work. Record actual occupancy
and distinguish target graphs from eager draft calls; four live client streams
alone did not prevent a later three-request replay in this investigation. Do
not use profiled request throughput as the optimized serving throughput.

## 9. Source and evidence map

Paths below are relative to the vLLM-Ascend repository root.

| Area | Source |
| --- | --- |
| Graph padding, mask, lifecycle states, persistent miss buffers | `vllm_ascend/distributed/kv_transfer/sparse_kv_offload/generalized_mtp_graph.py` |
| Head adaptation, eager snapshots, miss-list bridge | `vllm_ascend/distributed/kv_transfer/sparse_kv_offload/generalized_mtp.py` |
| Copy-SFA invocation and output compaction | `vllm_ascend/attention/sfa_kv_offload.py` |
| Tail descriptors and restore ordering | `vllm_ascend/distributed/kv_transfer/sparse_kv_offload/nano_cache.py`, `sparse_kv_offload_manager.py` in the same directory |
| LIM contract and miss-output indexing | `csrc/attention/fused_li_manage_mtp/` |
| Copy-SFA contract, dispatch, and attention | `csrc/attention/fused_copy_sfa_mtp/` |
| Conditional first-fill transfer | `csrc/attention/first_fill_scatter_copy/` |
| FlashComm1 token padding and graph dispatch | `vllm_ascend/worker/model_runner_v1.py` |

The retained workspace investigation is
`.vaws-local/profiling-investigation/nano-pd-20260906/`, with Run Manifest ID
`nano-pd-root-cause-20260906`. Runtime state and raw traces remain untracked;
the following artifact identifiers are provenance references, not public links:

- `rank0-comparison.json` and `cross-rank-comparison.json` contain the timing
  calculations and source references. The earlier benchmark is under
  `.vaws-local/performance-regression/nano-vs-unfused-10k-20260905/`.
- In `nano/c4/copy-index-window.csv`, original kernel row 51369 has a `3,128`
  source block-table input before the sixth graph launch. The target SFA retains
  four request slots, while subsequent eager draft rows 61192, 61386, and 61576
  have three-request metadata. This establishes the occupancy transition.
- Each backend's `c4-stats/operator_summary.csv` provides rank-specific raw row
  ranges. Workload correctness and all 16 ranks' profiling-output checks passed. Automatic
  main-layer inventories differ between backends and fused-name taxonomy
  coverage is incomplete, so paired target/draft attribution uses observed
  graph launches, native static/dynamic records, and shapes.

Setup failures were excluded from the successful captures. No source
optimization was applied during the paired profiling runs. Full recursive
cross-role source/build identity and weight-shard checksums were not established;
the D binaries and non-backend settings were held fixed within the comparison.
A fresh C1 trace and the proposed fixes' performance results are not part of
this evidence. Model services and the proxy were stopped after capture, with
MetaService and raw profiling artifacts preserved.
