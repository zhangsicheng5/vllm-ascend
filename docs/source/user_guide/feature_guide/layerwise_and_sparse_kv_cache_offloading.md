# Layerwise and Sparse KV Cache Offloading Guide

This guide explains how to configure:

- Layerwise KV cache offloading during the Prefill phase
- Sparse KV cache offloading during the Decode phase
- Combining both features in a disaggregated Prefill/Decode deployment

For the underlying architecture and implementation details, see
[Layerwise and Sparse KV Cache Offloading Design](../../developer_guide/Design_Documents/layerwise_and_sparse_kv_cache_offloading.md).

## Supported Models

The combined deployment currently supports the following sparse-attention
model families:

- [GLM-5.1](../../tutorials/models/GLM5.md)
- [GLM-5.2](../../tutorials/models/GLM5.2.md)
- [DeepSeek-V3.2](../../tutorials/models/DeepSeek-V3.2.md)

Other sparse-attention models have not been validated.

## 1. Install Dependencies

The installation steps are grouped by hardware. Only A3 series is currently
supported.

### Prefill Build Dependencies

=== "A3 series"

    Prefill requires MemFabric Hybrid and Memcache Hybrid. Install them in this
    order.

    #### MemFabric Hybrid

    Install MemFabric Hybrid release 1.2 on every Prefill node. This release
    requires NPU driver `25.5.1` or later.

    ```bash
    pip uninstall -y memfabric_hybrid
    git clone -b release/1.2 https://gitcode.com/Ascend/memfabric_hybrid.git
    cd memfabric_hybrid
    bash script/build_and_pack_run.sh
    bash output/memfabric_hybrid-1.2.0_linux_aarch64.run
    ```

    #### Memcache Hybrid

    Install Memcache Hybrid after MemFabric Hybrid:

    ```bash
    git clone https://gitcode.com/Ascend/memcache.git
    cd memcache
    git submodule update --recursive --init
    git -c submodule.3rdparty/memfabric_hybrid.branch=release/1.2 \
        submodule update --remote --recursive 3rdparty/memfabric_hybrid
    bash script/build_and_pack_run.sh --build_mode RELEASE
    bash output/memcache_hybrid-*_linux_aarch64.run
    ```

    Configure `mmc-meta.conf`:

    ```ini
    ock.mmc.meta_service_url = tcp://<META_HOST>:5000
    ock.mmc.meta_service.config_store_url = tcp://<CONFIG_STORE_HOST>:6000
    ock.mmc.meta.lease_ttl_ms = 30000
    ock.mmc.log_level = error
    ```

    Configure `mmc-local.conf` on every Prefill node:

    ```ini
    ock.mmc.meta_service_url = tcp://<META_HOST>:5000
    ock.mmc.local_service.config_store_url = tcp://<CONFIG_STORE_HOST>:6000
    ock.mmc.log_level = error
    ock.mmc.local_service.world_size = 256
    ock.mmc.local_service.protocol = device_sdma
    ock.mmc.local_service.dram.size = 10GB
    ```

    The two files must use the same MetaService endpoint. The LocalService
    Config Store endpoint must match the MetaService Config Store endpoint.

    - Set `world_size` to the maximum supported LocalService rank count.
    - Use `device_sdma` with HCCS.
    - Set `dram.size` to at least the total KV cache size required by the target
      sequence length and concurrency divided by the number of Prefill ranks.
      Round the result up to a whole GiB.

    > **Note:** The configuration paths below assume Python 3.11.10. If you use
    > another Python version, replace the Python installation and
    > `site-packages` directories with those of the active environment. Locate
    > its `site-packages` directory with:
    >
    > `python -c "import site; print(site.getsitepackages())"`

    Start MetaService in a separate process:

    ```bash
    source /usr/local/memcache_hybrid/set_env.sh
    source /usr/local/memfabric_hybrid/set_env.sh
    export MMC_META_CONFIG_PATH=/usr/local/python3.11.10/lib/python3.11/site-packages/memcache_hybrid/latest/config/mmc-meta.conf
    python -c "from memcache_hybrid import MetaService; MetaService.main()"
    ```

    Prepare every Prefill node before starting vLLM:

    ```bash
    source /usr/local/memcache_hybrid/set_env.sh
    source /usr/local/memfabric_hybrid/set_env.sh
    export MMC_LOCAL_CONFIG_PATH=/usr/local/python3.11.10/lib/python3.11/site-packages/memcache_hybrid/latest/config/mmc-local.conf
    export MEMFABRIC_HYBRID_EXTEND_LIB_PATH=/usr/local/memfabric_hybrid/1.2.0/aarch64-linux/lib64
    export PYTHONHASHSEED=0
    ```

### Decode Build Dependencies

=== "A3 series"

    > **Important:** MemFabric Hybrid release 1.2 must be installed on both
    > Prefill and Decode nodes. Memcache Hybrid is required only on Prefill.

    Use the same MemFabric Hybrid build and installation commands shown above.
    Decode also requires Clang and OpenMP. Prepare every Decode node:

    ```bash
    source /usr/local/memfabric_hybrid/set_env.sh
    export MEMFABRIC_HYBRID_EXTEND_LIB_PATH=/usr/local/memfabric_hybrid/1.2.0/aarch64-linux/lib64
    clang --version
    ls "$(clang --print-resource-dir)/include/omp.h"
    ```

    If Clang or OpenMP is missing:

    ```bash
    apt-get update
    apt-get install -y clang libomp-dev
    ```

    If the image provides a specific Clang version, install the matching OpenMP
    package, for example `libomp-17-dev` for Clang 17.

## 2. Layerwise KV Cache Offload on Prefill

Use this mode on a dedicated Prefill node with:

- `kv_role: "kv_producer"`;
- the Memcache backend;
- an MLA, SFA, or DSA attention backend; and
- eager execution.

For a combined deployment, Prefill TP must be greater than or equal to Decode
TP and divisible by it.

Add the following options to the Prefill launch command. `MultiConnector` lets
`AscendStoreConnector` offload layer buffers to Memcache while
`SfaRemoteD2HConnector` exposes the same buffers to Decode:

```bash
--enforce-eager \
--kv-transfer-config '{
    "kv_connector": "MultiConnector",
    "kv_role": "kv_producer",
    "kv_connector_extra_config": {
        "connectors": [
            {
                "kv_connector": "SfaRemoteD2HConnector",
                "kv_role": "kv_producer",
                "kv_connector_extra_config": {
                    "transfer_backend": "memfabric"
                }
            },
            {
                "kv_connector": "AscendStoreConnector",
                "kv_role": "kv_producer",
                "kv_connector_extra_config": {
                    "backend": "memcache",
                    "use_layerwise": true,
                    "layerwise_num_shared_buffers": 3,
                    "layerwise_independent_layers": [0]
                }
            }
        ]
    }
}'
```

Do not set `sparse_kv_offload_config` on Prefill. The
`AscendStoreConnector` entry uses the following buffer options:

| Parameter | Description |
| :--- | :--- |
| `layerwise_num_shared_buffers` | Number of reusable NPU buffers. Start with two to four and tune for memory and transfer bandwidth. |
| `layerwise_independent_layers` | Layers that keep dedicated buffers. The default is `[0]`; `"all"` disables cross-layer reuse. |

The following log confirms that buffer reuse is enabled:

```text
Layerwise KV cache reuse merged ... descriptors into ... descriptors using ... buffer assignments.
```

## 3. Sparse KV Cache Offload on Decode

Requirements:

- use disaggregated Prefill/Decode deployment;
- enable the feature only on Decode; and
- use Model Runner V1.

Add the following options to the Decode launch command:

```bash
--additional-config '{
    "sparse_kv_offload_config": {
        "enabled": true,
        "topk_buffer_size": 4096,
        "dram_size_per_dp_GB": 128
    }
}' \
--kv-transfer-config '{
    "kv_connector": "SfaRemoteD2HConnector",
    "kv_role": "kv_consumer",
    "kv_port": 20050,
    "kv_connector_extra_config": {
        "transfer_backend": "memfabric",
        "use_layerwise": true
    }
}'
```

On Decode, reserve
`decode_data_parallel_size * decode_tensor_parallel_size` consecutive ports
starting from `kv_port`.

| Parameter | Description |
| :--- | :--- |
| `topk_buffer_size` | Device hot-buffer size. It must be at least `index_topk` and divisible by `block_size`. Twice `index_topk` is a practical starting point. |
| `dram_size_per_dp_GB` | Host memory reserved per DP rank. It must hold the full KV cache. TP ranks share this pool. |
| `keep_device_kv_cache` | Debug-only option that retains the full device KV cache. Keep it `false` in production. |

## 4. Start the P/D Proxy

Start Prefill and Decode with the configurations above. After both nodes are
ready, start the proxy:

```bash
python examples/disaggregated_prefill_v1/load_balance_proxy_layerwise_server_example.py \
    --host 127.0.0.1 \
    --port 9000 \
    --prefiller-hosts 127.0.0.1 \
    --prefiller-ports 8100 \
    --decoder-hosts 127.0.0.1 \
    --decoder-ports 8200
```

For multi-node deployment, advertise reachable addresses instead of
`0.0.0.0`. Send inference requests to the proxy port (`9000` in this example).

## 5. Generalized LIM and Copy-SFA

The `nano` backend uses generalized LIM and copy-SFA when speculative decoding
is enabled, or when Sparse Decode Offload uses `keep_device_kv_cache=false`.
It supports up to six speculative tokens (seven query rows per request) and
BF16/FP16 indexer inputs. The native copy-SFA kernel accepts 8, 16, 32, 64 or
128 attention heads per rank. The adapter pads 1 through 7 heads to 8
and discards the padded outputs; GLM-5.2 TP16 uses this path with 4 heads.

Install the Decode dependencies above. With the
`quay.nju.edu.cn/ascend/vllm-ascend:nightly-main-a3` image, preserve its
vLLM 0.27.1 installation and build/reinstall the updated vLLM-Ascend sources.
For these native ABI updates, start with a clean generated `csrc/build`
directory; an incremental build can retain old device kernels with new host
tiling code. Preserve any needed build artifacts before clearing that cache.
Install matching MemFabric native libraries and Python bindings: the wheel
alone does not supply `libmf_hybm_accoffload.so`. The recorded validation used
MemFabric 1.2.1 from commit `0259c97a2fa01022708dbeffe4b5c5672bc424dc`
in an isolated prefix. Source that installation's `set_env.sh` and ensure its
Python bindings precede any image-provided version on `PYTHONPATH`.
Colocated decode offload does not require a Memcache service.

An A3 colocated DP1/TP16/MTP3 configuration is shown below. Adjust the native
package path if using an isolated prefix. The 64 GiB host pool was validated
for this sequence length and concurrency; check capacity again when changing
those settings. Shared-expert DP requires expert parallelism in this branch;
the service data-parallel size remains 1.

```bash
source /usr/local/memfabric_hybrid/set_env.sh
export MEMFABRIC_HYBRID_EXTEND_LIB_PATH=/usr/local/memfabric_hybrid/latest/aarch64-linux/lib64
export VLLM_ASCEND_SPARSE_KV_OFFLOAD_DEBUG=1
vllm serve /mnt/weight/GLM-5.2-w4a8/ \
    --tensor-parallel-size 16 \
    --data-parallel-size 1 \
    --enforce-eager \
    --max-num-seqs 2 \
    --max-model-len 16384 \
    --enable-expert-parallel \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
    --additional-config '{
        "enable_flashcomm1": true,
        "enable_shared_expert_dp": true,
        "enable_sparse_li_c8": false,
        "enable_sparse_sfa_c8": false,
        "sparse_kv_offload_config": {
            "enabled": true,
            "fused_op_type": "nano",
            "keep_device_kv_cache": true,
            "topk_buffer_size": 8192,
            "dram_size_per_dp_GB": 64
        }
    }'
```

This configuration completed repeated short and 10K-plus-token requests with
generalized LIM and CPU/GVA copy-SFA execution on A3. With FlashComm1 and
shared-expert DP enabled, all three prompt texts matched the offload-disabled
baseline exactly across two passes. This is a focused eager validation,
not a broad model-accuracy or performance result. Short prompts and mixed
prefill/decode batches use ordinary SFA on the retained device cache. Pure
decode batches use the generalized operators when every request has at least
2048 stable, fully offloaded tokens. Test prompts longer than 8192 tokens to
exercise MTP3 sparse replacement beyond the resident budget, and check
operator execution as well as generated output. The retained full cache makes
this a correctness/debug setup; it does not demonstrate memory savings.

### Experimental full-decode graph integration

The working implementation accepts `FULL_DECODE_ONLY` with
`keep_device_kv_cache=true`. The target-model graph path was validated with
DP1, TP16, MTP3, FlashComm1, shared-expert DP, and eager speculative drafting.
All 16 ranks captured and replayed the offload graph, and all three fixed
prompt texts matched the eager offload result exactly across two passes. A
standalone registered host-memory D2H → LIM → copy-SFA graph also passed five
replays covering fills, hits, replacement, resets, and changing transfer
descriptors.

Graph entries own persistent query boundaries, sequence lengths, block tables,
LIM states, and transfer metadata. Request ownership is refreshed before
replay, and padding uses hot-cache rows outside the scheduler's request pool.
Short, mixed, and nonuniform batches use eager execution on the retained full
device cache in this colocated configuration. Other graph modes remain
unsupported. The separate PD path is described below.

For the DP1/TP16/MTP3 validation case, use `--max-num-seqs 4` with
`--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[16]}'`.
FlashComm1 requires capture sizes divisible by TP size. With MTP3 and
`max_num_seqs=2`, the dispatcher filters out capture size 16 because its
uniform decode capacity is only eight tokens; enabling the configuration alone
therefore does not prove graph execution. Check capture and replay logs.

Run target graphs with `"enforce_eager": true` in `--speculative-config` for
GLM models. GLM speculative drafting remains eager because its current graph
profile inputs are incompatible with the sequence-parallel padding used by
this configuration. Compare a matched eager baseline with the same prompts
twice, and verify in the logs that long requests replay the offload graph.
Enabling `FULL_DECODE_ONLY` alone is not sufficient evidence of graph use.

### Experimental PD host-to-tail integration

For a Decode node receiving KV through `SfaRemoteD2HConnector`, set
`fused_op_type="nano"` and `keep_device_kv_cache=false`. The connector transfers
the main KV history into the shared host pool and the indexer cache into device
memory. It does not populate nano's two device tail blocks.

Disable prefix caching on Decode with `--no-enable-prefix-caching` for both
the `default` and `nano` offload backends. Decode prefix reuse is not supported
by the current PD offload integration.

The Decode attention path now writes its new KV into the host pool, then
restores only the tail needed by copy-SFA. For block size `B=128`, sequence
length `S`, and query width `Q`, the tail starts at
`L=floor((S-Q)/B)*B` and covers `[L,S)`. That span crosses at most two blocks
for the supported query widths. Each KV component uses up to two contiguous
copies, following the actual host block table and the request's device pool
row. Full-history KV is not materialized on the device.

Graph entries own persistent copy descriptors and an active-request mask.
Descriptor construction and H2D execute inside the captured graph before
copy-SFA, so every replay uses current sequence lengths, block tables, and pool
rows. Padding has zero copy lengths. Restoring the tail on every invocation
also replaces stale contents after speculative-token rejection or request-slot
reuse. Main-cache and indexer-cache mappings are tracked separately; draft
metadata may share them only when both caches belong to the same KV group.

Short requests and nonuniform query batches remain eager. They gather each
query's causal TopK KV set from host memory into a bounded device row before
ordinary SFA, then invalidate LIM residency before the next fused batch.
This fallback also works with `keep_device_kv_cache=false`. Each prepared MTP
step owns its query boundaries and sequence lengths, so preparing a later draft
step cannot change the earlier step's fallback routing or causal lengths.

The initial PD validation used GLM-5.2 on two A3 nodes, DP1/TP16,
FlashComm1, shared-expert DP, eager execution, and MTP disabled. Prefill used
Memcache layerwise offload with three shared buffers and independent layer 0;
Decode used an 8192-token hot buffer and a 64 GiB host pool. All six requests
matched the existing PD offload baseline exactly, including overlapping short
and 10K-plus-token requests and two passes through each prompt. A separate
registered-host-memory test passed five graph replays while changing tail
lengths, block mappings, pool rows, source contents, and padding. This does
not establish broad model accuracy or performance.

The matched MTP3 PD configuration was also validated with MTP enabled on both
Prefill and Decode. Eager execution and target-model `FULL_DECODE_ONLY` each
completed the same six requests with exact output matches to the PD baseline
and to each other. The long/short overlap lasted 11.5 seconds in eager mode and
12.0 seconds in graph mode. All 16 Decode ranks received metadata for 79 main
layers, including the draft layer, and logged target-model graph replay with
16 input tokens. GLM speculative drafting stayed eager. This is focused
correctness evidence for A3 DP1/TP16, GLM-5.2 BF16, MTP3, an 8192-token hot
buffer, and a 64 GiB host pool; performance tuning remains separate.

A matched three-run PD comparison used a 10,555-token prompt, 82 generated
tokens, four warmup requests, and eight measured requests per run. All 96
measured responses matched. With the same target graph and eager MTP3 drafter,
mean output throughput was 20.34 tokens/s for `default` versus 6.85 for `nano`
at concurrency 1, and 59.96 versus 56.04 at concurrency 4. Nano concurrency-4
results varied from 50.42 to 59.17 tokens/s. These rates include the full PD
request path. Single-request decode time per token increased from 32.10 to
125.45 ms; at concurrency 4 it remained approximately 41 ms for both backends.
Inactive graph rows currently use cold cache initialization on every replay.
Subsequent C4 profiling observed a four-to-three-request transition that added
approximately 134 ms of target LIM and first-fill kernel work on rank 0 in one
replay. The costly first-fill behavior was present across all 16 ranks. Extra
view/transpose-copy work also averaged 3.80 ms per rank per replay. These are
profiling measurements, not speedups from an implemented fix. See the
[Nano Sparse Offload Optimization Plan](../../developer_guide/performance_and_debug/nano_sparse_offload_optimization.md)
for the evidence, inactive-row kernel changes, shared miss-buffer layout,
head-count/DP2-TP8 discussion, and validation plan.

With the updated LIM and copy-SFA pins, a clean A3 build passed four focused
native operator checks and colocated GLM-5.2 DP2/TP8/MTP3 target-graph checks,
including 131,072-token serial and concurrency-2 requests. The latest
nano-disabled PD run passed a 10,555-token prompt and two 131,072-token serial
requests. Its concurrent requests reused unsupported Decode prefixes, so that
case is excluded from supported-configuration validation. Nano-enabled PD and
the controlled 128K throughput comparison remain pending for these new pins.

When testing PD with MTP, enable matching MTP configuration on Prefill so its
draft-layer KV is registered and transferred as well as the target-model KV.
Prefill layerwise offload remains eager. Keep GLM speculative drafting eager
when testing `FULL_DECODE_ONLY` on Decode.

For repeated verification with a retained Prefill process, use a fresh Decode
`kv_port` range after restarting Decode, or restart Prefill as well. The current
producer caches its MemFabric handshake by endpoint and does not invalidate it
when a Decode process restarts at the same address and port. Reserve the full
DP-by-TP port range described above.

## 6. Limitations

- Shared-buffer Layerwise Prefill Offload requires Memcache and eager mode.
- Context parallelism has not been validated with Layerwise Prefill Offload.
- Sparse Decode Offload supports DP and TP; CP and PP are not supported.
- PD offload requires prefix caching to be disabled on Decode.
- Generalized MTP offload has focused eager and target-model
  `FULL_DECODE_ONLY` colocated validation. GLM speculative drafting remains
  eager. True PD has focused Q1 eager and matched MTP3 eager/target-graph
  validation on GLM-5.2 A3 DP1/TP16. Q1 full-decode graphs and other graph modes
  remain unsupported in this integration.
- MemFabric is the only supported `SfaRemoteD2HConnector` transfer backend.
- Layerwise buffer reuse cannot currently be combined with
  `MooncakeLayerwiseConnector` because per-buffer transfer completion gating is
  not yet implemented. Support is planned in a follow-up update.
