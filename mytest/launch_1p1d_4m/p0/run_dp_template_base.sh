#!/bin/bash
# Online-DP form of ../../prefill/prefill.sh. Arguments are supplied by
# launch_online_dp.py: devices, HTTP port, DP size/rank/address/RPC port, TP size.

set -euo pipefail

if [[ $# -ne 7 ]]; then
  echo "usage: $0 VISIBLE_DEVICES HTTP_PORT DP_SIZE DP_RANK DP_ADDRESS DP_RPC_PORT TP_SIZE" >&2
  exit 2
fi

VISIBLE_DEVICES="$1"
SERVE_PORT="$2"
DP_SIZE="$3"
DP_RANK="$4"
DP_ADDRESS="$5"
DP_RPC_PORT="$6"
TP_SIZE="$7"

export http_proxy=""
export https_proxy=""
export no_proxy="localhost,127.0.0.1"

# source /usr/local/Ascend/ascend-toolkit/set_env.sh
# source /usr/local/memcache_hybrid/set_env.sh
# source /usr/local/memfabric_hybrid/set_env.sh
export PYTHONHASHSEED=0
export MMC_META_CONFIG_PATH=/mnt/share_space/l00933205/scripts/1p1d/p0/mmc-meta.conf
export MMC_LOCAL_CONFIG_PATH=/mnt/share_space/l00933205/scripts/1p1d/p0/mmc-local.conf

export ASCEND_CUSTOM_OPP_PATH=/home/l00948936/code/fuse_op/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer:${ASCEND_CUSTOM_OPP_PATH:-}
export LD_LIBRARY_PATH=/home/l00948936/code/fuse_op/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_api/lib/:${LD_LIBRARY_PATH:-}
export LD_LIBRARY_PATH=/usr/local/python3.12.13/lib:$LD_LIBRARY_PATH
# export PYTHONPATH=/mnt/share_space/l00933205/code/mte_fused/vllm-ascend:${PYTHONPATH:-}
# export PYTHONPATH=/mnt/share_space/l00933205/code/mte_fused/vllm:$PYTHONPATH

# MODEL_PATH="/mnt/weight/GLM-5.2-W4A8-0628"
MODEL_PATH="/mnt/share_space/z_offload/data/model_from_hf/GLM-5.2-W4A8C8-0713-MTP"
SERVE_HOST="10.246.63.45" # "10.246.63.46"
NET_IFACE="enp162s0f0"
KV_PORT=21070
KV_RANK=0

export VLLM_VERSION=0.29.0
export ASCEND_RT_VISIBLE_DEVICES="$VISIBLE_DEVICES"
export VLLM_ASCEND_ENABLE_NZ=1
export HCCL_OP_EXPANSION_MODE="AIV"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export VLLM_USE_V1=1
export HCCL_BUFFSIZE=1024
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_SERVER_DEV_MODE=1

ADDITIONAL_CONFIG='{"enable_dsa_cp": true, "enable_flashcomm1": true, "enable_sparse_li_c8": false}'
# export VLLM_ASCEND_KV_TRANSFER_BACKEND="memfabric"
# export VLLM_ASCEND_MF_VERIFY="0"
# export VLLM_ASCEND_SFA_DEBUG="0"
# export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE=1
# export VLLM_USE_FASTOKENS=1

export HCCL_IF_IP="$SERVE_HOST"
export GLOO_SOCKET_IFNAME="$NET_IFACE"
export TP_SOCKET_IFNAME="$NET_IFACE"
export HCCL_SOCKET_IFNAME="$NET_IFACE"

dp_args=()
if (( DP_SIZE > 1 )); then
  dp_args+=(
    --data-parallel-size "$DP_SIZE"
    --data-parallel-rank "$DP_RANK"
    --data-parallel-address "$DP_ADDRESS"
    --data-parallel-rpc-port "$DP_RPC_PORT"
  )
fi

exec vllm serve "$MODEL_PATH" \
  --host "$SERVE_HOST" \
  --port "$SERVE_PORT" \
  --served-model-name model \
  "${dp_args[@]}" \
  --tensor-parallel-size "$TP_SIZE" \
  --enable-expert-parallel \
  --max-model-len 35000 \
  --max-num-seqs 16 \
  --api-server-count 16 \
  --max-num-batched-tokens 8192 \
  --trust-remote-code \
  --enforce-eager \
  --quantization ascend \
  --gpu-memory-utilization 0.9 \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 1, "enforce_eager": true}' \
  --safetensors-load-strategy prefetch \
  --hf-overrides '{"use_index_cache": true}' \
  --additional-config "$ADDITIONAL_CONFIG" \
  --profiler_config '{"profiler":"torch", "torch_profiler_dir":"/home/z_offload/profile/test", "torch_profiler_with_stack":true, "torch_profiler_with_memory":false, "torch_profiler_record_shapes":true}' \
  --kv-transfer-config '{
        "kv_connector": "MooncakeConnectorV1",
        "kv_role": "kv_producer",
        "kv_port": "30000",
        "engine_id": "0",
        "kv_connector_extra_config": {
            "use_ascend_direct": true,
            "prefill": {"dp_size": 1, "tp_size": 16},
            "decode": { "dp_size": 8, "tp_size": 2}
        }
    }'

# --kv-transfer-config \
#     '{
#     "kv_connector": "MultiConnector",
#     "kv_role": "kv_producer",
#     "kv_connector_extra_config": {
#         "layerwise_num_shared_buffers":"3",
#         "layerwise_prefetch_layers":"3",
#         "layerwise_independent_layers":0,
#         "connectors": [
#             {
#                 "kv_connector": "SfaRemoteD2HConnector",
# 		        "kv_role": "kv_producer",
#                 "kv_connector_extra_config": {"transfer_backend": "memfabric", "use_layerwise": true}
#             }
#         ]
#     }
#     }'


