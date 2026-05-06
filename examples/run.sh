rm -rf /root/ascend/log/*
echo "/tmp/core.%p" | tee /proc/sys/kernel/core_pattern
# export PYTHONPATH=$PYTHONPATH:/home/t00612968/vllm
# export PYTHONPATH=$PYTHONPATH:/home/t00612968/vllm-ascend
export HCCL_OP_EXPANSION_MODE="AIV";
export OMP_PROC_BIND=false;
export OMP_NUM_THREADS=1;
export VLLM_USE_V1=1;
export HCCL_BUFFSIZE=200;
export VLLM_ASCEND_ENABLE_MLAPO=1;
export VLLM_RPC_TIMEOUT=3600000;
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600000;
export VLLM_ASCEND_ENABLE_FLASHCOMM1=0;
export DISABLE_L2_CACHE=1;
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
# export ASCEND_BUFFER_POOL=4:8;
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH;
export ASCEND_CONNECT_TIMEOUT=10000
export ASCEND_TRANSFER_TIMEOUT=10000
export TASK_QUEUE_ENABLE=1
export VLLM_ASCEND_ENABLE_NZ=1
export ASCEND_LAUNCH_BLOCKING=1
source /usr/local/memcache_hybrid/set_env.sh
source /usr/local/memfabric_hybrid/set_env.sh
# 配置文件环境变量
export MMC_LOCAL_CONFIG_PATH=/home/l00948936/code/mmc-local.conf
export VLLM_TORCH_PROFILER_DIR="./vllm-profiling"
export VLLM_TORCH_PROFILER_WITH_STACK=0
# layerwise新增变量
export NUM_REUSE_LAYERS=4
export NUM_LAYERS=61 # GLM-5
export VLLM_VERSION=0.19.0

vllm serve /home/weights/deepseek-ai/DeepSeek-V3.2-Exp-W8A8 \
    --host 0.0.0.0 \
    --port 8004 \
    --tensor-parallel-size 16 \
    --enforce-eager \
    --seed 1024 \
    --quantization ascend \
    --served-model-name ds3.2 \
    --enable-expert-parallel \
    --max-num-seqs 8 \
    --max-model-len 32769 \
    --max-num-batched-tokens 32769 \
    --trust-remote-code \
    --no-enable-chunked-prefill \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.85 \
    --async-scheduling \
    --additional-config '{"fuse_muls_add": true, "multistream_overlap_shared_expert": true, "ascend_compilation_config": {"enable_npugraph_ex": true}}' \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --kv-transfer-config \
    '{"kv_connector": "AscendStoreConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {"backend": "memcache","use_layerwise": true,"mooncake_rpc_port":"0"}}'
        # > log_p.log 2>&1