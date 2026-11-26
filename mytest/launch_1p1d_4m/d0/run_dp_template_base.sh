local_ip=`hostname -I | awk '{print $1}'`
nic_name=$(ifconfig -a | grep -B1 $local_ip | grep -v 'inet' | sed 's/://g' | awk '{print $1}')

export HCCL_OP_EXPANSION_MODE=""AIV""

export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name

#Mooncake
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1

export VLLM_VERSION=0.29.0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=256
export ACL_OP_INIT_MODE=1
export ASCEND_A3_ENABLE=1
export TASK_QUEUE_ENABLE=1
export ASCEND_RT_VISIBLE_DEVICES=$1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
export VLLM_ASCEND_ENABLE_FUSED_MC2=1

export MMC_LOCAL_CONFIG_PATH=/usr/local/python3.12.13/lib/python3.12/site-packages/memcache_hybrid/config/mmc-local.conf
export PYTHONHASHSEED=0

vllm serve /mnt/share_space/z_offload/data/model_from_hf/GLM-5.2-W4A8C8-0713-MTP \
    --host 0.0.0.0 \
    --port $2 \
    --data-parallel-size $3 \
    --data-parallel-rank $4 \
    --data-parallel-address $5 \
    --data-parallel-rpc-port $6 \
    --tensor-parallel-size $7 \
    --enable-expert-parallel \
    --seed 1024 \
    --served-model-name model \
    --max-model-len 201000 \
    --max-num-batched-tokens 4 \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY", "cudagraph_capture_sizes":[4]}' \
    --speculative-config '{"num_speculative_tokens": 3,  "method":"deepseek_mtp","enforce_eager":true}' \
    --additional-config '{"enable_fused_mc2": 1, "ascend_compilation_config":{"enable_npugraph_ex": true, "enable_static_kernel": false}}' \
    --trust-remote-code \
    --max-num-seqs 1 \
    --gpu-memory-utilization 0.92 \
    --async-scheduling \
    --quantization ascend \
    --enable-auto-tool-choice \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --kv-transfer-config \
    '{
    "kv_connector": "MooncakeConnectorV1",
    "kv_role": "kv_consumer",
    "kv_port": "30200",
    "engine_id": "1",
    "kv_connector_extra_config": {
        "use_ascend_direct": true,
        "prefill": {"dp_size": 1, "tp_size": 16},
        "decode": {"dp_size": 8, "tp_size": 2}
      }
    }'