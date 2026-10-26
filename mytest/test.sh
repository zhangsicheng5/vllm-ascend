# source /home/cann/rc1b070/ascend-toolkit/set_env.sh
# source /home/cann/rc1b070/nnal/atb/set_env.sh
nic_name="enp162s0f0"
local_ip=10.246.63.45
export HCCL_IF_IP=$local_ip
export HCCL_IF_BASE_PORT=50000
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name

export HCCL_BUFFSIZE=768
export VLLM_ASCEND_SFA_DEBUG=1
rm -rf ~/atc_data

# export ASCEND_LAUNCH_BLOCKING=1
# export VLLM_USE_V1=1
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
# export VLLM_LOGGING_LEVEL=DEBUG

export OMP_PROC_BIND=false
export OMP_NUM_THREADS=100

# kv offload
# source /usr/local/memfabric_hybrid/set_env.sh
# source /usr/local/memcache_hybrid/set_env.sh
# export LD_LIBRARY_PATH=/usr/local/python3.11.10/lib/python3.11/site-packages/memfabric_hybrid/lib/:$LD_LIBRARY_PATH
# export MMC_META_CONFIG_PATH=/usr/local/memcache_hybrid/latest/config/mmc-meta.conf
# export MMC_LOCAL_CONFIG_PATH=/usr/local/memcache_hybrid/latest/config/mmc-local.conf
# export MEMFABRIC_HYBRID_EXTEND_LIB_PATH=/usr/local/memfabric_hybrid/1.1.2/aarch64-linux/lib64
# export MEMFABRIC_HYBRID_EXTEND_LIB_PATH=/usr/local/memfabric_hybrid/1.2.0/aarch64-linux/lib64

# export VLLM_TORCH_PROFILER_DIR=/home/z00911889/profile/sparse_offload
# export VLLM_TORCH_PROFILER_WITH_STACK=1

# export ASCEND_RT_VISIBLE_DEVICES=2,3
# export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
# export ASCEND_RT_VISIBLE_DEVICES=6,7
# export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
# export ASCEND_RT_VISIBLE_DEVICES=10,11
# export ASCEND_RT_VISIBLE_DEVICES=14,15

# rm -rf ~/atc_data
rm -rf .torchair_cache/
rm -rf /root/.cache/vllm/torch_compile_cache/*
# rm -rf /root/.cache/torch_extensions/py311_cpu/cpu_sparse_attn

# python vllm_test.py 2>&1 | tee logs/t.log
python vllm_test_glm52.py 2>&1 | tee logs/t.log

# python -m debugpy --listen 0.0.0.0:56301 --wait-for-client vllm_test.py 2>&1 | tee logs/t.log
# python -m debugpy --listen 0.0.0.0:56301 --wait-for-client vllm_test_glm52.py 2>&1 | tee logs/t.log
