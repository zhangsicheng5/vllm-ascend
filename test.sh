llm = LLM(
    model=model_path,
    # trust_remote_code=True,
    enforce_eager=True,
    # compilation_config={"cudagraph_mode":"FULL_DECODE_ONLY"},
    # compilation_config={"cudagraph_mode":"PIECEWISE"},
    # compilation_config={"cudagraph_mode":"FULL_DECODE_ONLY", "cudagraph_capture_sizes":[1, 2, 4]},
    tensor_parallel_size=16,
    # decode_context_parallel_size=2,
    # prefill_context_parallel_size=2,
    # cp_kv_cache_interleave_size=128,
    enable_expert_parallel=True,
    # enable_chunked_prefill=False,
    # enable_prefix_caching=False,
    gpu_memory_utilization=0.9,
    # gpu_memory_utilization=0.76, # tp1, 6 layers, 32k, uti 0.76, bs 3
    # gpu_memory_utilization=0.862, # tp2, 10 layers, 32k, uti 0.86, bs 1
    # gpu_memory_utilization=0.84, # tp2, 10 layers, 32k, graph, uti 0.84, bs 1
    quantization="ascend",
    max_num_seqs=4,
    max_model_len=4096,
    max_num_batched_tokens=4096,
    # max_model_len=33792,
    # max_num_batched_tokens=33792,
    additional_config={
        "use_offload": True,
    },
    # speculative_config={
    #     "num_speculative_tokens": 1,
    #     "method": "deepseek_mtp"
    # },
    kv_transfer_config = {
        "kv_connector": "AscendStoreConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
        "backend": "memcache",
        "mooncake_rpc_port":"0",
        "use_layerwise": True,
        "discard_partial_chunks": False,
        }
    },
    # block_size=1024,
    async_scheduling=False,
    disable_hybrid_kv_cache_manager=False,
    profiler_config={
        "profiler": "torch",
        # "torch_profiler_dir": "/home/z00911889/profile/v32_l10_tp2_baseline_bs4_seq256_graph",
        "torch_profiler_dir": "/home/z00911889/profile/test",
        "torch_profiler_with_stack": True,
    }
)