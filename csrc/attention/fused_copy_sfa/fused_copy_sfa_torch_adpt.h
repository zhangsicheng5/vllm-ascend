#ifndef FUSED_COPY_STA_TORCH_ADPT_H
#define FUSED_COPY_STA_TORCH_ADPT_H

namespace vllm_ascend {

// MTP0 (query_len=1) fused scatter_copy + sparse_tail_attention.
// DRAM-to-HBM KV copy and sparse+tail attention are fused into one AscendC
// launch. HBM caches are mutated in place; the caller owns attention_out.
//
// Parameters:
//   query_rope              - bf16/fp16 [B, N, 64],  read-only, attention query (rope part)
//   query                   - bf16/fp16 [B, N, 512], read-only, attention query (nope part)
//   actual_seq_lengths_query - int32 [B], read-only, TND cumulative query lengths
//   actual_seq_lengths_kv   - int32 [B], read-only, C+tail KV length per request
//   num_cache_tokens        - int32 [B], read-only, HBM cache token budget C per request
//   topk_dst_slots          - int32 [B, 1, 2048], read-only, top-2048 HBM logical slots (first miss_counts are copy dests)
//   topk_src_ids            - int32 [B, 2048], read-only, top-2048 source token IDs (first miss_counts need DRAM copy)
//   miss_counts             - int32 [B], read-only, number of tokens to copy from DRAM to HBM per request
//   hbm_block_table         - int32 [B, HBM_MAX_BLOCKS], read-only, HBM block table
//   dram_block_table        - int32 [B, DRAM_MAX_BLOCKS], read-only, DRAM block table
//   hbm_k_rope              - bf16/fp16 [HBM_BLOCKS, 128, 1, 64], read-write, HBM KV cache (rope), attention input and copy dest
//   hbm_kv_cache            - bf16/fp16 [HBM_BLOCKS, 128, 1, 512], read-write, HBM KV cache (nope), attention input and copy dest
//   dram_k_rope             - bf16/fp16 [DRAM_BLOCKS, 128, 64], read-only, host DRAM KV (rope)
//   dram_kv_cache           - bf16/fp16 [DRAM_BLOCKS, 128, 512], read-only, host DRAM KV (nope)
//   scale_value             - float, read-only, attention scale
//   attention_out           - bf16/fp16 [B, N, 512], write-only, sparse attention result
//
// Offloaded DRAM KV sources (dram_k_rope / dram_kv_cache) may reside on host
// (CPU / memfabric GVA) or NPU-tagged swapped views. Other tensors must match
// the query NPU device.
inline void
npu_fused_copy_sfa(
    const at::Tensor& query_rope,
    const at::Tensor& query,
    const at::Tensor& actual_seq_lengths_query,
    const at::Tensor& actual_seq_lengths_kv,
    const at::Tensor& num_cache_tokens,
    const at::Tensor& topk_dst_slots,
    const at::Tensor& topk_src_ids,
    const at::Tensor& miss_counts,
    const at::Tensor& hbm_block_table,
    const at::Tensor& dram_block_table,
    at::Tensor hbm_k_rope,
    at::Tensor hbm_kv_cache,
    const at::Tensor& dram_k_rope,
    const at::Tensor& dram_kv_cache,
    double scale_value,
    at::Tensor attention_out) {
  TORCH_CHECK(query.device().is_privateuseone(),
              "Fused Attention+SCATTER inputs must be on NPU.");
  TORCH_CHECK(query.dim() == 3 && query.size(0) >= 1 &&
                  query.size(1) >= 1 && query.size(1) <= 128 &&
                  query.size(2) == 512,
              "query must be [B, N, 512] with B >= 1 and "
              "1 <= N <= 128.");
  const auto batch_size = query.size(0);
  TORCH_CHECK(hbm_kv_cache.dim() == 4 &&
                  hbm_kv_cache.size(1) == 128 &&
                  hbm_kv_cache.size(2) == 1 &&
                  hbm_kv_cache.size(3) == 512,
              "HBM CKV must be [blocks, 128, 1, 512].");
  TORCH_CHECK(hbm_k_rope.dim() == 4 &&
                  hbm_k_rope.size(0) == hbm_kv_cache.size(0) &&
                  hbm_k_rope.size(1) == 128 &&
                  hbm_k_rope.size(2) == 1 &&
                  hbm_k_rope.size(3) == 64,
              "HBM KPE must be [blocks, 128, 1, 64].");
  TORCH_CHECK(dram_kv_cache.dim() == 3 &&
                  dram_kv_cache.size(1) == 128 &&
                  dram_kv_cache.size(2) == 512,
              "DRAM CKV must be [blocks, 128, 512].");
  TORCH_CHECK(dram_k_rope.dim() == 3 &&
                  dram_k_rope.size(0) == dram_kv_cache.size(0) &&
                  dram_k_rope.size(1) == 128 &&
                  dram_k_rope.size(2) == 64,
              "DRAM KPE must be [blocks, 128, 64].");
  TORCH_CHECK(topk_dst_slots.dim() == 3 &&
                  topk_dst_slots.size(0) == batch_size &&
                  topk_dst_slots.size(1) == 1 &&
                  topk_dst_slots.size(2) == 2048,
              "sparse_slots must be [B, 1, 2048].");
  TORCH_CHECK(num_cache_tokens.dim() == 1 &&
                  num_cache_tokens.size(0) == batch_size &&
                  hbm_block_table.dim() == 2 &&
                  hbm_block_table.size(0) == batch_size &&
                  actual_seq_lengths_query.dim() == 1 &&
                  actual_seq_lengths_query.size(0) == batch_size &&
                  actual_seq_lengths_kv.dim() == 1 &&
                  actual_seq_lengths_kv.size(0) == batch_size &&
                  dram_block_table.dim() == 2 &&
                  dram_block_table.size(0) == batch_size &&
                  topk_src_ids.dim() == 2 &&
                  topk_src_ids.size(0) == batch_size &&
                  topk_src_ids.size(1) == 2048 &&
                  miss_counts.dim() == 1 &&
                  miss_counts.size(0) == batch_size,
              "Fused Attention+SCATTER metadata shapes are invalid.");
  TORCH_CHECK(query_rope.dim() == 3 &&
                  query_rope.size(0) == batch_size &&
                  query_rope.size(1) == query.size(1) &&
                  query_rope.size(2) == 64,
              "query_rope must be [B, N, 64].");
  TORCH_CHECK(attention_out.sizes() == query.sizes(),
              "attention_out must have the query shape.");

  const auto dtype = query.scalar_type();
  TORCH_CHECK(dtype == at::kHalf || dtype == at::kBFloat16,
              "Fused Attention+SCATTER supports fp16 or bf16.");
  for (const at::Tensor* tensor :
       std::array<const at::Tensor*, 6>{
           &hbm_kv_cache, &query_rope, &hbm_k_rope, &dram_k_rope,
           &dram_kv_cache, &attention_out}) {
    TORCH_CHECK(tensor->scalar_type() == dtype,
                "All fused floating-point inputs must share one dtype.");
  }
  for (const at::Tensor* tensor :
       std::array<const at::Tensor*, 8>{
           &topk_dst_slots, &num_cache_tokens, &hbm_block_table,
           &actual_seq_lengths_query, &actual_seq_lengths_kv,
           &dram_block_table, &topk_src_ids, &miss_counts}) {
    TORCH_CHECK(tensor->scalar_type() == at::kInt,
                "All fused metadata inputs must be int32.");
  }
  const auto device = query.device();
  // Offloaded DRAM KV may reside on host or NPU; everything else must match query.
  for (const at::Tensor* tensor :
       std::array<const at::Tensor*, 13>{
           &query, &hbm_kv_cache, &topk_dst_slots, &num_cache_tokens,
           &hbm_block_table, &actual_seq_lengths_query,
           &actual_seq_lengths_kv, &query_rope, &hbm_k_rope,
           &dram_block_table, &topk_src_ids, &miss_counts, &attention_out}) {
    TORCH_CHECK(tensor->device() == device,
                "All non-DRAM fused inputs must be on the same NPU.");
    TORCH_CHECK(tensor->is_contiguous(),
                "All fused inputs must be contiguous.");
  }
  TORCH_CHECK(dram_k_rope.is_contiguous() && dram_kv_cache.is_contiguous(),
              "DRAM fused inputs must be contiguous.");

  std::string query_layout = "TND";
  std::string kv_layout = "PA_BSND";
  char* query_layout_ptr = const_cast<char*>(query_layout.c_str());
  char* kv_layout_ptr = const_cast<char*>(kv_layout.c_str());
  constexpr int64_t kSparseBlockSize = 1;
  constexpr int64_t kSparseMode = 3;
  auto keepalive = std::make_tuple(
      query_rope, query, actual_seq_lengths_query, actual_seq_lengths_kv,
      num_cache_tokens, topk_dst_slots, topk_src_ids, miss_counts,
      hbm_block_table, dram_block_table,
      hbm_k_rope, dram_k_rope, dram_kv_cache,
      hbm_kv_cache, attention_out);
  EXEC_NPU_CMD_ORDERED(
      aclnnFusedCopySfa,
      keepalive,
      query,
      hbm_kv_cache,
      hbm_kv_cache,
      topk_dst_slots,
      num_cache_tokens,
      hbm_block_table,
      actual_seq_lengths_query,
      actual_seq_lengths_kv,
      query_rope,
      hbm_k_rope,
      dram_k_rope,
      dram_kv_cache,
      dram_block_table,
      topk_src_ids,
      miss_counts,
      scale_value,
      kSparseBlockSize,
      query_layout_ptr,
      kv_layout_ptr,
      kSparseMode,
      attention_out);
}

}  // namespace vllm_ascend

#endif
