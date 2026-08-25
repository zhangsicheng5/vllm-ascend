#ifndef FUSED_COPY_STA_MTP_TORCH_ADPT_H
#define FUSED_COPY_STA_MTP_TORCH_ADPT_H

namespace vllm_ascend {

// MTP3 (query_len=4) fused source-aware DRAM-to-HBM copy + causal sparse
// attention in one AscendC launch. HBM caches are mutated in place and the
// caller owns attention_out; no cache aliases are exposed through the public
// torch.library schema.
//
// Parameters:
//   query_rope              - bf16/fp16 [B*4, N, 64],  read-only, 4-way attention query (rope part)
//   query                   - bf16/fp16 [B*4, N, 512], read-only, 4-way attention query (nope part)
//   actual_seq_lengths_query - int32 [B], read-only, TND cumulative query lengths, values are [4, 8, ..., B*4]
//   actual_seq_lengths_kv   - int32 [B], read-only, final KV length for the 4th query per request
//   num_cache_tokens        - int32 [B], read-only, HBM cache token budget C per request
//   topk_dst_slots          - int32 [B*4, 1, 2048], read-only, 4-way top-2048 HBM logical slots
//   topk_src_ids            - int32 [B*4, 1, 2048], read-only, 4-way top-2048 source token IDs (HBM hit positions are -1)
//   miss_src_ids            - int32 [B, 8192], read-only, unique miss source IDs in the 4-way union (first miss_counts valid)
//   miss_dst_slots          - int32 [B, 8192], read-only, unique miss HBM logical slots (first miss_counts valid)
//   miss_counts             - int32 [B], read-only, unique union miss token count per request
//   hbm_block_table         - int32 [B, HBM_MAX_BLOCKS], read-only, HBM block table
//   dram_block_table        - int32 [B, DRAM_MAX_BLOCKS], read-only, DRAM block table
//   hbm_k_rope              - bf16/fp16 [HBM_BLOCKS, 128, 1, 64], read-write, HBM KV cache (rope), attention input and copy dest
//   hbm_kv_cache            - bf16/fp16 [HBM_BLOCKS, 128, 1, 512], read-write, HBM KV cache (nope), attention input and copy dest
//   dram_k_rope             - bf16/fp16 [DRAM_BLOCKS, 128, 64], read-only, host DRAM KV (rope)
//   dram_kv_cache           - bf16/fp16 [DRAM_BLOCKS, 128, 512], read-only, host DRAM KV (nope)
//   scale_value             - float, read-only, attention scale
//   attention_out           - bf16/fp16 [B*4, N, 512], write-only, 4-way causal sparse attention result
//
// Offloaded DRAM KV sources (dram_k_rope / dram_kv_cache) may reside on host
// or on the query NPU device. All other tensors must be on the query NPU.
inline void npu_fused_copy_sfa_mtp(
    const at::Tensor& query_rope,
    const at::Tensor& query,
    const at::Tensor& actual_seq_lengths_query,
    const at::Tensor& actual_seq_lengths_kv,
    const at::Tensor& num_cache_tokens,
    const at::Tensor& topk_dst_slots,
    const at::Tensor& topk_src_ids,
    const at::Tensor& miss_src_ids,
    const at::Tensor& miss_dst_slots,
    const at::Tensor& miss_counts,
    const at::Tensor& hbm_block_table,
    const at::Tensor& dram_block_table,
    at::Tensor hbm_k_rope,
    at::Tensor hbm_kv_cache,
    const at::Tensor& dram_k_rope,
    const at::Tensor& dram_kv_cache,
    double scale_value,
    at::Tensor attention_out) {
  constexpr int64_t kBlockSize = 128;
  constexpr int64_t kQueryCount = 4;
  constexpr int64_t kTopK = 2048;
  constexpr int64_t kCkvDim = 512;
  constexpr int64_t kKpeDim = 64;

  TORCH_CHECK(query.device().is_privateuseone(),
              "Fused MTP copy+Attention inputs must be on NPU.");
  TORCH_CHECK(query.dim() == 3 && query.size(2) == kCkvDim,
              "Fused MTP query must be [4B, N, 512].");
  TORCH_CHECK(num_cache_tokens.dim() == 1 && num_cache_tokens.size(0) > 0,
              "Fused MTP cache_tokens must be [B] with B > 0.");
  const int64_t batch_size = num_cache_tokens.size(0);
  TORCH_CHECK(query.size(0) == batch_size * kQueryCount,
              "Fused MTP copy+Attention requires four queries per request.");
  TORCH_CHECK(query_rope.dim() == 3 &&
                  query_rope.size(0) == query.size(0) &&
                  query_rope.size(1) == query.size(1) &&
                  query_rope.size(2) == kKpeDim,
              "Fused MTP query_rope must be [4B, N, 64].");
  TORCH_CHECK(topk_dst_slots.dim() == 3 &&
                  topk_dst_slots.size(0) == query.size(0) &&
                  topk_dst_slots.size(1) == 1 &&
                  topk_dst_slots.size(2) == kTopK,
              "Fused MTP topk_slots must be [4B, 1, 2048].");
  TORCH_CHECK(topk_src_ids.sizes() == topk_dst_slots.sizes(),
              "Fused MTP topk_source_ids must match topk_slots.");
  TORCH_CHECK(miss_src_ids.dim() == 2 &&
                  miss_src_ids.size(0) == batch_size &&
                  miss_src_ids.size(1) == kQueryCount * kTopK &&
                  miss_dst_slots.sizes() == miss_src_ids.sizes(),
              "Fused MTP compact union misses must be [B, 8192].");
  TORCH_CHECK(miss_counts.dim() == 1 &&
                  miss_counts.size(0) == batch_size,
              "Fused MTP miss_counts must be [B].");
  TORCH_CHECK(hbm_block_table.dim() == 2 &&
                  hbm_block_table.size(0) == batch_size &&
                  hbm_block_table.size(1) > 0 &&
                  dram_block_table.dim() == 2 &&
                  dram_block_table.size(0) == batch_size &&
                  dram_block_table.size(1) > 0,
              "Fused MTP block tables must be [B, max_blocks].");
  TORCH_CHECK(actual_seq_lengths_query.dim() == 1 &&
                  actual_seq_lengths_query.size(0) == batch_size &&
                  actual_seq_lengths_kv.dim() == 1 &&
                  actual_seq_lengths_kv.size(0) == batch_size,
              "Fused MTP actual sequence lengths must be [B].");

  TORCH_CHECK(hbm_k_rope.dim() == 4 &&
                  hbm_k_rope.size(1) == kBlockSize &&
                  hbm_k_rope.size(2) == 1 &&
                  hbm_k_rope.size(3) == kKpeDim,
              "Fused MTP HBM KPE must be [blocks, 128, 1, 64].");
  TORCH_CHECK(hbm_kv_cache.dim() == 4 &&
                  hbm_kv_cache.size(0) == hbm_k_rope.size(0) &&
                  hbm_kv_cache.size(1) == kBlockSize &&
                  hbm_kv_cache.size(2) == 1 &&
                  hbm_kv_cache.size(3) == kCkvDim,
              "Fused MTP HBM CKV must be [blocks, 128, 1, 512].");
  TORCH_CHECK(dram_k_rope.dim() == 3 &&
                  dram_k_rope.size(1) == kBlockSize &&
                  dram_k_rope.size(2) == kKpeDim,
              "Fused MTP DRAM KPE must be [blocks, 128, 64].");
  TORCH_CHECK(dram_kv_cache.dim() == 3 &&
                  dram_kv_cache.size(0) == dram_k_rope.size(0) &&
                  dram_kv_cache.size(1) == kBlockSize &&
                  dram_kv_cache.size(2) == kCkvDim,
              "Fused MTP DRAM CKV must be [blocks, 128, 512].");
  TORCH_CHECK(attention_out.sizes() == query.sizes(),
              "Fused MTP attention_out must have the query shape.");

  const auto dtype = query.scalar_type();
  TORCH_CHECK(dtype == at::kHalf || dtype == at::kBFloat16,
              "Fused MTP copy+Attention supports fp16 or bf16.");
  for (const at::Tensor* tensor :
       std::array<const at::Tensor*, 7>{
           &query_rope, &hbm_k_rope, &hbm_kv_cache, &dram_k_rope,
           &dram_kv_cache, &attention_out, &query}) {
    TORCH_CHECK(tensor->scalar_type() == dtype,
                "All fused MTP floating-point tensors must share one dtype.");
  }
  for (const at::Tensor* tensor :
       std::array<const at::Tensor*, 10>{
           &actual_seq_lengths_query, &actual_seq_lengths_kv,
           &num_cache_tokens, &topk_dst_slots, &topk_src_ids,
           &miss_src_ids, &miss_dst_slots, &miss_counts,
           &hbm_block_table,
           &dram_block_table}) {
    TORCH_CHECK(tensor->scalar_type() == at::kInt,
                "All fused MTP metadata tensors must be int32.");
  }

  const auto device = query.device();
  // Offloaded DRAM KV may reside on host or NPU; everything else must match query.
  for (const at::Tensor* tensor :
       std::array<const at::Tensor*, 15>{
           &query_rope, &query, &actual_seq_lengths_query,
           &actual_seq_lengths_kv, &num_cache_tokens, &topk_dst_slots,
           &topk_src_ids, &miss_src_ids, &miss_dst_slots, &miss_counts,
           &hbm_block_table, &dram_block_table, &hbm_k_rope,
           &hbm_kv_cache, &attention_out}) {
    TORCH_CHECK(tensor->device() == device,
                "All non-DRAM fused MTP tensors must be on the same NPU.");
    TORCH_CHECK(tensor->is_contiguous(),
                "All fused MTP tensors must be contiguous.");
  }
  TORCH_CHECK(dram_k_rope.is_contiguous() && dram_kv_cache.is_contiguous(),
              "DRAM fused MTP tensors must be contiguous.");

  std::string query_layout = "TND";
  std::string kv_layout = "PA_BSND";
  char* query_layout_ptr = const_cast<char*>(query_layout.c_str());
  char* kv_layout_ptr = const_cast<char*>(kv_layout.c_str());
  constexpr int64_t kSparseBlockSize = 1;
  constexpr int64_t kSparseMode = 3;
  auto keepalive = std::make_tuple(
      query_rope, query, actual_seq_lengths_query,
      actual_seq_lengths_kv, num_cache_tokens, topk_dst_slots,
      topk_src_ids, miss_src_ids, miss_dst_slots, miss_counts,
      hbm_block_table, dram_block_table, hbm_k_rope,
      hbm_kv_cache, dram_k_rope, dram_kv_cache, attention_out);
  EXEC_NPU_CMD_ORDERED(
      aclnnFusedCopySfaMtp,
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
      miss_src_ids,
      miss_dst_slots,
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
