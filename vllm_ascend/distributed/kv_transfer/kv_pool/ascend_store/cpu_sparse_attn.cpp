#include <torch/extension.h>
#include <limits>
#include <optional>
#include <iostream>
#include <chrono>
#include <string>
#include <stdexcept>
#include <vector>
#include <ATen/Parallel.h>
#include <torch/script.h>

#include <acl/acl.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <omp.h>
#include <assert.h>
#include <torch/torch.h>
#include <pthread.h>


void bindToCore(int core_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);

    pthread_t current_thread = pthread_self();
    int rc = pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
    if (rc != 0) {
        std::cerr << "Error calling pthread_setaffinity_np: " << rc << "\n";
    }
}

void bindToCore(std::vector<int> core_ids) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    for(int core_id : core_ids) {
        CPU_SET(core_id, &cpuset);
    }

    pthread_t current_thread = pthread_self();
    int rc = pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
    if (rc != 0) {
        std::cerr << "Error calling pthread_setaffinity_np: " << rc << "\n";
    }
}

void print_current_thread_affinity() {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);

    // Get affinity of the current calling thread
    pthread_t current_thread = pthread_self();
    int result = pthread_getaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
    
    if (result != 0) {
        std::cerr << "Error getting affinity mask" << std::endl;
        return;
    }

    // Print out which CPUs are available to this thread
    std::cout << "Thread is allowed to run on CPUs: ";
    for (int i = 0; i < CPU_SETSIZE; i++) {
        if (CPU_ISSET(i, &cpuset)) {
            std::cout << i << " ";
        }
    }
    std::cout << std::endl;
}

at::Tensor restore_tensor(uintptr_t ptr_val, const std::vector<int64_t>& shape, torch::ScalarType dtype = torch::kBFloat16) {
    if (ptr_val == 0) return at::Tensor(); // 处理空指针情况
    auto options = torch::TensorOptions().dtype(dtype).device(torch::kCPU);
    return torch::from_blob(reinterpret_cast<void*>(ptr_val), shape, options);
}

void
get_kv_topk_kernel(
    const at::Tensor& k_nope_h, // [block_num, block_size, head_kv, nope_dim]
    const at::Tensor& k_pe_h, // [block_num, block_size, head_kv, rope_dim]
    const at::Tensor& topk_indices, // [num_token, 1, topk]
    const at::Tensor& actual_seq_lengths_query, // [num_reqs], == num_token if without spec decode
    const at::Tensor& block_table, // [num_reqs, max_block_num]
    const int& thread_num,
    at::Tensor& k_nope_topk,
    at::Tensor& k_pe_topk
) {
    auto k_nope = k_nope_h.contiguous();
    auto k_pe = k_pe_h.contiguous();

    TORCH_CHECK(topk_indices.size(1) == 1, "topk_indices must have shape [N, 1, S]");
    auto topk_indices_squeezed = topk_indices.squeeze(1); // [N, S]

    int32_t token_num = topk_indices.size(0);  // Batch size
    int32_t pe_dim = k_pe.size(3);    // Embedding dimension of pe
    int32_t num_page = k_nope.size(0);    // Number of pages
    int32_t page_size = k_nope.size(1);   // Page size
    int32_t head_kv = k_nope.size(2);      // Number of heads of key/value
    int32_t value_dim = k_nope.size(3); // head dim of k_nope & value
    int32_t sqen_len = topk_indices_squeezed.size(1);  // Source sequence length (TopK)
    int32_t max_num_pages = block_table.size(1);  // Max number of pages per request
    int32_t num_requests = actual_seq_lengths_query.size(0);

    // auto cpu_pinned_opt = k_nope_h.options()
    //     .device(at::kCPU)         // 限定CPU设备
    //     .pinned_memory(true);     // 开启页锁定内存
    // at::Tensor k_nope_cpu_pinned = at::empty({token_num, sqen_len, head_kv, value_dim}, cpu_pinned_opt);
    // at::Tensor k_pe_cpu_pinned = at::empty({token_num, sqen_len, head_kv, pe_dim}, cpu_pinned_opt);

    bfloat16_t* k_nope_ptr = static_cast<bfloat16_t*>(k_nope.data_ptr());
    bfloat16_t* k_pe_ptr = static_cast<bfloat16_t*>(k_pe.data_ptr());
    int32_t* topk_indices_ptr = static_cast<int32_t*>(topk_indices_squeezed.data_ptr());
    int32_t* actual_seq_lengths_query_ptr = static_cast<int32_t*>(actual_seq_lengths_query.data_ptr());
    int32_t* block_table_ptr = static_cast<int32_t*>(block_table.data_ptr());

    // bfloat16_t* k_nope_pinned_ptr = static_cast<bfloat16_t*>(k_nope_cpu_pinned.data_ptr());
    // bfloat16_t* k_pe_pinned_ptr = static_cast<bfloat16_t*>(k_pe_cpu_pinned.data_ptr());
    bfloat16_t* k_nope_pinned_ptr = static_cast<bfloat16_t*>(k_nope_topk.data_ptr());
    bfloat16_t* k_pe_pinned_ptr = static_cast<bfloat16_t*>(k_pe_topk.data_ptr());

    auto req_ids = at::empty({token_num}, at::kInt);   // at::empty，分配一块指定形状和数据类型的内存空间，创建一维张量，int32类型
    auto page_indices = at::empty({token_num, sqen_len}, at::kInt);   //创建二维张量
    auto pos_in_page = at::empty({token_num, sqen_len}, at::kInt);

    int32_t* req_ids_ptr = req_ids.data_ptr<int32_t>();
    int32_t* page_indices_ptr = page_indices.data_ptr<int32_t>();
    int32_t* pos_in_page_ptr = pos_in_page.data_ptr<int32_t>();

    for (int32_t r = 0; r < num_requests; ++r) {
        int32_t start = (r == 0) ? 0 : actual_seq_lengths_query_ptr[r - 1];
        int32_t end = actual_seq_lengths_query_ptr[r];

        std::fill(req_ids_ptr + start, req_ids_ptr + end, r);
    }

    // multi thread related. Currently parallel in seq_len(topk) dim.
    // int thread_num = 16;
    int size_per_thread = sqen_len / thread_num;
    std::vector<int> thread_size(thread_num, size_per_thread);
    thread_size[thread_num - 1] += sqen_len - size_per_thread * thread_num;
    std::vector<int> thread_start(thread_num);
    int cumsum = 0;
    for(int i = 0; i < thread_num; ++i) {
        thread_start[i] = cumsum;
        cumsum += thread_size[i];
    }
    // std::cout << "thread start: "; for(auto i : thread_start) std::cout << i << " "; std::cout << std::endl;
    // std::cout << "thread size: "; for(auto i : thread_size) std::cout << i << " "; std::cout << std::endl;

    at::set_num_threads(thread_num);
    omp_set_num_threads(thread_num);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int t_start = thread_start[tid];
        int t_end = t_start + thread_size[tid];
        for (int32_t i = 0; i < token_num; ++i) {
            int32_t req_id = req_ids_ptr[i];
            // for (int32_t s = 0; s < sqen_len; ++s) {
            for (int32_t s = t_start; s < t_end; ++s) {
                int32_t global_token_pos = topk_indices_ptr[i * sqen_len + s];

                if (global_token_pos < 0) {
                    continue;
                    // page_indices_ptr[i * sqen_len + s] = 0;
                    // pos_in_page_ptr[i * sqen_len + s] = 0;
                    // for (int32_t h = 0; h < head_kv; ++h) {
                    //     for (int32_t d = 0; d < value_dim; ++d) {
                    //         k_nope_pinned_ptr[i * sqen_len * head_kv * value_dim +
                    //                          s * head_kv * value_dim +
                    //                          h * value_dim + d] = static_cast<bfloat16_t>(0.0f);
                    //     }
                    //     for (int32_t d = 0; d < pe_dim; ++d) {
                    //         k_pe_pinned_ptr[i * sqen_len * head_kv * pe_dim +
                    //                        s * head_kv * pe_dim +
                    //                        h * pe_dim + d] = static_cast<bfloat16_t>(0.0f);
                    //     }
                    // }
                    // continue;
                }

                int32_t rel_page_idx = global_token_pos / page_size;
                int32_t offset_in_page = global_token_pos % page_size;

                int32_t physical_page_id = block_table_ptr[req_id * max_num_pages + rel_page_idx];

                page_indices_ptr[i * sqen_len + s] = physical_page_id;
                pos_in_page_ptr[i * sqen_len + s] = offset_in_page;

                for (int32_t h = 0; h < head_kv; ++h) {
                    for (int32_t d = 0; d < value_dim; ++d) {
                        k_nope_pinned_ptr[i * sqen_len * head_kv * value_dim +
                                    s * head_kv * value_dim +
                                    h * value_dim + d] =
                            k_nope_ptr[physical_page_id * page_size * head_kv * value_dim +
                                    offset_in_page * head_kv * value_dim +
                                    h * value_dim + d];
                    }
                    for (int32_t d = 0; d < pe_dim; ++d) {
                        k_pe_pinned_ptr[i * sqen_len * head_kv * pe_dim +
                                    s * head_kv * pe_dim +
                                    h * pe_dim + d] =
                            k_pe_ptr[physical_page_id * page_size * head_kv * pe_dim +
                                    offset_in_page * head_kv * pe_dim +
                                    h * pe_dim + d];
                    }
                }
            }
        }
    }

    return;
}

void
get_kv_topk(
    uintptr_t k_nope_h_ptr,
    uintptr_t k_pe_h_ptr,
    uintptr_t topk_indices_ptr,
    uintptr_t actual_seq_lengths_query_ptr,
    uintptr_t block_table_ptr,
    uintptr_t k_nope_topk_ptr,
    uintptr_t k_pe_topk_ptr,
    std::vector<int64_t> k_nope_h_shape, // [block_num, block_size, head_kv, nope_dim]
    std::vector<int64_t> k_pe_h_shape, // [block_num, block_size, head_kv, rope_dim]
    std::vector<int64_t> topk_indices_shape, // [num_token, 1, topk]
    std::vector<int64_t> actual_seq_lengths_query_shape, // [num_reqs], == num_token if without spec decode
    std::vector<int64_t> block_table_shape, // [num_reqs, max_block_num]
    std::vector<int64_t> k_nope_topk_shape, // output, [num_reqs, topk, head_kv, nope_dim]
    std::vector<int64_t> k_pe_topk_shape, // output, [num_reqs, topk, head_kv, rope_dim]
    const int& thread_num
) {
    at::Tensor k_nope_h = restore_tensor(k_nope_h_ptr, k_nope_h_shape);
    at::Tensor k_pe_h = restore_tensor(k_pe_h_ptr, k_pe_h_shape);
    at::Tensor topk_indices = restore_tensor(topk_indices_ptr, topk_indices_shape, torch::kInt);
    at::Tensor actual_seq_lengths_query = restore_tensor(actual_seq_lengths_query_ptr, actual_seq_lengths_query_shape, torch::kInt);
    at::Tensor block_table = restore_tensor(block_table_ptr, block_table_shape, torch::kInt);
    at::Tensor k_nope_topk = restore_tensor(k_nope_topk_ptr, k_nope_topk_shape);
    at::Tensor k_pe_topk = restore_tensor(k_pe_topk_ptr, k_pe_topk_shape);
    get_kv_topk_kernel(
        k_nope_h,
        k_pe_h,
        topk_indices,
        actual_seq_lengths_query,
        block_table,
        thread_num,
        k_nope_topk,
        k_pe_topk
    );
    return;
}

int32_t
compute_addrs(
    const at::Tensor& topk_idx,
    const at::Tensor& block_table,
    const int32_t block_size,
    const int32_t token_size_bytes_k,
    const int32_t token_size_bytes_v,
    const int64_t gvas_k_base,
    const int64_t gvas_v_base,
    const int64_t addr_k_base,
    const int64_t addr_v_base,
    const int32_t max_num_threads,
    at::Tensor& gvas_buffer, // int64
    at::Tensor& addr_buffer, // int64
    at::Tensor& size_buffer // int32
) {
    const int32_t num_reqs = topk_idx.size(0);
    const int32_t topk = topk_idx.size(1);
    const int32_t max_num_tokens_to_load = num_reqs * topk;
    const int32_t max_num_blocks = block_table.size(1);
    const int32_t block_size_bytes_k = block_size * token_size_bytes_k;
    const int32_t block_size_bytes_v = block_size * token_size_bytes_v;
    TORCH_CHECK(gvas_buffer.size(0) >= max_num_tokens_to_load * 2,
        "gvas_buffer size not enough for loading.");
    TORCH_CHECK(addr_buffer.size(0) >= max_num_tokens_to_load * 2,
        "addr_buffer size not enough for loading.");
    TORCH_CHECK(size_buffer.size(0) >= max_num_tokens_to_load * 2,
        "size_buffer size not enough for loading.");
    TORCH_CHECK(gvas_buffer.scalar_type() == at::kLong,
        "gvas_buffer wrong dtype, should be int64.");
    TORCH_CHECK(addr_buffer.scalar_type() == at::kLong,
        "addr_buffer wrong dtype, should be int64.");
    TORCH_CHECK(size_buffer.scalar_type() == at::kInt,
        "size_buffer wrong dtype, should be int32.");

    const int32_t* topk_idx_ptr = static_cast<int32_t*>(topk_idx.data_ptr());
    const int32_t* block_table_ptr = static_cast<int32_t*>(block_table.data_ptr());
    int64_t* gvas_buffer_ptr = static_cast<int64_t*>(gvas_buffer.data_ptr());
    int64_t* addr_buffer_ptr = static_cast<int64_t*>(addr_buffer.data_ptr());
    int32_t* size_buffer_ptr = static_cast<int32_t*>(size_buffer.data_ptr());

    // int n_threads = 8;
    int n_threads = std::min(num_reqs, max_num_threads);

    std::vector<int32_t> num_tokens_to_load_req(num_reqs);
#pragma omp parallel for num_threads(n_threads)
    for (size_t req_idx = 0; req_idx < num_reqs; ++req_idx) {
        int32_t num_tokens_to_load = 0;
        for (size_t token_idx = 0; token_idx < topk; ++token_idx) {
            int32_t token_indice = topk_idx_ptr[req_idx * topk + token_idx];
            if (token_indice > -1) {
                ++num_tokens_to_load;
            }
        }
        num_tokens_to_load_req[req_idx] = num_tokens_to_load;
    }
    int32_t num_tokens_to_load_sum = std::accumulate(num_tokens_to_load_req.begin(), num_tokens_to_load_req.end(), 0);
    std::vector<int32_t> req_start_locs(num_reqs);
    int32_t cumsum = 0;
    for (size_t req_idx = 0; req_idx < num_reqs; ++req_idx) {
        req_start_locs[req_idx] = cumsum;
        cumsum += num_tokens_to_load_req[req_idx];
    }

#pragma omp parallel for num_threads(n_threads)
    for (size_t req_idx = 0; req_idx < num_reqs; ++req_idx) {
        int32_t req_start_loc_k = req_start_locs[req_idx];
        int32_t req_start_loc_v = num_tokens_to_load_sum + req_start_loc_k;
        int32_t req_offset = 0;
        for (size_t token_idx = 0; token_idx < topk; ++token_idx) {
            int32_t token_indice = topk_idx_ptr[req_idx * topk + token_idx];
            if (token_indice == -1) {
                continue;
            }
            // gvas
            int32_t block_id = token_indice / block_size;
            int32_t offset_in_block = token_indice % block_size;
            int32_t block_indice = block_table_ptr[req_idx * max_num_blocks + block_id];
            int64_t gvas_k = gvas_k_base + block_indice * block_size_bytes_k + offset_in_block * token_size_bytes_k;
            int64_t gvas_v = gvas_v_base + block_indice * block_size_bytes_v + offset_in_block * token_size_bytes_v;
            gvas_buffer_ptr[req_start_loc_k + req_offset] = gvas_k;
            gvas_buffer_ptr[req_start_loc_v + req_offset] = gvas_v;
            // addr
            int64_t addr_k = addr_k_base + (req_idx * topk + token_idx) * token_size_bytes_k;
            int64_t addr_v = addr_v_base + (req_idx * topk + token_idx) * token_size_bytes_v;
            addr_buffer_ptr[req_start_loc_k + req_offset] = addr_k;
            addr_buffer_ptr[req_start_loc_v + req_offset] = addr_v;
            ++req_offset;
        }
    }
    // size
    std::fill_n(size_buffer_ptr, num_tokens_to_load_sum, token_size_bytes_k / 2);
    std::fill_n(&(size_buffer_ptr[num_tokens_to_load_sum]), num_tokens_to_load_sum, token_size_bytes_v / 2);

    return num_tokens_to_load_sum;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    namespace py = pybind11;
    m.def("get_kv_topk", &get_kv_topk, "High performance topk combine");
    m.def("compute_addrs", &compute_addrs, "Compute sparse h2d needed src(cpu) and dst(npu) address");
}
