#include "fused_copy_sfa_tiling.h"

#include <cstdint>
#include <cstring>
#include <initializer_list>

#include "error/ops_error.h"
#include "register/op_def_registry.h"

namespace optiling {
namespace {

constexpr uint32_t QUERY = 0;
constexpr uint32_t KEY = 1;
constexpr uint32_t VALUE = 2;
constexpr uint32_t SPARSE_INDICES = 3;
constexpr uint32_t HBM_BLOCK_TABLE = 5;
constexpr uint32_t HBM_KEY_ROPE = 9;
constexpr uint32_t DRAM_KEY_ROPE = 10;
constexpr uint32_t DRAM_KV_CACHE = 11;
constexpr uint32_t DRAM_BLOCK_TABLE = 12;
constexpr uint32_t SOURCE_TOKEN_IDS = 13;
constexpr uint32_t COPY_COUNTS = 14;

constexpr int64_t BLOCK_SIZE = 128;
constexpr int64_t CKV_DIM = 512;
constexpr int64_t KPE_DIM = 64;
constexpr int64_t SPARSE_COUNT = 2048;
constexpr int64_t MAX_LOCAL_HEADS = 128;

bool IsShape(const gert::Shape &shape, std::initializer_list<int64_t> dims)
{
    if (shape.GetDimNum() != dims.size()) {
        return false;
    }
    size_t idx = 0;
    for (int64_t dim : dims) {
        if (dim >= 0 && shape.GetDim(idx) != dim) {
            return false;
        }
        ++idx;
    }
    return true;
}

ge::graphStatus CheckFusedInputs(
    gert::TilingContext *context,
    uint32_t &batchSize,
    uint32_t &copyCap,
    uint32_t &hbmMaxBlocks,
    uint32_t &dramMaxBlocks,
    uint32_t &attentionMBase)
{
    auto query = context->GetInputShape(QUERY);
    auto key = context->GetInputShape(KEY);
    auto value = context->GetInputShape(VALUE);
    auto sparse = context->GetInputShape(SPARSE_INDICES);
    auto hbmTable = context->GetInputShape(HBM_BLOCK_TABLE);
    auto hbmRope = context->GetInputShape(HBM_KEY_ROPE);
    auto dramRope = context->GetInputShape(DRAM_KEY_ROPE);
    auto dramKv = context->GetInputShape(DRAM_KV_CACHE);
    auto dramTable = context->GetInputShape(DRAM_BLOCK_TABLE);
    auto sourceIds = context->GetInputShape(SOURCE_TOKEN_IDS);
    auto copyCounts = context->GetInputShape(COPY_COUNTS);
    OPS_ERR_IF(query == nullptr || key == nullptr || value == nullptr || sparse == nullptr ||
                   hbmTable == nullptr || hbmRope == nullptr || dramRope == nullptr ||
                   dramKv == nullptr || dramTable == nullptr || sourceIds == nullptr ||
                   copyCounts == nullptr,
               OPS_LOG_E(context->GetNodeName(), "A required fused input shape is missing."),
               return ge::GRAPH_FAILED);

    const gert::Shape q = query->GetStorageShape();
    const gert::Shape hbmKv = key->GetStorageShape();
    const gert::Shape hbmValue = value->GetStorageShape();
    const gert::Shape slots = sparse->GetStorageShape();
    const gert::Shape hbmBt = hbmTable->GetStorageShape();
    const gert::Shape hbmKpe = hbmRope->GetStorageShape();
    const gert::Shape dramKpe = dramRope->GetStorageShape();
    const gert::Shape dramCkv = dramKv->GetStorageShape();
    const gert::Shape dramBt = dramTable->GetStorageShape();
    const gert::Shape src = sourceIds->GetStorageShape();
    const gert::Shape counts = copyCounts->GetStorageShape();

    OPS_ERR_IF(!IsShape(q, {-1, -1, CKV_DIM}) ||
                   q.GetDim(1) <= 0 ||
                   q.GetDim(1) > MAX_LOCAL_HEADS,
               OPS_LOG_E(context->GetNodeName(),
                         "GLM-5.1 query must be [B,N,512], 1 <= N <= 128."),
               return ge::GRAPH_FAILED);
    batchSize = static_cast<uint32_t>(q.GetDim(0));
    attentionMBase = static_cast<uint32_t>(q.GetDim(1));
    OPS_ERR_IF(batchSize == 0,
               OPS_LOG_E(context->GetNodeName(), "Batch size must be positive."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(!IsShape(hbmKv, {-1, BLOCK_SIZE, 1, CKV_DIM}) ||
                   !IsShape(hbmValue, {hbmKv.GetDim(0), BLOCK_SIZE, 1, CKV_DIM}) ||
                   !IsShape(hbmKpe, {hbmKv.GetDim(0), BLOCK_SIZE, 1, KPE_DIM}),
               OPS_LOG_E(context->GetNodeName(),
                         "HBM CKV/KPE must be [blocks,128,1,512/64], and key/value must match."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(!IsShape(dramCkv, {-1, BLOCK_SIZE, CKV_DIM}) ||
                   !IsShape(dramKpe, {dramCkv.GetDim(0), BLOCK_SIZE, KPE_DIM}),
               OPS_LOG_E(context->GetNodeName(),
                         "DRAM CKV/KPE must be [blocks,128,512/64]."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(!IsShape(slots, {batchSize, 1, SPARSE_COUNT}) ||
                   !IsShape(hbmBt, {batchSize, -1}) ||
                   !IsShape(dramBt, {batchSize, -1}) ||
                   !IsShape(src, {batchSize, -1}) ||
                   !IsShape(counts, {batchSize}),
               OPS_LOG_E(context->GetNodeName(),
                         "slots/tables/source IDs/counts have inconsistent batch shapes."),
               return ge::GRAPH_FAILED);

    copyCap = static_cast<uint32_t>(src.GetDim(1));
    hbmMaxBlocks = static_cast<uint32_t>(hbmBt.GetDim(1));
    dramMaxBlocks = static_cast<uint32_t>(dramBt.GetDim(1));
    OPS_ERR_IF(copyCap != SPARSE_COUNT ||
                   hbmMaxBlocks == 0 || dramMaxBlocks == 0,
               OPS_LOG_E(context->GetNodeName(),
                         "source IDs must have top-k capacity 2048 and block tables must be non-empty."),
               return ge::GRAPH_FAILED);

    auto queryDesc = context->GetInputDesc(QUERY);
    OPS_ERR_IF(queryDesc == nullptr,
               OPS_LOG_E(context->GetNodeName(), "GetInputDesc(QUERY) returned nullptr."),
               return ge::GRAPH_FAILED);
    const ge::DataType floatingType = queryDesc->GetDataType();
    OPS_ERR_IF(floatingType != ge::DT_BF16 && floatingType != ge::DT_FLOAT16,
               OPS_LOG_E(context->GetNodeName(), "Floating inputs must be bf16/fp16."),
               return ge::GRAPH_FAILED);
    for (uint32_t idx : {KEY, VALUE, HBM_KEY_ROPE, DRAM_KEY_ROPE, DRAM_KV_CACHE}) {
        auto desc = context->GetInputDesc(idx);
        OPS_ERR_IF(desc == nullptr || desc->GetDataType() != floatingType,
                   OPS_LOG_E(context->GetNodeName(), "All floating inputs must share one dtype."),
                   return ge::GRAPH_FAILED);
    }
    for (uint32_t idx : {SPARSE_INDICES, HBM_BLOCK_TABLE, DRAM_BLOCK_TABLE,
                         SOURCE_TOKEN_IDS, COPY_COUNTS}) {
        auto desc = context->GetInputDesc(idx);
        OPS_ERR_IF(desc == nullptr || desc->GetDataType() != ge::DT_INT32,
                   OPS_LOG_E(context->GetNodeName(), "Fused metadata must be int32."),
                   return ge::GRAPH_FAILED);
    }

    auto platformInfo = context->GetPlatformInfo();
    OPS_ERR_IF(platformInfo == nullptr,
               OPS_LOG_E(context->GetNodeName(), "Platform info is missing."),
               return ge::GRAPH_FAILED);
    auto platform = platform_ascendc::PlatformAscendC(platformInfo);
    const uint32_t aicNum = platform.GetCoreNumAic();
    const uint32_t aivNum = platform.GetCoreNumAiv();
    OPS_ERR_IF(aicNum == 0 || aivNum < aicNum * 2U,
               OPS_LOG_E(context->GetNodeName(),
                          "The fused Attention schedule requires two AIV "
                          "cores per physical AIC core."),
               return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

}  // namespace

ge::graphStatus TilingFusedCopySfa(
    gert::TilingContext *context)
{
    STATilingInfo staInfo;
    STAInfoParser parser(context);
    if (parser.Parse(staInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    uint32_t batchSize = 0;
    uint32_t copyCap = 0;
    uint32_t hbmMaxBlocks = 0;
    uint32_t dramMaxBlocks = 0;
    uint32_t attentionMBase = 0;
    if (CheckFusedInputs(context, batchSize, copyCap, hbmMaxBlocks, dramMaxBlocks,
                         attentionMBase) !=
        ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    STATilingCheck checker(staInfo);
    if (checker.Process() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // Reuse the unmodified production STA tiler. Its serialized payload is
    // exactly the prefix of the fused payload; the two fused-only uint32
    // fields are appended below without changing the baseline implementation.
    STAMlaTiling staTiling(context);
    if (staTiling.DoOpTiling(&staInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    auto raw = context->GetRawTilingData();
    OPS_ERR_IF(raw == nullptr,
               OPS_LOG_E(context->GetNodeName(), "Raw tiling data is missing."),
               return ge::GRAPH_FAILED);
    FusedCopySfaTilingData fusedTiling;
    const size_t baseSize = raw->GetDataSize();
    const size_t fusedSize = fusedTiling.GetDataSize();
    constexpr size_t fusedSuffixSize = sizeof(uint32_t) * 2U;
    OPS_ERR_IF(fusedSize != baseSize + fusedSuffixSize ||
                   raw->GetCapacity() < fusedSize,
        OPS_LOG_E(context->GetNodeName(),
                  "Unexpected STA/fused tiling layout: base=%zu, fused=%zu, capacity=%zu.",
                  baseSize, fusedSize, raw->GetCapacity()),
        return ge::GRAPH_FAILED);
    auto *payload = static_cast<uint8_t *>(raw->GetData());
    std::memcpy(payload + baseSize, &copyCap, sizeof(copyCap));
    std::memcpy(payload + baseSize + sizeof(copyCap),
                &dramMaxBlocks, sizeof(dramMaxBlocks));
    raw->SetDataSize(fusedSize);

    size_t *workspaces = context->GetWorkspaceSizes(1);
    OPS_ERR_IF(workspaces == nullptr,
               OPS_LOG_E(context->GetNodeName(), "Workspace array is missing."),
               return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareFusedCopySfa(
    gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(FusedCopySfa)
    .Tiling(TilingFusedCopySfa)
    .TilingParse<FusedCopySfaCompileInfo>(
        TilingPrepareFusedCopySfa);

}  // namespace optiling
