/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sparse_tail_attention_tiling.cpp
 * \brief
 */

#include <map>
#include <vector>
#include <numeric>
#include <algorithm>
#include <graph/utils/type_utils.h>
#include "error/ops_error.h"
#include "register/op_def_registry.h"
#include "../op_kernel/sparse_tail_attention_template_tiling_key.h"
#include "sparse_tail_attention_tiling.h"

using std::map;
using std::string;
using std::pair;

using namespace ge;
using namespace AscendC;
namespace optiling {

constexpr uint32_t PRE_LOAD_NUM = 2;
constexpr uint32_t BLOCK_TABLE_ELEM_BYTE = 4;
constexpr int32_t SPARSE_MODE_BAND = 4;

static const std::string QUERY_NAME = "query";
static const std::string KEY_NAME = "key";
static const std::string VALUE_NAME = "value";
static const std::string BLOCK_TABLE_NAME = "block_table";
static const std::string SPARSE_INDICES_NAME = "sparse_indices";
static const std::string QUERY_ROPE_NAME = "query_rope";
static const std::string KEY_ROPE_NAME = "key_rope";
static const std::string ATTEN_OUT_NAME = "attention_out";

const std::map<std::string, std::vector<ge::DataType>> DTYPE_SUPPORT_MAP = {
    {QUERY_NAME,                  {ge::DT_FLOAT16, ge::DT_BF16}},
    {KEY_NAME,                    {ge::DT_FLOAT16, ge::DT_BF16}},
    {VALUE_NAME,                  {ge::DT_FLOAT16, ge::DT_BF16}},
    {QUERY_ROPE_NAME,             {ge::DT_FLOAT16, ge::DT_BF16}},
    {KEY_ROPE_NAME,               {ge::DT_FLOAT16, ge::DT_BF16}},
    {ATTEN_OUT_NAME,              {ge::DT_FLOAT16, ge::DT_BF16}},
    {SPARSE_INDICES_NAME,         {ge::DT_INT32}}
};

const std::map<std::string, std::vector<STALayout>> LAYOUT_SUPPORT_MAP = {
    {QUERY_NAME,             {STALayout::BSND, STALayout::TND}},
    {KEY_NAME,               {STALayout::BSND, STALayout::TND, STALayout::PA_BSND}},
    {VALUE_NAME,             {STALayout::BSND, STALayout::TND, STALayout::PA_BSND}},
    {ATTEN_OUT_NAME,         {STALayout::BSND, STALayout::TND}},
};

const std::map<ge::DataType, std::string> DATATYPE_TO_STRING_MAP = {
    {ge::DT_UNDEFINED, "DT_UNDEFINED"},           // Used to indicate a DataType field has not been set.
    {ge::DT_FLOAT, "DT_FLOAT"},                   // float type
    {ge::DT_FLOAT16, "DT_FLOAT16"},               // fp16 type
    {ge::DT_INT8, "DT_INT8"},                     // int8 type
    {ge::DT_INT16, "DT_INT16"},                   // int16 type
    {ge::DT_UINT16, "DT_UINT16"},                 // uint16 type
    {ge::DT_UINT8, "DT_UINT8"},                   // uint8 type
    {ge::DT_INT32, "DT_INT32"},                   // uint32 type
    {ge::DT_INT64, "DT_INT64"},                   // int64 type
    {ge::DT_UINT32, "DT_UINT32"},                 // unsigned int32
    {ge::DT_UINT64, "DT_UINT64"},                 // unsigned int64
    {ge::DT_BOOL, "DT_BOOL"},                     // bool type
    {ge::DT_DOUBLE, "DT_DOUBLE"},                 // double type
    {ge::DT_DUAL, "DT_DUAL"},                     // dual output type
    {ge::DT_DUAL_SUB_INT8, "DT_DUAL_SUB_INT8"},   // dual output int8 type
    {ge::DT_DUAL_SUB_UINT8, "DT_DUAL_SUB_UINT8"}, // dual output uint8 type
    {ge::DT_COMPLEX32, "DT_COMPLEX32"},           // complex32 type
    {ge::DT_COMPLEX64, "DT_COMPLEX64"},           // complex64 type
    {ge::DT_COMPLEX128, "DT_COMPLEX128"},         // complex128 type
    {ge::DT_QINT8, "DT_QINT8"},                   // qint8 type
    {ge::DT_QINT16, "DT_QINT16"},                 // qint16 type
    {ge::DT_QINT32, "DT_QINT32"},                 // qint32 type
    {ge::DT_QUINT8, "DT_QUINT8"},                 // quint8 type
    {ge::DT_QUINT16, "DT_QUINT16"},               // quint16 type
    {ge::DT_RESOURCE, "DT_RESOURCE"},             // resource type
    {ge::DT_STRING_REF, "DT_STRING_REF"},         // string ref type
    {ge::DT_STRING, "DT_STRING"},                 // string type
    {ge::DT_VARIANT, "DT_VARIANT"},               // dt_variant type
    {ge::DT_BF16, "DT_BFLOAT16"},                 // dt_bfloat16 type
    {ge::DT_INT4, "DT_INT4"},                     // dt_variant type
    {ge::DT_UINT1, "DT_UINT1"},                   // dt_variant type
    {ge::DT_INT2, "DT_INT2"},                     // dt_variant type
    {ge::DT_UINT2, "DT_UINT2"}                    // dt_variant type
};

struct SparseTailAttentionCompileInfo {
    int64_t core_num;
};

static const std::map<STALayout, std::vector<STAAxis>> STA_LAYOUT_AXIS_MAP = {
    {STALayout::BSND, {STAAxis::B, STAAxis::S, STAAxis::N, STAAxis::D}},
    {STALayout::TND, {STAAxis::T, STAAxis::N, STAAxis::D}},
    {STALayout::PA_BSND, {STAAxis::Bn, STAAxis::Bs, STAAxis::N, STAAxis::D}},
};

static const std::map<STALayout, size_t> STA_LAYOUT_DIM_MAP = {
    {STALayout::BSND, DIM_NUM_FOUR},
    {STALayout::TND, DIM_NUM_THREE},
    {STALayout::PA_BSND, DIM_NUM_FOUR},
};

static std::string GetShapeStr(gert::Shape shape)
{
    std::ostringstream oss;
    oss << "[";
    if (shape.GetDimNum() > 0) {
        for (size_t i = 0; i < shape.GetDimNum() - 1; ++i) {
            oss << shape.GetDim(i) << ", ";
        }
        oss << shape.GetDim(shape.GetDimNum() - 1);
    }
    oss << "]";
    return oss.str();
}

static std::string STADataTypeToSerialString(ge::DataType type)
{
    const auto it = DATATYPE_TO_STRING_MAP.find(type);
    if (it != DATATYPE_TO_STRING_MAP.end()) {
        return it->second;
    } else {
        OPS_LOG_E("SparseTailAttention", "datatype %d is not supported", type);
        return "UNDEFINED";
    }
}

string STATensorDesc2String(const gert::StorageShape *shape, const gert::CompileTimeTensorDesc *tensor)
{
    if (shape == nullptr || tensor == nullptr) {
        return "nil ";
    }

    std::ostringstream oss;
    oss << "(dtype: " << ge::TypeUtils::DataTypeToAscendString(tensor->GetDataType()).GetString() << "),";
    oss << "(shape:" << STAShape2String(shape->GetStorageShape()) << "),";
    oss << "(ori_shape:" << STAShape2String(shape->GetOriginShape()) << "),";
    oss << "(format: "
        << ge::TypeUtils::FormatToAscendString(
               static_cast<ge::Format>(ge::GetPrimaryFormat(tensor->GetStorageFormat())))
               .GetString()
        << "),";
    oss << "(ori_format: " << ge::TypeUtils::FormatToAscendString(tensor->GetOriginFormat()).GetString() << ") ";

    return oss.str();
}

string STADebugTilingContext(const gert::TilingContext *context)
{
    std::ostringstream oss;
    for (size_t i = 0; i < context->GetComputeNodeInfo()->GetInputsNum(); ++i) {
        oss << "input" << i << ": ";
        oss << STATensorDesc2String(context->GetInputShape(i), context->GetInputDesc(i));
    }

    for (size_t i = 0; i < context->GetComputeNodeInfo()->GetOutputsNum(); ++i) {
        oss << "output" << i << ": ";
        oss << STATensorDesc2String(context->GetOutputShape(i), context->GetOutputDesc(i));
    }
    return oss.str();
}

std::string STALayoutToSerialString(STALayout layout)
{
    switch (layout) {
        case STALayout::BSND: return "BSND";
        case STALayout::TND: return "TND";
        case STALayout::PA_BSND: return "PA_BSND";
        default: return "UNKNOWN";
    }
}

ge::graphStatus STAMlaTiling::SetBlockDim(uint32_t blockDim)
{
    context_->SetBlockDim(blockDim);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus STAMlaTiling::SetTilingKey(uint64_t tilingKey)
{
    context_->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus STAMlaTiling::SetWorkspaceSize(uint64_t workspaceSize)
{
    OPS_ERR_IF(context_->GetWorkspaceSizes(1) == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "workSpaceSize got from ge is nullptr"),
        return ge::GRAPH_FAILED);
    size_t *workSpaces = context_->GetWorkspaceSizes(1);
    workSpaces[0] = workspaceSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus STAMlaTiling::SetTilingData(TilingDef &tilingData)
{
    OPS_ERR_IF(context_->GetRawTilingData() == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "RawTilingData got from GE context is nullptr."),
        return ge::GRAPH_FAILED);

    tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus STAMlaTiling::GetPlatformInfo()
{
    OPS_ERR_IF(staInfo_->platformInfo == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(staInfo_->opName, "GetPlatformInfo is nullptr."), return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(staInfo_->platformInfo);
    libapiSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();
    aivNum_ = ascendcPlatform.GetCoreNumAiv();
    aicNum_ = ascendcPlatform.GetCoreNumAic();

    OPS_ERR_IF(aicNum_ == 0 || aivNum_ == 0,
        OPS_REPORT_VECTOR_INNER_ERR(staInfo_->opName, "num of core obtained is 0."), return GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

void STAMlaTiling::GenTilingKey()
{
    tilingKey_ = GET_TPL_TILING_KEY(0U, static_cast<uint32_t>(STALayout::TND),
                                   static_cast<uint32_t>(STALayout::PA_BSND), 1U);

    OPS_LOG_I(staInfo_->opName, "STA tilingKey_: %lu.", tilingKey_);
}

void STAMlaTiling::ZeroTensorProcess()
{
    if (staInfo_->s2Size == 0) {
        staInfo_->s2Size = 1024;
    }
}

void STAMlaTiling::InitParams()
{
    perfMode_ = STAPerfMode::V_TEMPLATE_MODE;
   
    coreNum_ = aicNum_;

    headDimAlign_ = Align(staInfo_->qkHeadDim, BYTE_BLOCK);
    ZeroTensorProcess();
}

void STAMlaTiling::CalcUbBmm()
{
    uint32_t cubeMSize = staInfo_->gSize * staInfo_->s1Size;
    uint32_t maxMSize = mBaseSize_; 
    if (cubeMSize > maxMSize) {
        cubeMSize = maxMSize;
    }
    mmResUbSize_ = sInnerSizeAlign_ * Align(cubeMSize, 16U);
    bmm2ResUbSize_ = headDimAlign_ * Align(cubeMSize, 16U);

    qPreSizeMla_ = staInfo_->gSize * (headDimAlign_ + 64U) * staInfo_->s1Size;
}

void STAMlaTiling::CheckUbSpace()
{
    CalcUbBmm();
}

void STAMlaTiling::CalcInnerSize(uint32_t s2Size)
{
    sInnerSize_ = 512;
    if (splitKVFlag_ && staInfo_->qLayout != STALayout::TND) {
        if (s2Size == 256) {
            sInnerSize_ = 128;
        } else if (s2Size > 256 && s2Size <= sInnerSize_) {
            sInnerSize_ = (sInnerSize_ + 1) / 2;
        }
    }

    sInnerLoopTimes_ = (s2Size + sInnerSize_ - 1) / sInnerSize_;
    sInnerSizeTail_ = s2Size - (sInnerLoopTimes_ - 1) * sInnerSize_;
    if (sInnerSize_ > s2Size) {
        sInnerSize_ = s2Size;
    }
    sInnerSizeAlign_ = Align(sInnerSize_, BYTE_BLOCK);

    CheckUbSpace();
}

void STAMlaTiling::SplitBalanced()
{
    CalcInnerSize(staInfo_->s2Size);

    InnerSplitParams innerSplitParams;
    innerSplitParams.s1GBaseSize = staInfo_->gSize; 
    innerSplitParams.s2BaseSize = sInnerSize_;
    tilingData_.innerSplitParams.set_mBaseSize(innerSplitParams.s1GBaseSize);
    tilingData_.innerSplitParams.set_s2BaseSize(innerSplitParams.s2BaseSize);

    usedCoreNum_ = aicNum_;
}

void STAMlaTiling::Split()
{
    SplitBalanced();
}

void STAMlaTiling::FillTilingBaseParamsMla()
{
    tilingData_.baseParams.set_batchSize(staInfo_->bSize);
    tilingData_.baseParams.set_seqSize(staInfo_->s2Size);
    tilingData_.baseParams.set_qSeqSize(staInfo_->s1Size);
    tilingData_.baseParams.set_blockSize(staInfo_->blockSize);
    tilingData_.baseParams.set_maxBlockNumPerBatch(staInfo_->maxBlockNumPerBatch);
    tilingData_.baseParams.set_scaleValue(staInfo_->scaleValue);
    tilingData_.baseParams.set_nNumOfQInOneGroup(staInfo_->n1Size / staInfo_->n2Size);
    tilingData_.baseParams.set_actualLenDimsQ(staInfo_->actualLenDimsQ);
    tilingData_.baseParams.set_actualLenDimsKV(staInfo_->actualLenDimsKV);
    tilingData_.baseParams.set_outputLayout(static_cast<uint32_t>(staInfo_->outLayout));
    tilingData_.baseParams.set_sparseMode(staInfo_->sparseMode);
    tilingData_.baseParams.set_sparseBlockSize(staInfo_->sparseBlockSize);
    tilingData_.baseParams.set_sparseBlockCount(OFFLOAD_SPARSE_COMPUTE_COUNT);
}

// for flash decode
void STAMlaTiling::FillTilingSplitKVMla()
{
    tilingData_.splitKVParams.set_s2(kvSplitPart_);

    tilingData_.splitKVParams.set_accumOutSize(aicNum_ * 2 * staInfo_->n2Size * mBaseSize_ * headDimAlign_);
    tilingData_.splitKVParams.set_logSumExpSize(2 * aicNum_ * 2 * staInfo_->n2Size * mBaseSize_ *
                                                (BYTE_BLOCK / BLOCK_TABLE_ELEM_BYTE));

    if (!splitKVFlag_) {
        tilingData_.splitKVParams.set_s2(0);
    }
}

void STAMlaTiling::FillTilingSingleCoreParamsMla()
{
    tilingData_.singleCoreParams.set_usedCoreNum(usedCoreNum_);
}

void STAMlaTiling::FillTilingSingleCoreTensorSizeMla()
{
    tilingData_.singleCoreTensorSize.set_mmResUbSize(mmResUbSize_);
    tilingData_.singleCoreTensorSize.set_bmm2ResUbSize(bmm2ResUbSize_);
}

void STAMlaTiling::FillTiling()
{
    FillTilingBaseParamsMla();
    FillTilingSplitKVMla();
    FillTilingSingleCoreParamsMla();
    FillTilingSingleCoreTensorSizeMla();
}

uint32_t STAMlaTiling::CalcBalanceFDParamNums(const uint32_t actCoreNum)
{
    return actCoreNum * 2 * staInfo_->n2Size * mBaseSize_;
}

void STAMlaTiling::NormalCalcFDWorkSpace(const uint32_t actCoreNum)
{
    if (splitKVFlag_) {
        uint32_t accumOutSize = 0;
        uint32_t logSumExpSize = 0;
        uint32_t FDParamNums = CalcBalanceFDParamNums(actCoreNum);
        accumOutSize = FDParamNums * headDimAlign_;
        logSumExpSize = 2 * FDParamNums * (BYTE_BLOCK / staInfo_->blockTypeSize);
        workspaceSize_ += (accumOutSize + logSumExpSize) * staInfo_->blockTypeSize;
        if (staInfo_->socVersion == platform_ascendc::SocVersion::ASCEND310P) {
            workspaceSize_ += static_cast<size_t>(actCoreNum) * 32;
        }
    }
}

void STAMlaTiling::CalcFDWorkSpace(const uint32_t actCoreNum)
{
    NormalCalcFDWorkSpace(actCoreNum);
}

void STAMlaTiling::GetWorkspaceSize()
{
    uint32_t mmResElemSize = 4;
    uint32_t vec1ResElemSize = 2;
    uint32_t bmm2ResElemSize = 4;
    uint32_t qPreProcResElemSize = 0;
    uint32_t nUpdateElemSize = 4;
    uint32_t softmaxSumElemSize = 4;
    float kvDtypeRatio = 1.0;

    workspaceSize_ = libapiSize_;
    uint32_t preLoadNum = 1;
    uint32_t actCoreNum = coreNum_;
    preLoadNum = PRE_LOAD_NUM;

    workspaceSize_ += preLoadNum * (mmResUbSize_ * actCoreNum * mmResElemSize);
    workspaceSize_ += preLoadNum * static_cast<size_t>(static_cast<float>(mmResUbSize_ * actCoreNum * vec1ResElemSize) * kvDtypeRatio);
    workspaceSize_ += preLoadNum * bmm2ResUbSize_ * actCoreNum * bmm2ResElemSize;
    workspaceSize_ += preLoadNum * static_cast<size_t>(static_cast<float>(qPreSizeMla_ * actCoreNum * qPreProcResElemSize) * kvDtypeRatio);
    workspaceSize_ += preLoadNum * mBaseSize_ * actCoreNum * nUpdateElemSize;
    workspaceSize_ += preLoadNum * mBaseSize_ * actCoreNum * softmaxSumElemSize;
    workspaceSize_ += 4 * 512 * (512 + 64) * 2 * actCoreNum;
    workspaceSize_ += 4 * 128 * 4 * (2 * actCoreNum);

    CalcFDWorkSpace(actCoreNum);
}

void STAMlaTiling::CalcBlockDim()
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(staInfo_->platformInfo);
    auto aicNum = usedCoreNum_;
    auto aivNum = 2 * usedCoreNum_;

    blockDim_ = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    OPS_LOG_I(staInfo_->opName, "STA block dim: %u aiv Num: %u aic Num: %u.", blockDim_, aivNum, aicNum);
}

ge::graphStatus STAMlaTiling::DoOpTiling(STATilingInfo *staInfo)
{
    staInfo_ = staInfo;
    if (GetPlatformInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    InitParams();
    Split();
    FillTiling();
    CalcBlockDim();
    GetWorkspaceSize();
    GenTilingKey();

    if ((SetBlockDim(blockDim_) != ge::GRAPH_SUCCESS) ||
        (SetTilingKey(tilingKey_) != ge::GRAPH_SUCCESS) ||
        (SetWorkspaceSize(workspaceSize_) != ge::GRAPH_SUCCESS) ||
        (SetTilingData(tilingData_) != ge::GRAPH_SUCCESS)) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingSparseTailAttention(gert::TilingContext *context)
{
    STATilingInfo staInfo;
    STAInfoParser staInfoParser(context);
    if (staInfoParser.Parse(staInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    STATilingCheck tilingChecker(staInfo);
    if (tilingChecker.Process() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    STAMlaTiling tiling(context);
    return tiling.DoOpTiling(&staInfo);
}

ge::graphStatus TilingSparseTailAttentionMtp(gert::TilingContext *context)
{
    const ge::graphStatus status = TilingSparseTailAttention(context);
    if (status == ge::GRAPH_SUCCESS) {
        // MTP3 has one fixed kernel specialization: query_len=4, TND query,
        // PA_BSND KV and V-template. Keep its dispatch key independent from
        // the configurable single-query attention template key.
        context->SetTilingKey(1U);
    }
    return status;
}

ge::graphStatus TilingPrepareForSparseTailAttention(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus STATilingCheck::GetExpectedShape(gert::Shape &shapeExpected,
    const STATilingShapeCompareParam &param, const STALayout &layout) const
{
    if (layout == STALayout::BSND) {
        shapeExpected = gert::Shape({param.B, param.S, param.N, param.D});
    } else if (layout == STALayout::TND) {
        shapeExpected = gert::Shape({param.T, param.N, param.D});
    } else if (layout == STALayout::PA_BSND) {
        shapeExpected = gert::Shape({param.Bn, param.Bs, param.N, param.D});
    } else {
        OPS_LOG_E(opName_, "layout %s is unsupported", STALayoutToSerialString(layout).c_str());
        return ge::GRAPH_FAILED;
    }
    if (shapeExpected.GetDim(0) == 0) {
        OPS_LOG_E(opName_, "expected shape is %s, the first dim should not be 0.", GetShapeStr(shapeExpected).c_str());
        return ge::GRAPH_PARAM_INVALID;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus STATilingCheck::CompareShape(STATilingShapeCompareParam &param,
    const gert::Shape &shape, const STALayout &layout, const std::string &name) const
{
    gert::Shape shapeExpected;
    if (GetExpectedShape(shapeExpected, param, layout) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (shape.GetDimNum() != shapeExpected.GetDimNum()) {
        OPS_LOG_E(opName_,
            "%s dimension is %zu, expected dimension is %zu.",
            name.c_str(), shape.GetDimNum(), shapeExpected.GetDimNum());
        return ge::GRAPH_FAILED;
    }

    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        if (shape.GetDim(i) != shapeExpected.GetDim(i)) {
            OPS_LOG_E(opName_, "%s layout is %s, shape is %s, expected shape is %s.",
                name.c_str(), STALayoutToSerialString(layout).c_str(),
                GetShapeStr(shape).c_str(), GetShapeStr(shapeExpected).c_str());
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

void STATilingCheck::LogErrorDtypeSupport(const std::vector<ge::DataType> &expectDtypeList,
    const ge::DataType &actualDtype, const std::string &name) const
{
    std::ostringstream oss;
    for (size_t i = 0; i < expectDtypeList.size(); ++i) {
        oss << STADataTypeToSerialString(expectDtypeList[i]);
        if (i < expectDtypeList.size() - 1) {
            oss << ", ";
        }
    }
    OPS_LOG_E(opName_, "Tensor %s only supports dtype %s, but got %s",
        name.c_str(), oss.str().c_str(), STADataTypeToSerialString(actualDtype).c_str());
}

ge::graphStatus STATilingCheck::CheckDtypeSupport(const gert::CompileTimeTensorDesc *desc,
    const std::string &name) const
{
    if (desc != nullptr) {
        const auto& it = DTYPE_SUPPORT_MAP.find(name);
        OPS_ERR_IF(it == DTYPE_SUPPORT_MAP.end(),
            OPS_LOG_E(opName_, "%s datatype support list should be specified in DTYPE_SUPPORT_MAP", name.c_str()),
            return ge::GRAPH_FAILED);
        auto &expectDtypeList = it->second;
        OPS_ERR_IF(std::find(
            expectDtypeList.begin(), expectDtypeList.end(), desc->GetDataType()) == expectDtypeList.end(),
            LogErrorDtypeSupport(expectDtypeList, desc->GetDataType(), name),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

template <typename T>
void STATilingCheck::LogErrorNumberSupport(const std::vector<T> &expectNumberList,
    const T &actualValue, const std::string &name, const std::string &subName) const
{
    std::ostringstream oss;
    for (size_t i = 0; i < expectNumberList.size(); ++i) {
        oss << std::to_string(expectNumberList[i]);
        if (i < expectNumberList.size() - 1) {
            oss << ", ";
        }
    }

    OPS_LOG_E(opName_, "%s %s only supports %s, but got %s",
              name.c_str(), subName.c_str(), oss.str().c_str(), std::to_string(actualValue).c_str());
}

template <typename T>
void STATilingCheck::LogErrorDimNumSupport(const std::vector<T> &expectNumberList,
    const T &actualValue, const std::string &name) const
{
    LogErrorNumberSupport(expectNumberList, actualValue, name, "dimension");
}

ge::graphStatus STATilingCheck::CheckDimNumInLayoutSupport(const STALayout &layout,
    const gert::StorageShape *shape, const std::string &name) const
{
    const auto& dimIt = STA_LAYOUT_DIM_MAP.find(layout);
    OPS_ERR_IF(shape->GetStorageShape().GetDimNum() != dimIt->second,
        OPS_LOG_E(opName_, "When layout is %s, %s dimension should be %zu, but it's %zu",
            STALayoutToSerialString(layout).c_str(), name.c_str(), dimIt->second,
            shape->GetStorageShape().GetDimNum()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus STATilingCheck::CheckDimNumSupport(const gert::StorageShape *shape,
    const std::vector<size_t> &expectDimNumList, const std::string &name) const
{
    if (shape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    if (std::find(expectDimNumList.begin(), expectDimNumList.end(),
        shape->GetStorageShape().GetDimNum()) == expectDimNumList.end()) {
        LogErrorDimNumSupport(expectDimNumList, shape->GetStorageShape().GetDimNum(), name);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}


void STATilingCheck::LogErrorLayoutSupport(const std::vector<STALayout> &expectLayoutList,
    const STALayout &actualLayout, const std::string &name) const
{
    std::ostringstream oss;
    for (size_t i = 0; i < expectLayoutList.size(); ++i) {
        oss << STALayoutToSerialString(expectLayoutList[i]);
        if (i < expectLayoutList.size() - 1) {
            oss << ", ";
        }
    }
    OPS_LOG_E(opName_, "Tensor %s only supports layout %s, but got %s",
        name.c_str(), oss.str().c_str(), STALayoutToSerialString(actualLayout).c_str());
}

ge::graphStatus STATilingCheck::CheckLayoutSupport(const STALayout &actualLayout, const std::string &name) const
{
    const auto& it = LAYOUT_SUPPORT_MAP.find(name);
    OPS_ERR_IF(it == LAYOUT_SUPPORT_MAP.end(),
        OPS_LOG_E(opName_, "%s layout support list should be specified in LAYOUT_SUPPORT_MAP", name.c_str()),
        return ge::GRAPH_FAILED);
    auto &expectLayoutList = it->second;
    OPS_ERR_IF(std::find(
        expectLayoutList.begin(), expectLayoutList.end(), actualLayout) == expectLayoutList.end(),
        LogErrorLayoutSupport(expectLayoutList, actualLayout, name),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus STATilingCheck::CheckSingleParaQuery() const
{
    const std::vector<size_t> queryDimNumList = {DIM_NUM_THREE, DIM_NUM_FOUR};
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.query.desc, QUERY_NAME) ||
        ge::GRAPH_SUCCESS != CheckLayoutSupport(qLayout_, QUERY_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(opParamInfo_.query.shape, queryDimNumList, QUERY_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumInLayoutSupport(qLayout_, opParamInfo_.query.shape, QUERY_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus STATilingCheck::CheckSingleParaKey() const
{
    const std::vector<size_t> keyDimNumList = {DIM_NUM_FOUR, DIM_NUM_THREE};
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.key.desc, KEY_NAME) ||
        ge::GRAPH_SUCCESS != CheckLayoutSupport(kvLayout_, KEY_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(opParamInfo_.key.shape, keyDimNumList, KEY_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumInLayoutSupport(kvLayout_, opParamInfo_.key.shape, KEY_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus STATilingCheck::CheckSingleParaNumHeads() const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus STATilingCheck::CheckSingleParaKvHeadNums() const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus STATilingCheck::CheckSingleParaSparseMode() const
{
    OPS_ERR_IF((*opParamInfo_.sparseMode != 3 && *opParamInfo_.sparseMode != 0),
        OPS_LOG_E(opName_, "sparseMode must == 0/3, but got: %ld.", *opParamInfo_.sparseMode),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus STATilingCheck::CheckSingleParaSparseBlockSize() const
{
    OPS_ERR_IF((*opParamInfo_.sparseBlockSize <= 0),
        OPS_LOG_E(opName_, "sparseBlockSize should be greater than 0, but got: %ld.", *opParamInfo_.sparseBlockSize),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus STATilingCheck::CheckSingleParaSparseIndices() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.sparseIndices.desc, SPARSE_INDICES_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus STATilingCheck::CheckSinglePara() const
{
    if (ge::GRAPH_SUCCESS != CheckSingleParaQuery() ||
        ge::GRAPH_SUCCESS != CheckSingleParaKey() ||
        ge::GRAPH_SUCCESS != CheckSingleParaSparseIndices() || 
        ge::GRAPH_SUCCESS != CheckSingleParaNumHeads() ||
        ge::GRAPH_SUCCESS != CheckSingleParaKvHeadNums() ||
        ge::GRAPH_SUCCESS != CheckSingleParaSparseMode() ||
        ge::GRAPH_SUCCESS != CheckSingleParaSparseBlockSize()) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus STATilingCheck::CheckRopeExistence()
{
    OPS_ERR_IF((opParamInfo_.queryRope.tensor != nullptr && opParamInfo_.keyRope.tensor == nullptr),
        OPS_LOG_E(opName_, "KeyRope is null, but queryRope exists, they should be both null or exist."),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF((opParamInfo_.queryRope.tensor == nullptr && opParamInfo_.keyRope.tensor != nullptr),
        OPS_LOG_E(opName_, "QueryRope is null, but keyRope exists, they should be both null or exist."),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.keyRope.desc == nullptr || opParamInfo_.queryRope.desc == nullptr,
        OPS_LOG_E(opName_, "In Mla situation, desc of keyRope and queryRope should not be null"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus STATilingCheck::CheckExists(const void *pointer, const std::string &name) const
{
    OPS_ERR_IF(pointer == nullptr,
        OPS_LOG_E(opName_, "%s should not be null", name.c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus STATilingCheck::CheckNotExists(const void *pointer, const std::string &name) const
{
    OPS_ERR_IF(pointer != nullptr,
        OPS_LOG_E(opName_, "%s should be null", name.c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus STATilingCheck::CheckExistsByMap(const std::map<std::string, const void *> &paramMap) const
{
    for (const auto& kv : paramMap) {
        if (CheckExists(kv.second, kv.first) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus STATilingCheck::CheckNotExistsByMap(const std::map<std::string, const void *> &paramMap) const
{
    for (const auto& kv : paramMap) {
        if (CheckNotExists(kv.second, kv.first) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus STATilingCheck::CheckExistenceByMap(std::map<std::string, const void *> &existMap,
    std::map<std::string, const void *> &notExistMap) const
{
    if (CheckExistsByMap(existMap) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckNotExistsByMap(notExistMap) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

template <typename T>
ge::graphStatus STATilingCheck::CheckAttrValueByMap(std::map<std::string, std::pair<const T *, T>> &attrMap) const
{
    for (auto const &kv : attrMap) {
        const std::string &name = kv.first;
        const std::pair<const T *, T> &pointerValuePair = kv.second;
        if (pointerValuePair.first == nullptr) {
            OPS_LOG_E(opName_, "Attr %s should not be nullptr", name.c_str());
            return ge::GRAPH_FAILED;
        }

        if (*(pointerValuePair.first) != pointerValuePair.second) {
            std::ostringstream ossExpect;
            ossExpect << std::to_string(pointerValuePair.second);
            std::ostringstream ossActual;
            ossActual << std::to_string(*(pointerValuePair.first));
            OPS_LOG_E(opName_,
                "%s value should be %s, but got %s",
                name.c_str(),
                ossExpect.str().c_str(),
                ossActual.str().c_str());
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus STATilingCheck::CheckParaExistenceMlaNoquant() const
{
    if (kvStorageMode_ != KvStorageMode::PAGE_ATTENTION) {
        return ge::GRAPH_SUCCESS;
    }
    std::map<std::string, const void *> mlaNoquantParamExistMap = {
        {"actualSeqLengths", opParamInfo_.actualSeqLengths.tensor},
        {"blockTable", opParamInfo_.blockTable.tensor},
    };
    std::map<std::string, const void *> mlaNoquantParamNotExistMap = {};
    if (CheckExistenceByMap(mlaNoquantParamExistMap, mlaNoquantParamNotExistMap) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus STATilingCheck::CheckParaExistenceMla() const
{
    return CheckParaExistenceMlaNoquant();
}

ge::graphStatus STATilingCheck::CheckParaExistence()
{
    if (ge::GRAPH_SUCCESS != CheckRopeExistence()) {
        return ge::GRAPH_FAILED;
    }

    return CheckParaExistenceMla();
}

ge::graphStatus STATilingCheck::GetActualSeqLenSize(uint32_t &size, const gert::Tensor *tensor,
    const STALayout &layoutQuery, const std::string &name)
{
    if (tensor == nullptr) {
        OPS_LOG_E(opName_, "when layout of query is %s, %s must be provided.",
            STALayoutToSerialString(layoutQuery).c_str(), name.c_str());
        return ge::GRAPH_FAILED;
    }
    int64_t shapeSize = tensor->GetShapeSize();
    if (shapeSize <= 0) {
        OPS_LOG_E(opName_, "the shape size of %s is %ld, it should be greater than 0.",
            name.c_str(), shapeSize);
        return ge::GRAPH_FAILED;
    }
    size = static_cast<uint32_t>(shapeSize);
    return ge::GRAPH_SUCCESS;
}

void STATilingCheck::SetSTAShapeCompare()
{
    queryShapeCmp_ = opParamInfo_.query.shape->GetStorageShape();
    topkShapeCmp_ = opParamInfo_.sparseIndices.shape->GetStorageShape();
    keyShapeCmp_ = opParamInfo_.key.shape->GetStorageShape();
    valueShapeCmp_ = opParamInfo_.value.shape->GetStorageShape();
    attenOutShapeCmp_ = opParamInfo_.attenOut.shape->GetStorageShape();
    queryRopeShapeCmp_ = opParamInfo_.queryRope.tensor->GetStorageShape();
    keyRopeShapeCmp_ = opParamInfo_.keyRope.tensor->GetStorageShape();
}

ge::graphStatus STATilingCheck::CheckBlockTable() const
{
    if (kvStorageMode_ != KvStorageMode::PAGE_ATTENTION) {
        OPS_ERR_IF(opParamInfo_.blockTable.tensor != nullptr,
            OPS_LOG_E(opName_, "when the layout_kv is %s, %s should be null",
                STALayoutToSerialString(kvLayout_).c_str(), BLOCK_TABLE_NAME.c_str()),
            return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }
    
    uint32_t blockTableBatch = opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(0);
    OPS_ERR_IF(blockTableBatch != bSize_,
        OPS_LOG_E(opName_, "%s's first dimension(%u) should be equal to batch size(%u)",
            BLOCK_TABLE_NAME.c_str(), blockTableBatch, bSize_),
        return ge::GRAPH_FAILED);
    
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus STATilingCheck::CheckDTypeConsistency(const ge::DataType &actualDtype,
    const ge::DataType &expectDtype, const std::string &name) const
{
    if (actualDtype != expectDtype) {
        OPS_LOG_E(opName_, "%s dtype should be %s, but it's %s.", name.c_str(),
            STADataTypeToSerialString(expectDtype).c_str(),
            STADataTypeToSerialString(actualDtype).c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus STATilingCheck::CheckQRopeShape()
{
    STATilingShapeCompareParam shapeParams;
    shapeParams.B = bSize_;
    shapeParams.N = n1Size_;
    shapeParams.S = s1Size_;
    shapeParams.D = ropeHeadDim_;
    shapeParams.T = qTSize_;
    return CompareShape(shapeParams, queryRopeShapeCmp_, qLayout_, QUERY_ROPE_NAME);
}

ge::graphStatus STATilingCheck::CheckTopkShape()
{
    STATilingShapeCompareParam shapeParams;
    shapeParams.B = bSize_;
    shapeParams.N = n2Size_;
    shapeParams.S = s1Size_;
    shapeParams.D = sparseBlockCount_;
    shapeParams.T = qTSize_;
    return CompareShape(shapeParams, topkShapeCmp_, topkLayout_, SPARSE_INDICES_NAME);
}

ge::graphStatus STATilingCheck::CheckAttenOutShape()
{
    STATilingShapeCompareParam shapeParams;
    shapeParams.B = bSize_;
    shapeParams.N = n1Size_;
    shapeParams.S = s1Size_;
    shapeParams.D = vHeadDim_;
    shapeParams.T = qTSize_;
    if (CompareShape(shapeParams, attenOutShapeCmp_, outLayout_, ATTEN_OUT_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus STATilingCheck::CheckAttenOut()
{
    if (ge::GRAPH_SUCCESS != CheckDTypeConsistency(opParamInfo_.attenOut.desc->GetDataType(),
        inputQType_, ATTEN_OUT_NAME) ||
        ge::GRAPH_SUCCESS != CheckAttenOutShape()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus STATilingCheck::CheckQRope()
{
    if (ge::GRAPH_SUCCESS != CheckDTypeConsistency(opParamInfo_.queryRope.desc->GetDataType(),
        inputQType_, QUERY_ROPE_NAME) ||
        ge::GRAPH_SUCCESS != CheckQRopeShape()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus STATilingCheck::CheckTopK()
{
    if (ge::GRAPH_SUCCESS != CheckTopkShape()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus STATilingCheck::CheckVAndKRopeShapeForBatchContinuous()
{
    STATilingShapeCompareParam shapeParams;
    shapeParams.B = bSize_;
    shapeParams.N = n2Size_;
    shapeParams.S = s2Size_;
    shapeParams.T = kvTSize_;
    shapeParams.D = qkHeadDim_;
    if (CompareShape(shapeParams, keyShapeCmp_, kvLayout_, KEY_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    shapeParams.D = vHeadDim_;
    if (CompareShape(shapeParams, valueShapeCmp_, kvLayout_, VALUE_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    shapeParams.D = ropeHeadDim_;
    if (CompareShape(shapeParams, keyRopeShapeCmp_, kvLayout_, KEY_ROPE_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

uint32_t STATilingCheck::GetTypeSize(ge::DataType dtype) const
{
    uint32_t typeSize = NUM_BYTES_FLOAT16;
    switch (dtype) {
        case ge::DT_FLOAT16:
            typeSize = NUM_BYTES_FLOAT16;
            break;
        case ge::DT_BF16:
            typeSize = NUM_BYTES_BF16;
            break;
        default:
            typeSize = NUM_BYTES_FLOAT16;
    }
    return typeSize;
}

ge::graphStatus STATilingCheck::CheckVAndKRopeShapeForPageAttention()
{
    int64_t blockNum = keyShapeCmp_.GetDim(0);
    OPS_ERR_IF(blockNum <= 0,
        OPS_LOG_E(opName_, "The first dim(%ld) of key should be greater than 0", blockNum),
        return ge::GRAPH_FAILED);
    STATilingShapeCompareParam shapeParams;
    shapeParams.Bn = blockNum;
    shapeParams.N = n2Size_;
    shapeParams.Bs = blockSize_;
    shapeParams.D = vHeadDim_;
    shapeParams.T = kvTSize_;
    if (CompareShape(shapeParams, valueShapeCmp_, kvLayout_, VALUE_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    shapeParams.D = ropeHeadDim_;
    if (CompareShape(shapeParams, keyRopeShapeCmp_, kvLayout_, KEY_ROPE_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus STATilingCheck::CheckVAndKRopeShape()
{
    if (kvStorageMode_ == KvStorageMode::BATCH_CONTINUOUS) {
        return CheckVAndKRopeShapeForBatchContinuous();
    }

    if (kvStorageMode_ == KvStorageMode::PAGE_ATTENTION) {
        return CheckVAndKRopeShapeForPageAttention();
    }

    OPS_LOG_E(opName_, "storage mode of key and value is %u, it is incorrect.", static_cast<uint32_t>(kvStorageMode_));
    return ge::GRAPH_FAILED;
}

ge::graphStatus STATilingCheck::CheckVAndKRope()
{
    if (ge::GRAPH_SUCCESS != CheckDTypeConsistency(opParamInfo_.value.desc->GetDataType(),
        inputKvType_, VALUE_NAME) ||
        ge::GRAPH_SUCCESS != CheckDTypeConsistency(opParamInfo_.keyRope.desc->GetDataType(),
        inputKvType_, KEY_ROPE_NAME) || ge::GRAPH_SUCCESS != CheckVAndKRopeShape()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus STATilingCheck::CheckActualSeqLensQ()
{
    if (ge::GRAPH_SUCCESS != CheckActualSeqLensQDType() ||
        ge::GRAPH_SUCCESS != CheckActualSeqLensQShape()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus STATilingCheck::CheckActualSeqLensQDType()
{
    if (opParamInfo_.actualSeqLengthsQ.tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (opParamInfo_.actualSeqLengthsQ.desc == nullptr) {
        OPS_LOG_E(opName_, "actualSeqLengthsQ is not empty,"
            "but actualSeqLengthsQ's dtype is nullptr.");
            return ge::GRAPH_FAILED;
    }
    if (opParamInfo_.actualSeqLengthsQ.desc->GetDataType() != ge::DT_INT32) {
        OPS_LOG_E(opName_, "actualSeqLengthsQ's dtype is %s, it should be DT_INT32.",
            STADataTypeToSerialString(opParamInfo_.actualSeqLengthsQ.desc->GetDataType()).c_str());
            return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus STATilingCheck::CheckActualSeqLensQShape()
{
    if (opParamInfo_.actualSeqLengthsQ.tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    uint32_t shapeSize = 0;
    if (GetActualSeqLenSize(shapeSize, opParamInfo_.actualSeqLengthsQ.tensor, qLayout_, "actualSeqLengthsQ") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (shapeSize != bSize_) {
        OPS_LOG_E(opName_, "actualSeqLengthsQ shape size is %u, it should be equal to batch size[%u]",
            shapeSize, bSize_);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus STATilingCheck::CheckActualSeqLens()
{
    if (std::string(opParamInfo_.layoutKV) == "TND" && opParamInfo_.actualSeqLengths.tensor == nullptr) {
        OPS_LOG_E(opName_,
                  "when the layout of key and value is TND, "
                  "the actualSeqLengths of key and value should not be empty.");
        return ge::GRAPH_PARAM_INVALID;
    }
    if (ge::GRAPH_SUCCESS != CheckActualSeqLensDType() ||
        ge::GRAPH_SUCCESS != CheckActualSeqLensShape()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus STATilingCheck::CheckActualSeqLensDType()
{
    if (opParamInfo_.actualSeqLengths.tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (opParamInfo_.actualSeqLengths.desc == nullptr) {
        OPS_LOG_E(opName_, "actualSeqLengths is not empty,"
            "but actualSeqLengths's dtype is nullptr.");
            return ge::GRAPH_FAILED;
    }
    if (opParamInfo_.actualSeqLengths.desc->GetDataType() != ge::DT_INT32) {
        OPS_LOG_E(opName_, "actualSeqLengths's dtype is %s, it should be DT_INT32.",
            STADataTypeToSerialString(opParamInfo_.actualSeqLengths.desc->GetDataType()).c_str());
            return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus STATilingCheck::CheckActualSeqLensShape()
{
    if (opParamInfo_.actualSeqLengths.tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    uint32_t shapeSize = 0;
    if(GetActualSeqLenSize(shapeSize, opParamInfo_.actualSeqLengths.tensor, kvLayout_, "actualSeqLengths") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (shapeSize != bSize_) {
        OPS_LOG_E(opName_, "actualSeqLengths shape size is %u, it should be equal to batch size[%u].",
            shapeSize, bSize_);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus STATilingCheck::CheckMultiParaConsistency()
{
    SetSTAShapeCompare();
    if (ge::GRAPH_SUCCESS != CheckVAndKRope() ||
        ge::GRAPH_SUCCESS != CheckQRope() ||
        ge::GRAPH_SUCCESS != CheckTopK() ||
        ge::GRAPH_SUCCESS != CheckAttenOut() ||
        ge::GRAPH_SUCCESS != CheckActualSeqLensQ() ||
        ge::GRAPH_SUCCESS != CheckActualSeqLens() ||
        ge::GRAPH_SUCCESS != CheckBlockTable()) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus STATilingCheck::CheckFeatureMlaNoQuantShape() const
{
    OPS_ERR_IF(bSize_ <= 0,
        OPS_LOG_E(opName_, "batch_size should be greater than 0, but got %u", bSize_),
        return ge::GRAPH_FAILED);
        
    OPS_ERR_IF(qTSize_ <= 0 && (qLayout_ == STALayout::TND),
        OPS_LOG_E(opName_, "T_size of query should be greater than 0, but got %u", qTSize_),
        return ge::GRAPH_FAILED);

    OPS_ERR_IF(n1Size_ <= 0,
        OPS_LOG_E(opName_, "q_head_num should be greater than 0, but got %u", n1Size_),
        return ge::GRAPH_FAILED);

    OPS_ERR_IF(n2Size_ != 1,
        OPS_LOG_E(opName_, "kv_head_num should be 1, but got %u", n2Size_),
        return ge::GRAPH_FAILED);

    OPS_ERR_IF(n1Size_ % n2Size_ != 0,
        OPS_LOG_E(opName_, "q_head_num(%u) must be divisible by kv_head_num(%u)", n1Size_, n2Size_),
        return ge::GRAPH_FAILED);

    std::vector<uint32_t> gSizeSupportList = {1, 2, 4, 8, 16, 32, 64, 128};
    OPS_ERR_IF(std::find(gSizeSupportList.begin(), gSizeSupportList.end(), gSize_) == gSizeSupportList.end(),
        OPS_LOG_E(opName_, "group num should be in 1, 2, 4, 8, 16, 32, 64, 128, but got %u", gSize_),
        return ge::GRAPH_FAILED);

    OPS_ERR_IF(qkHeadDim_ != 512,
        OPS_LOG_E(opName_, "qk_head_dim only support 512, but got %u", qkHeadDim_),
        return ge::GRAPH_FAILED);

    OPS_ERR_IF(qkHeadDim_ != vHeadDim_,
        OPS_LOG_E(opName_, "qk_head_dim[%u] should be equal to v_head_dim[%u]", qkHeadDim_, vHeadDim_),
        return ge::GRAPH_FAILED);

    OPS_ERR_IF(ropeHeadDim_ != 64,
        OPS_LOG_E(opName_, "rope_head_dim should be 64, but got %u", ropeHeadDim_),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus STATilingCheck::CheckFeatureMlaNoQuantLayout() const
{
    const std::vector<std::string> layoutSupportList = {
        "BSND",
        "TND"
    };
    std::string layoutQuery = opParamInfo_.layoutQuery;
    OPS_ERR_IF(std::find(layoutSupportList.begin(), layoutSupportList.end(), layoutQuery) == layoutSupportList.end(),
        OPS_LOG_E(opName_, "layoutQuery only supports BSND/TND, but got %s", layoutQuery.c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus STATilingCheck::CheckFeatureMlaNoQuantDtype() const
{
    OPS_ERR_IF(inputQType_ != ge::DT_BF16 && inputQType_ != ge::DT_FLOAT16,
        OPS_LOG_E(opName_, "query dtype only support %s and %s, but got %s",
            STADataTypeToSerialString(ge::DT_BF16).c_str(), STADataTypeToSerialString(ge::DT_FLOAT16).c_str(),
            STADataTypeToSerialString(inputQType_).c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus STATilingCheck::CheckFeatureMlaNoquantPa() const
{
    if (kvStorageMode_ != KvStorageMode::PAGE_ATTENTION) {
        return ge::GRAPH_SUCCESS;
    }

    OPS_ERR_IF(blockSize_ <= 0 || blockSize_ > static_cast<int32_t>(MAX_BLOCK_SIZE),
        OPS_LOG_E(opName_, "when page attention is enabled, block_size(%ld) should be in range (0, %u].",
        blockSize_, MAX_BLOCK_SIZE), return ge::GRAPH_FAILED);
    
    OPS_ERR_IF(blockSize_ % 16 > 0,
        OPS_LOG_E(opName_, "when page attention is enabled, block_size(%ld) should be 16-aligned.",
        blockSize_), return ge::GRAPH_FAILED);
    
    OPS_ERR_IF(blockSize_ % sparseBlockSize_ > 0,
        OPS_LOG_E(opName_, "when page attention is enabled, block_size(%ld) must be divided by sparse_block_size(%ld), but now the remainder is %ld.",
        blockSize_, sparseBlockSize_, blockSize_ % sparseBlockSize_), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus STATilingCheck::CheckFeatureMlaNoquant() const
{
    if (ge::GRAPH_SUCCESS != CheckFeatureMlaNoQuantShape() ||
        ge::GRAPH_SUCCESS != CheckFeatureMlaNoQuantLayout() ||
        ge::GRAPH_SUCCESS != CheckFeatureMlaNoQuantDtype() ||
        ge::GRAPH_SUCCESS != CheckFeatureMlaNoquantPa()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus STATilingCheck::CheckFeatureMla() const
{
    return CheckFeatureMlaNoquant();
}

ge::graphStatus STATilingCheck::CheckFeature() const
{
    return CheckFeatureMla();
}

ge::graphStatus STATilingCheck::CheckDecodeOnly() const
{
    OPS_ERR_IF(qLayout_ != STALayout::TND || kvLayout_ != STALayout::PA_BSND,
        OPS_LOG_E(opName_, "only layout_query=TND and layout_kv=PA_BSND are supported"),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(sparseBlockSize_ != 1 || *opParamInfo_.sparseMode != 3,
        OPS_LOG_E(opName_, "sparse_block_size must be 1 and sparse_mode must be 3"),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(n1Size_ == 0 || n1Size_ > 128 || n2Size_ != 1 || gSize_ != n1Size_,
        OPS_LOG_E(opName_, "decode requires 1 <= N1 <= 128, N2=1 and group size=N1"),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(qkHeadDim_ != 512 || vHeadDim_ != 512 || ropeHeadDim_ != 64,
        OPS_LOG_E(opName_, "decode requires qk/value head dim 512 and rope head dim 64"),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(blockSize_ != 128,
        OPS_LOG_E(opName_, "PA block size must be 128, but got %ld", blockSize_),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(sparseBlockCount_ != OFFLOAD_SPARSE_INDICES_CAPACITY,
        OPS_LOG_E(opName_, "sparse_indices last dim must be %u, but got %u",
                  OFFLOAD_SPARSE_INDICES_CAPACITY, sparseBlockCount_),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(qTSize_ != bSize_ &&
               qTSize_ != bSize_ * OFFLOAD_MTP3_QUERY_COUNT,
        OPS_LOG_E(opName_, "decode TND query token count must be B or 4B"),
        return ge::GRAPH_FAILED);

    const gert::Shape &cacheTokensShape = opParamInfo_.cacheTokens.shape->GetStorageShape();
    OPS_ERR_IF(cacheTokensShape.GetDimNum() != 1 || cacheTokensShape.GetDim(0) != bSize_,
        OPS_LOG_E(opName_, "cache_tokens must have shape [B]"), return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.cacheTokens.desc->GetDataType() != ge::DT_INT32,
        OPS_LOG_E(opName_, "cache_tokens must be int32"), return ge::GRAPH_FAILED);

    const ge::DataType dtype = opParamInfo_.query.desc->GetDataType();
    OPS_ERR_IF(opParamInfo_.key.desc->GetDataType() != dtype ||
               opParamInfo_.value.desc->GetDataType() != dtype ||
               opParamInfo_.queryRope.desc->GetDataType() != dtype ||
               opParamInfo_.keyRope.desc->GetDataType() != dtype || outputType_ != dtype,
        OPS_LOG_E(opName_, "all floating-point inputs and output must have the same dtype"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

void STATilingCheck::Init()
{
    opName_ = staInfo_.opName;
    platformInfo_ = staInfo_.platformInfo;
    opParamInfo_ = staInfo_.opParamInfo;
    socVersion_ = staInfo_.socVersion;

    bSize_ = staInfo_.bSize;
    n1Size_ = staInfo_.n1Size;
    n2Size_ = staInfo_.n2Size;
    s1Size_ = staInfo_.s1Size;
    s2Size_ = staInfo_.s2Size;
    gSize_ = staInfo_.gSize;
    qkHeadDim_ = staInfo_.qkHeadDim;
    vHeadDim_ = staInfo_.vHeadDim;
    ropeHeadDim_ = staInfo_.ropeHeadDim;
    maxBlockNumPerBatch_ = staInfo_.maxBlockNumPerBatch;
    qTSize_ = staInfo_.qTSize;
    kvTSize_ = staInfo_.kvTSize;
    blockSize_ = staInfo_.blockSize;
    sparseBlockCount_ = staInfo_.sparseBlockCount;
    sparseBlockSize_ = staInfo_.sparseBlockSize;

    inputQType_ = staInfo_.inputQType;
    inputKvType_ = staInfo_.inputKvType;
    inputQRopeType_ = staInfo_.inputQRopeType;
    inputKRopeType_ = staInfo_.inputKRopeType;
    outputType_ = staInfo_.outputType;

    qLayout_ = staInfo_.qLayout;
    topkLayout_ = staInfo_.topkLayout;
    kvLayout_ = staInfo_.kvLayout;
    outLayout_ = staInfo_.outLayout;

    kvStorageMode_ = staInfo_.kvStorageMode;
    l2CacheSize_ = staInfo_.l2CacheSize;
}

ge::graphStatus STATilingCheck::Process()
{
    Init();
    if (CheckDecodeOnly() != ge::GRAPH_SUCCESS ||
        CheckSinglePara() != ge::GRAPH_SUCCESS ||
        CheckParaExistence() != ge::GRAPH_SUCCESS ||
        CheckFeature() != ge::GRAPH_SUCCESS ||
        CheckMultiParaConsistency() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

bool STAInfoParser::HasAxis(const STAAxis &axis, const STALayout &layout, const gert::Shape &shape) const
{   
    const auto& layoutIt = STA_LAYOUT_AXIS_MAP.find(layout);
    if (layoutIt == STA_LAYOUT_AXIS_MAP.end()) {
        return false;
    }

    const std::vector<STAAxis>& axes = layoutIt->second;
    const auto& axisIt = std::find(axes.begin(), axes.end(), axis);
    if (axisIt == axes.end()) {
        return false;
    }
    const auto& dimIt = STA_LAYOUT_DIM_MAP.find(layout);
    if (dimIt == STA_LAYOUT_DIM_MAP.end() || dimIt->second != shape.GetDimNum()) {
        return false;
    }
    return true;
}

size_t STAInfoParser::GetAxisIdx(const STAAxis &axis, const STALayout &layout) const
{
    const std::vector<STAAxis>& axes = STA_LAYOUT_AXIS_MAP.find(layout)->second;
    const auto& axisIt = std::find(axes.begin(), axes.end(), axis);
    return std::distance(axes.begin(), axisIt);
}

uint32_t STAInfoParser::GetAxisNum(const gert::Shape &shape, const STAAxis &axis,const STALayout &layout) const
{
    return HasAxis(axis, layout, shape) ? shape.GetDim(GetAxisIdx(axis, layout)) : invalidDimValue_;
}

ge::graphStatus STAInfoParser::CheckRequiredInOutExistence() const
{
    OPS_ERR_IF(opParamInfo_.query.shape == nullptr, OPS_LOG_E(opName_, "Shape of tensor query is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.query.desc == nullptr, OPS_LOG_E(opName_, "Desc of tensor query is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.key.shape == nullptr, OPS_LOG_E(opName_, "Shape of tensor k is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.key.desc == nullptr, OPS_LOG_E(opName_, "Desc of tensor k is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.value.shape == nullptr, OPS_LOG_E(opName_, "Shape of tensor value is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.value.desc == nullptr, OPS_LOG_E(opName_, "Desc of tensor value is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.sparseIndices.shape == nullptr, OPS_LOG_E(opName_, "Shape of tensor sparseIndices is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.sparseIndices.desc == nullptr, OPS_LOG_E(opName_, "Desc of tensor sparseIndices is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.cacheTokens.shape == nullptr, OPS_LOG_E(opName_, "Shape of tensor cacheTokens is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.cacheTokens.desc == nullptr, OPS_LOG_E(opName_, "Desc of tensor cacheTokens is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.attenOut.shape == nullptr, OPS_LOG_E(opName_, "Shape of tensor output is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.attenOut.desc == nullptr, OPS_LOG_E(opName_, "Desc of tensor output is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.queryRope.tensor == nullptr, OPS_LOG_E(opName_, "Shape of queryRope is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.queryRope.desc == nullptr, OPS_LOG_E(opName_, "Desc of queryRope is nullptr"),
               return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus STAInfoParser::CheckRequiredAttrExistence() const
{
    OPS_ERR_IF(opParamInfo_.layoutQuery == nullptr, OPS_LOG_E(opName_, "attr layoutQuery is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.layoutKV == nullptr, OPS_LOG_E(opName_, "attr layoutKV is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.sparseBlockSize == nullptr, OPS_LOG_E(opName_, "attr sparseBlockSize is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.scaleValue == nullptr, OPS_LOG_E(opName_, "attr scaleValue is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.sparseMode == nullptr, OPS_LOG_E(opName_, "attr sparseMode is nullptr"),
               return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus STAInfoParser::CheckRequiredParaExistence() const
{
    if (CheckRequiredInOutExistence() != ge::GRAPH_SUCCESS ||
        CheckRequiredAttrExistence() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus STAInfoParser::GetActualSeqLenSize(uint32_t &size, const gert::Tensor *tensor,
    STALayout &layout, const std::string &name)
{
    if ((tensor == nullptr)) {
        OPS_LOG_E(opName_, "when layout of query is %s, %s must be provided.",
            STALayoutToSerialString(layout).c_str(), name.c_str());
        return ge::GRAPH_FAILED;
    }
    int64_t shapeSize = tensor->GetShapeSize();
    if (shapeSize <= 0) {
        OPS_LOG_E(opName_, "the shape size of %s is %ld, it should be greater than 0.",
            name.c_str(), shapeSize);
        return ge::GRAPH_FAILED;
    }
    size = static_cast<uint32_t>(shapeSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus STAInfoParser::GetActualSeqLenQSize(uint32_t &size)
{
    return GetActualSeqLenSize(size, opParamInfo_.actualSeqLengthsQ.tensor, qLayout_, "actualSeqLengthsQ");
}

ge::graphStatus STAInfoParser::GetOpName()
{
    if (context_->GetNodeName() == nullptr) {
        OPS_LOG_E("SparseTailAttention", "opName got from TilingContext is nullptr");
        return ge::GRAPH_FAILED;
    }
    opName_ = context_->GetNodeName();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus STAInfoParser::GetNpuInfo()
{
    platformInfo_ = context_->GetPlatformInfo();
    OPS_ERR_IF(platformInfo_ == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(opName_, "GetPlatformInfo is nullptr."), return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo_);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    OPS_ERR_IF(aicNum == 0 || aivNum == 0,
        OPS_REPORT_VECTOR_INNER_ERR(opName_, "num of core obtained is 0."), return GRAPH_FAILED);

    socVersion_ = ascendcPlatform.GetSocVersion();
    if (socVersion_ != platform_ascendc::SocVersion::ASCEND910B &&
        socVersion_ != platform_ascendc::SocVersion::ASCEND910_93) {
        OPS_REPORT_VECTOR_INNER_ERR(opName_, "SOC Version[%d] is not supported.", static_cast<int32_t>(socVersion_));
        return GRAPH_FAILED;
    }

    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, l2CacheSize_);

    return ge::GRAPH_SUCCESS;
}

void STAInfoParser::GetOptionalInputParaInfo()
{
    opParamInfo_.blockTable.tensor = context_->GetInputTensor(BLOCK_TABLE_INPUT_INDEX);
    opParamInfo_.actualSeqLengthsQ.tensor = context_->GetInputTensor(ACT_SEQ_LEN_Q_INPUT_INDEX);
    opParamInfo_.actualSeqLengthsQ.desc = context_->GetInputDesc(ACT_SEQ_LEN_Q_INPUT_INDEX);
    opParamInfo_.actualSeqLengths.tensor = context_->GetInputTensor(ACT_SEQ_LEN_KV_INPUT_INDEX);
    opParamInfo_.actualSeqLengths.desc = context_->GetInputDesc(ACT_SEQ_LEN_KV_INPUT_INDEX);
    opParamInfo_.queryRope.tensor = context_->GetInputTensor(QUERY_ROPE_INPUT_INDEX);
    opParamInfo_.queryRope.desc = context_->GetInputDesc(QUERY_ROPE_INPUT_INDEX);
    opParamInfo_.keyRope.tensor = context_->GetInputTensor(KEY_ROPE_INPUT_INDEX);
    opParamInfo_.keyRope.desc = context_->GetInputDesc(KEY_ROPE_INPUT_INDEX);
}

void STAInfoParser::GetInputParaInfo()
{
    opParamInfo_.query.desc = context_->GetInputDesc(QUERY_INPUT_INDEX);
    opParamInfo_.query.shape = context_->GetInputShape(QUERY_INPUT_INDEX);
    opParamInfo_.key.desc = context_->GetInputDesc(KEY_INPUT_INDEX);
    opParamInfo_.key.shape = context_->GetInputShape(KEY_INPUT_INDEX);
    opParamInfo_.value.desc = context_->GetInputDesc(VALUE_INPUT_INDEX);
    opParamInfo_.value.shape = context_->GetInputShape(VALUE_INPUT_INDEX);
    opParamInfo_.sparseIndices.desc = context_->GetInputDesc(SPARSE_INDICES_INPUT_INDEX);
    opParamInfo_.sparseIndices.shape = context_->GetInputShape(SPARSE_INDICES_INPUT_INDEX);
    opParamInfo_.cacheTokens.desc = context_->GetInputDesc(CACHE_TOKENS_INPUT_INDEX);
    opParamInfo_.cacheTokens.shape = context_->GetInputShape(CACHE_TOKENS_INPUT_INDEX);
    GetOptionalInputParaInfo();
}

void STAInfoParser::GetOutputParaInfo()
{
    opParamInfo_.attenOut.desc = context_->GetOutputDesc(OUTPUT_INDEX);
    opParamInfo_.attenOut.shape = context_->GetOutputShape(OUTPUT_INDEX);
}

ge::graphStatus STAInfoParser::GetAttrParaInfo()
{
    auto attrs = context_->GetAttrs();
    OPS_ERR_IF(attrs == nullptr, OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "attrs got from ge is nullptr"),
               return ge::GRAPH_FAILED);

    opParamInfo_.layoutQuery = attrs->GetStr(LAYOUT_QUERY_ATTR_INDEX);
    opParamInfo_.layoutKV = attrs->GetStr(LAYOUT_KV_ATTR_INDEX);
    opParamInfo_.sparseBlockSize = attrs->GetAttrPointer<int64_t>(SPARSE_BLOCK_SIZE_ATTR_INDEX);
    opParamInfo_.scaleValue = attrs->GetAttrPointer<float>(SCALE_VALUE_ATTR_INDEX);
    opParamInfo_.sparseMode = attrs->GetAttrPointer<int64_t>(SPARSE_MODE_ATTR_INDEX);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus STAInfoParser::GetOpParaInfo()
{
    GetInputParaInfo();
    GetOutputParaInfo();
    if (ge::GRAPH_SUCCESS != GetAttrParaInfo()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus STAInfoParser::GetInOutDataType()
{
    inputQType_ = opParamInfo_.query.desc->GetDataType();
    inputKvType_ = opParamInfo_.key.desc->GetDataType();
    outputType_ = opParamInfo_.attenOut.desc->GetDataType();
    if (opParamInfo_.queryRope.desc != nullptr) {
        inputQRopeType_ = opParamInfo_.queryRope.desc->GetDataType();
    }
    if (opParamInfo_.keyRope.desc != nullptr) {
        inputKRopeType_ = opParamInfo_.keyRope.desc->GetDataType();
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus STAInfoParser::GetBatchSize()
{
    if (qLayout_ == STALayout::TND) {
        return GetActualSeqLenQSize(bSize_);
    } else { // BSND
        bSize_ = GetAxisNum(queryShape_, STAAxis::B, qLayout_);
        return ge::GRAPH_SUCCESS;
    }
}

ge::graphStatus STAInfoParser::GetQTSize()
{
    qTSize_ = (qLayout_ == STALayout::TND) ? GetAxisNum(queryShape_, STAAxis::T, qLayout_) : 0;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus STAInfoParser::GetKVTSize()
{
    kvTSize_ = (kvLayout_ == STALayout::TND) ? GetAxisNum(keyShape_, STAAxis::T, kvLayout_) : 0;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus STAInfoParser::GetQkHeadDim()
{
    qkHeadDim_ = GetAxisNum(queryShape_, STAAxis::D, qLayout_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus STAInfoParser::GetS1Size()
{
    if (qLayout_ == STALayout::TND) {
        s1Size_ = GetAxisNum(queryShape_, STAAxis::T, qLayout_);
        return ge::GRAPH_SUCCESS;
    } else { // BSND
        s1Size_ = GetAxisNum(queryShape_, STAAxis::S, qLayout_);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus STAInfoParser::GetKvStorageMode()
{
    if (kvLayout_ == STALayout::PA_BSND) {
        kvStorageMode_ = KvStorageMode::PAGE_ATTENTION;
    } else {
        kvStorageMode_ = KvStorageMode::BATCH_CONTINUOUS;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus STAInfoParser::GetKvLayout()
{
    const map<string, STALayout> layoutKVMap = {
        {"BSND",        STALayout::BSND},
        {"PA_BSND",     STALayout::PA_BSND},
        {"TND",         STALayout::TND}
    };

    std::string layout(opParamInfo_.layoutKV);
    auto it = layoutKVMap.find(layout);
    if (it != layoutKVMap.end()) {
        kvLayout_ = it->second;
    } else {
        OPS_LOG_E(opName_, "layoutKV is %s, it is unsupported.", layout.c_str());
        return ge::GRAPH_FAILED;
    }
    if (kvLayout_ != STALayout::PA_BSND && qLayout_ != kvLayout_) {
        OPS_LOG_E(opName_, "When layoutKV is not PA_BSND, layoutKV must be the same as layoutQ.");
        return ge::GRAPH_FAILED;
    }
    uint32_t keyDimNum = opParamInfo_.key.shape->GetStorageShape().GetDimNum();
    if (kvLayout_ == STALayout::PA_BSND && keyDimNum != 4U) {
        OPS_LOG_E(opName_, "When layoutKV is PA_BSND, kvDimNum must be 4, but now is %d.", keyDimNum);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus STAInfoParser::GetS2SizeForBatchContinuous()
{
    if (kvLayout_ == STALayout::BSND) { // BSND
        s2Size_ = GetAxisNum(keyShape_, STAAxis::S, kvLayout_);
    } else if (kvLayout_ == STALayout::TND) {
        s2Size_ = GetAxisNum(keyShape_, STAAxis::T, kvLayout_);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus STAInfoParser::GetMaxBlockNumPerBatch()
{
    if (opParamInfo_.blockTable.tensor == nullptr) {
        OPS_LOG_E(opName_, "the layout_kv is %s, blockTable must be provided.", STALayoutToSerialString(kvLayout_).c_str());
        return ge::GRAPH_FAILED;
    }
    uint32_t dimNum = opParamInfo_.blockTable.tensor->GetStorageShape().GetDimNum();
    if (dimNum != DIM_NUM_TWO) {
        OPS_LOG_E(opName_, "the dim num of block_table is %u, it should be %u.", dimNum, DIM_NUM_TWO);
        return ge::GRAPH_FAILED;
    }
    if (opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(1) <= 0) {
        OPS_LOG_E(opName_, "%s's second dimension(%ld) should be greater than 0",
            BLOCK_TABLE_NAME.c_str(), opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(1));
        return ge::GRAPH_FAILED;
    }
    maxBlockNumPerBatch_ = opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus STAInfoParser::GetBlockSize()
{
    blockSize_ = GetAxisNum(keyShape_, STAAxis::Bs, kvLayout_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus STAInfoParser::GetSparseBlockCount()
{
    sparseBlockCount_ = GetAxisNum(sparseIndicesShape_, STAAxis::K, qLayout_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus STAInfoParser::GetS2SizeForPageAttention()
{
    if (GetMaxBlockNumPerBatch() != ge::GRAPH_SUCCESS || GetBlockSize() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    s2Size_ = maxBlockNumPerBatch_ * blockSize_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus STAInfoParser::GetS2Size()
{
    if (kvStorageMode_ == KvStorageMode::BATCH_CONTINUOUS) {
        return GetS2SizeForBatchContinuous();
    }
    return GetS2SizeForPageAttention();
}

ge::graphStatus STAInfoParser::GetValueHeadDim()
{
    vHeadDim_ = GetAxisNum(valueShape_, STAAxis::D, kvLayout_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus STAInfoParser::GetRopeHeadDim()
{
    ropeHeadDim_ = GetAxisNum(queryRopeShape_, STAAxis::D, qLayout_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus STAInfoParser::GetQueryAndOutLayout()
{
    const map<string, pair<STALayout, STALayout>> layoutMap = {
        {"BSND",        {STALayout::BSND,    STALayout::BSND}},
        {"TND",         {STALayout::TND,     STALayout::TND }},
    };

    std::string layout(opParamInfo_.layoutQuery);
    auto it = layoutMap.find(layout);
    if (it != layoutMap.end()) {
        qLayout_ = it->second.first;
        outLayout_ = it->second.second;
    } else {
        OPS_LOG_E(opName_, "layoutQuery is %s, it is unsupported.", layout.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus STAInfoParser::GetTopkLayout()
{
    topkLayout_ = qLayout_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus STAInfoParser::GetN1Size()
{
    n1Size_ = GetAxisNum(queryShape_, STAAxis::N, qLayout_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus STAInfoParser::GetN2Size()
{
    n2Size_ = GetAxisNum(keyShape_, STAAxis::N, kvLayout_);
    return ge::GRAPH_SUCCESS;
}

void STAInfoParser::SetSTAShape()
{
    queryShape_ = opParamInfo_.query.shape->GetStorageShape();
    keyShape_ = opParamInfo_.key.shape->GetStorageShape();
    valueShape_ = opParamInfo_.value.shape->GetStorageShape();
    sparseIndicesShape_ = opParamInfo_.sparseIndices.shape->GetStorageShape();
    queryRopeShape_ = opParamInfo_.queryRope.tensor->GetStorageShape();
}

ge::graphStatus STAInfoParser::GetGSize()
{
    if (n2Size_ != 0) {
        gSize_ = n1Size_ / n2Size_;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus STAInfoParser::GetActualseqInfo()
{
    maxActualseq_ = static_cast<uint32_t>(s2Size_);
    if (opParamInfo_.actualSeqLengths.tensor != nullptr) {
        actualLenDimsKV_ = opParamInfo_.actualSeqLengths.tensor->GetShapeSize();
    }
    if (opParamInfo_.actualSeqLengthsQ.tensor != nullptr) {
        actualLenDimsQ_ = opParamInfo_.actualSeqLengthsQ.tensor->GetShapeSize();
    }
    return ge::GRAPH_SUCCESS;
}

void STAInfoParser::GenerateInfo(STATilingInfo &staInfo)
{
    staInfo.opName = opName_;
    staInfo.platformInfo = platformInfo_;
    staInfo.opParamInfo = opParamInfo_;
    staInfo.socVersion = socVersion_;

    staInfo.bSize = bSize_;
    staInfo.n1Size = n1Size_;
    staInfo.n2Size = n2Size_;
    staInfo.s1Size = s1Size_;
    staInfo.s2Size = s2Size_;
    staInfo.gSize = gSize_;
    staInfo.qkHeadDim = qkHeadDim_;
    staInfo.vHeadDim = vHeadDim_;
    staInfo.ropeHeadDim = ropeHeadDim_;
    staInfo.qTSize = qTSize_;
    staInfo.kvTSize = kvTSize_;
    staInfo.sparseBlockSize = *opParamInfo_.sparseBlockSize;
    staInfo.sparseBlockCount = sparseBlockCount_;

    staInfo.inputQType = inputQType_;
    staInfo.inputKvType = inputKvType_;
    staInfo.inputQRopeType = inputQRopeType_;
    staInfo.inputKRopeType = inputKRopeType_;
    staInfo.outputType = outputType_;

    staInfo.kvStorageMode = kvStorageMode_;
    staInfo.l2CacheSize = l2CacheSize_;

    staInfo.totalBlockNum = opParamInfo_.key.shape->GetStorageShape().GetDim(0);
    staInfo.scaleValue = *opParamInfo_.scaleValue;
    staInfo.pageAttentionFlag = (kvStorageMode_ == KvStorageMode::PAGE_ATTENTION);
    staInfo.blockSize = blockSize_;
    staInfo.blockTypeSize =  sizeof(float);
    staInfo.maxBlockNumPerBatch = maxBlockNumPerBatch_;

    staInfo.actualLenDimsQ = actualLenDimsQ_;
    staInfo.actualLenDimsKV = actualLenDimsKV_;
    staInfo.maxActualseq = maxActualseq_;
    staInfo.actualSeqLenFlag = (opParamInfo_.actualSeqLengths.tensor != nullptr);
    staInfo.isSameSeqAllKVTensor = isSameSeqAllKVTensor_;
    staInfo.isSameActualseq = isSameActualseq_;

    staInfo.sparseMode = *opParamInfo_.sparseMode;

    staInfo.qLayout = qLayout_;
    staInfo.topkLayout = topkLayout_;
    staInfo.kvLayout = kvLayout_;
    staInfo.outLayout = outLayout_;
}

ge::graphStatus STAInfoParser::Parse(STATilingInfo &staInfo)
{
    if (context_ == nullptr) {
        OPS_LOG_E("SparseTailAttention", "tiling context is nullptr!");
        return ge::GRAPH_FAILED;
    }
    OPS_LOG_FULL(DLOG_INFO, "SparseTailAttention", "TilingContext: %s", STADebugTilingContext(context_).c_str());
    if (ge::GRAPH_SUCCESS != GetOpName() ||
        ge::GRAPH_SUCCESS != GetNpuInfo() ||
        ge::GRAPH_SUCCESS != GetOpParaInfo() ||
        ge::GRAPH_SUCCESS != CheckRequiredParaExistence()) {
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != GetInOutDataType() ||
        ge::GRAPH_SUCCESS != GetQueryAndOutLayout() ||
        ge::GRAPH_SUCCESS != GetTopkLayout() ||
        ge::GRAPH_SUCCESS != GetKvLayout() ||
        ge::GRAPH_SUCCESS != GetKvStorageMode()) {
        return ge::GRAPH_FAILED;
    }

    SetSTAShape();
    if (
        ge::GRAPH_SUCCESS != GetN1Size() ||
        ge::GRAPH_SUCCESS != GetN2Size() ||
        ge::GRAPH_SUCCESS != GetGSize() ||
        ge::GRAPH_SUCCESS != GetBatchSize() ||
        ge::GRAPH_SUCCESS != GetQTSize() ||
        ge::GRAPH_SUCCESS != GetKVTSize() ||
        ge::GRAPH_SUCCESS != GetS1Size() ||
        ge::GRAPH_SUCCESS != GetQkHeadDim() ||
        ge::GRAPH_SUCCESS != GetS2Size() ||
        ge::GRAPH_SUCCESS != GetValueHeadDim() ||
        ge::GRAPH_SUCCESS != GetRopeHeadDim() ||
        ge::GRAPH_SUCCESS != GetSparseBlockCount()) {
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != GetActualseqInfo()) {
        return ge::GRAPH_FAILED;
    }

    GenerateInfo(staInfo);
    return ge::GRAPH_SUCCESS;
}


IMPL_OP_OPTILING(SparseTailAttention)
    .Tiling(TilingSparseTailAttention)
    .TilingParse<SparseTailAttentionCompileInfo>(TilingPrepareForSparseTailAttention);

IMPL_OP_OPTILING(SparseTailAttentionMtp)
    .Tiling(TilingSparseTailAttentionMtp)
    .TilingParse<SparseTailAttentionCompileInfo>(TilingPrepareForSparseTailAttention);
} // namespace optiling
