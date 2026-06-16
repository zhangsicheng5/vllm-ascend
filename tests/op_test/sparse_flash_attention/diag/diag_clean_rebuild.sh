#!/usr/bin/env bash
# 干净重建 vllm-ascend SparseFlashAttention 的两层产物，避免"脏环境"段错误。
#
# 背景：本算子有两层独立产物，build_aclnn.sh 只自清 csrc/build 和 csrc/output，
#       已安装的 vendors/ 是覆盖式安装、不会删旧文件。改了 def 接口后若残留旧的
#       aclnn wrapper / 旧 .so，两层不同步 → EXEC_NPU_CMD 调到旧签名 → segfault。
#
# 用法（在 vllm-ascend 仓库根目录跑，或任意目录——脚本会自动 cd 到 git 根）：
#   bash tests/op_test/sparse_flash_attention/diag/diag_clean_rebuild.sh <soc> [mode]
#     <soc>  : ascend910b / ascend910_93 等（层1构建必填）
#     mode   : all (默认) | layer1 | layer2
#              layer1 = 只重建 CANN 算子包（改了 def/proto/tiling/op_kernel 时）
#              layer2 = 只重建 Python 扩展（改了 torch_binding*/torch_adpt 时）
#              all    = 两层都重建（改了 def 接口时必须，如 step 4a）
#
# 示例：
#   bash .../diag/diag_clean_rebuild.sh ascend910b           # 全清重建
#   bash .../diag/diag_clean_rebuild.sh ascend910b layer1    # 只重建算子包
#   bash .../diag/diag_clean_rebuild.sh "" layer2            # 只重建 Python 扩展
#
# 把输出 tee 到日志方便回贴：
#   bash .../diag/diag_clean_rebuild.sh ascend910b 2>&1 | tee /tmp/diag_rebuild.log

set -u

SOC="${1:-}"
MODE="${2:-all}"

# 定位仓库根目录
ROOT_DIR="$(git rev-parse --show-toplevel 2>/dev/null)"
if [[ -z "$ROOT_DIR" ]]; then
    echo "[FATAL] 不在 git 仓库内，无法定位仓库根目录" >&2
    exit 1
fi
cd "$ROOT_DIR" || exit 1
echo "=== 仓库根: $ROOT_DIR ==="
echo "=== 模式: $MODE   SOC: ${SOC:-<未提供>} ==="
echo

rebuild_layer1() {
    if [[ -z "$SOC" ]]; then
        echo "[FATAL] 重建层1（CANN 算子包）需要 <soc> 参数，例如 ascend910b" >&2
        exit 1
    fi
    # build_aclnn.sh 只认 ^ascend910b / ^ascend910_93，其它值会静默 exit 0 什么都不建。
    # 这里先拦住，避免 rm vendors 之后又空跑，导致 "binary bin not found"。
    if [[ ! "$SOC" =~ ^ascend910b && ! "$SOC" =~ ^ascend910_93 ]]; then
        echo "[FATAL] soc='$SOC' 不被 build_aclnn.sh 支持（只认 ascend910b* / ascend910_93*）。" >&2
        echo "        传错 soc 会让 build_aclnn.sh 静默 exit 0、不构建任何算子。" >&2
        exit 1
    fi
    echo "=== [层1] 清理 CANN 算子包产物 ==="
    echo "    删除: csrc/build  csrc/output  vllm_ascend/_cann_ops_custom/vendors"
    rm -rf csrc/build csrc/output vllm_ascend/_cann_ops_custom/vendors
    echo

    echo "=== [层1] 构建 + 安装 CANN 算子包（soc=$SOC） ==="
    bash csrc/build_aclnn.sh "$ROOT_DIR" "$SOC"
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "[FATAL] build_aclnn.sh 失败（rc=$rc）。常见原因：def 改了输出但 kernel 入口签名" >&2
        echo "        没同步 → binary gen 报 template mismatch / no matching function。" >&2
        exit $rc
    fi
    # 关键校验：build_aclnn.sh 可能 rc=0 却没产出（soc 静默 exit 0，或 kernel binary gen
    # 失败但未传播）。这里确认 sparse_flash_attention 的产物真的进了 vendors，否则运行期
    # 会报 "binary bin not found"。
    local sfa_artifacts
    sfa_artifacts=$(find vllm_ascend/_cann_ops_custom/vendors -iname "*sparse_flash_attention*" 2>/dev/null | head -5)
    if [[ -z "$sfa_artifacts" ]]; then
        echo "[FATAL] 层1 build 后 vendors 里找不到 sparse_flash_attention 产物！" >&2
        echo "        说明算子包没真正构建（soc 空跑 / kernel binary gen 失败）。" >&2
        echo "        请检查上面的 build 日志里 sparse_flash_attention 段是否有 error / no matching function。" >&2
        exit 1
    fi
    echo "=== [层1] 完成 ==="
    echo "    sparse_flash_attention 产物（节选）:"
    echo "$sfa_artifacts"
    echo
}

rebuild_layer2() {
    echo "=== [层2] 清理 Python 扩展产物 ==="
    echo "    删除: build  vllm_ascend.egg-info  _C_ascend*.so  __pycache__"
    rm -rf build vllm_ascend.egg-info
    find . -name "_C_ascend*.so" -delete 2>/dev/null
    find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
    echo

    echo "=== [层2] 安装 Python 扩展（--no-build-isolation 避免重装 torch 卡死） ==="
    pip install -e . --no-build-isolation --no-deps --force-reinstall
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "[FATAL] pip install 失败（rc=$rc）" >&2
        exit $rc
    fi
    echo "=== [层2] 完成 ==="
    find . -name "_C_ascend*.so" 2>/dev/null
    echo
}

case "$MODE" in
    all)
        rebuild_layer1
        rebuild_layer2
        ;;
    layer1)
        rebuild_layer1
        ;;
    layer2)
        rebuild_layer2
        ;;
    *)
        echo "[FATAL] 未知 mode: $MODE （应为 all | layer1 | layer2）" >&2
        exit 1
        ;;
esac

# [关键] 安装完 CANN 算子包后，必须 source set_env.bash 才能让运行期找到 kernel 二进制，
# 否则执行算子报 EZ9999 AclNN_Inner_Error: "The binary bin not found!"（官方解释为
# "检查环境变量是否正确"）。注意：脚本里 source 只对本脚本 shell 生效，退出后不保留，
# 所以下面既在脚本内 source（让本脚本的验证可用），也在结尾打印命令提醒你在自己的 shell 再跑一次。
# vendor 名可能不是 custom_transformer，用 glob 兜底。
SET_ENV="$(ls "$ROOT_DIR"/vllm_ascend/_cann_ops_custom/vendors/*/bin/set_env.bash 2>/dev/null | head -1)"
if [[ -n "$SET_ENV" && -f "$SET_ENV" ]]; then
    echo "=== source set_env.bash（注册算子二进制查找路径） ==="
    # shellcheck disable=SC1090
    source "$SET_ENV"
    echo "    source: $SET_ENV"
    echo
else
    echo "[WARN] 未找到 vendors/*/bin/set_env.bash（layer2-only 模式且从未建过算子包？）" >&2
    echo
fi

echo "=== 验证：算子是否重新注册 ==="
# 注意：必须 enable_custom_op() 才会真正加载/注册自定义算子，
# 光 import vllm_ascend 不会，hasattr 会是 False（误判）。
python -c "
import torch
import torch_npu  # noqa: registers npu device
import vllm_ascend  # noqa
from vllm_ascend.utils import enable_custom_op
enable_custom_op()
print('npu_sparse_flash_attention registered:', hasattr(torch.ops._C_ascend, 'npu_sparse_flash_attention'))
"
echo
echo "################################################################################"
echo "# 跑回归前，必须先在你当前的 shell 里 source 环境变量（脚本里的 source 不会留到你的 shell）："
echo "#"
echo "#   source $SET_ENV"
echo "#"
echo "# 然后跑离散 probe / 测试："
echo "#   python3 tests/op_test/sparse_flash_attention/diag/probe_discrete_sfa.py"
echo "#   pytest -s -v tests/ut/attention/a2/test_sfa_discrete_indices.py"
echo "################################################################################"
