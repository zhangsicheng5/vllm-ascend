# fused_li_manage_mtp_c8 单算子精度验证

量化 ABI（query/index_key_cache int8 + fp16 dequant scale、index_weights bf16、
miss 输出容量 [B, 32768]）的上板精度套件。

## 文件

| 文件 | 角色 |
|---|---|
| `src_test_quant.py` | **主入口**：6 模式 54 场景（correctness / replacement / first_decode / lifecycle / steady_strict / long），源算子 UT 矩阵的量化移植 |
| `fused_li_manage_mtp_c8_quant_full.py` | 断言引擎与运行器（mgmt/lifecycle/warmfix/boundary/c0/invalid/qsweep/keytag/window 模式 + 六算子 `--only matrix` 对拍） |
| `fused_li_manage_mtp_c8_src_test.py` | 场景构造（build_src_case / STRICT_SCENARIOS / 受控 union / 随机 block_table） |
| `fused_li_manage_mtp_c8_quant_test.py` | int8 动态量化 + 量化闭式 golden（逐位复刻核内 fp16 舍入点）+ 重合率阈值 |
| `fused_li_manage_mtp_c8_test.py` | golden 基础实现（golden_score / golden_topk）；bf16 ABI 时代的 8-case 基线，**直接运行会因 int8 dtype 检查失败**，现仅作库被上方脚本引用 |
| `fused_li_manage_mtp_c8.json` | bf16 时代 8-case 配置（供 `fused_li_manage_mtp_c8_test.py`） |

## 判定口径

1. mgmt 段逐元素精确断言（由 kernel 自身 topk 驱动）：miss_counts / miss_src 列表序 /
   miss_dst 互异+落点 / 池双射与槽位守恒 / 未参与行零改写 / 同池幂等不动点；
2. `B/A` 与 golden(原始 bf16) topk 集合重合率 ≥ 98.5%；
3. `B/C` 与 golden(反量化重建) 重合率 ≥ 99.5%；
4. tie-free 时与闭式量化 golden 的 topk **全序逐位一致**。

## 运行（先构建并安装算子包：`python setup.py build_ext --inplace`）

```bash
REPO=<仓库根目录>
export PYTHONPATH=$REPO
export ASCEND_CUSTOM_OPP_PATH=$REPO/vllm_ascend/_cann_ops_custom/vendors/custom_transformer
export LD_LIBRARY_PATH=$ASCEND_CUSTOM_OPP_PATH/op_api/lib:$LD_LIBRARY_PATH
export ASCEND_RT_VISIBLE_DEVICES=0

python3 src_test_quant.py                            # 54 场景主套件
python3 fused_li_manage_mtp_c8_quant_full.py         # 全部模式
python3 fused_li_manage_mtp_c8_quant_full.py --only matrix   # 六算子对拍
```

六算子矩阵：A=golden(bf16)　B=本算子(int8)　C=golden(dequant)　D=源
`fused_li_manage_mtp`(bf16，需 `NANOVLLM_CUST_OPAPI_LIB` 双算子隔离环境加载
nanovllm 侧 `libcust_opapi.so`，缺省自动 SKIP)　E=官方 `npu_lightning_indexer`
　F=官方 `lightning_indexer_quant`。D 需为 miss 容量 32768 的新接口版本
（ops_lim_standardization 之后），旧 16384 接口的 D 会在 host 校验被拒。
