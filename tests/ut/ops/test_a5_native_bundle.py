# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host packaging/argument contracts; no library compilation or device launch."""

import ctypes
import hashlib
import json
import struct

import pytest
import torch

from vllm_ascend.ops.a5_bundle_manifest import METADATA_FIELDS, OPERATOR_MODULES, REQUIRED_HOST_FILES, REQUIRED_PROFILES
from vllm_ascend.ops.a5_native_bundle import A5NativeBundle, HostSchema, HostSlot, NativeKernel


def frame_kernel():
    # One real strided tensor and an INT64 scalar exercise both descriptor
    # fields and the width that differs from the MQ INT32 scalar contract.
    slots = (HostSlot * 2)()
    tensor, scalar = slots
    tensor.kind, tensor.dtype_code, tensor.dtype_bits = 1, 2, 32
    tensor.dtype_lanes = 1
    tensor.logical_rank = tensor.physical_rank = 2
    tensor.offset, tensor.size, tensor.alignment = 56, 40, 8
    tensor.pointer_offset, tensor.shape_offset, tensor.stride_offset = 0, 8, 24
    tensor.physical_shape_offset, tensor.physical_stride_offset = 8, 24
    scalar.kind, scalar.dtype_code, scalar.dtype_bits = 2, 0, 64
    scalar.dtype_lanes = 1
    scalar.offset, scalar.size, scalar.alignment = 96, 8, 8
    schema = HostSchema()
    schema.slot_count, schema.frame_bytes, schema.frame_alignment = 2, 104, 8
    schema.slots = slots
    kernel = NativeKernel.__new__(NativeKernel)
    kernel.schema = schema
    return kernel, slots


def test_native_frame_preserves_stride_alignment_and_int64():
    kernel, slots = frame_kernel()
    query = torch.arange(12, dtype=torch.float32).reshape(4, 3)[::2]
    large_scalar = 1 << 40
    raw, frame = kernel.frame((query, large_scalar))
    assert ctypes.addressof(frame) % 8 == 0
    assert struct.unpack_from("<Q", frame, 56)[0] == query.data_ptr()
    assert struct.unpack_from("<qq", frame, 64) == (2, 3)
    assert struct.unpack_from("<qq", frame, 80) == (6, 1)
    assert struct.unpack_from("<q", frame, 96)[0] == large_scalar
    _, second = kernel.frame((query, 17))
    assert ctypes.addressof(second) != ctypes.addressof(frame)
    assert struct.unpack_from("<q", frame, 96)[0] == large_scalar
    assert struct.unpack_from("<q", second, 96)[0] == 17


def test_native_frame_rejects_dtype_count_and_device_scalar_coercion():
    kernel, slots = frame_kernel()
    with pytest.raises(TypeError, match="dtype mismatch"):
        kernel.frame((torch.ones(2, 3, dtype=torch.int32), 1))
    with pytest.raises(ValueError, match="arguments"):
        kernel.frame((torch.ones(2, 3),))
    with pytest.raises(TypeError, match="device tensors"):
        kernel.frame((torch.ones(2, 3), torch.tensor(1)))


def write_bundle(tmp_path, status="READY"):
    files = {}
    # These are explicit host test doubles; no compiled fixture library is
    # loaded. Native arithmetic and hardware ABI acceptance remain untested.
    source = "".join(
        f"def {name}(*args, **kwargs):\n    return 'compute'\n"
        f"def {name}_metadata(*args, **kwargs):\n    return 'metadata'\n"
        for name in OPERATOR_MODULES
    )
    for name in REQUIRED_HOST_FILES:
        path = tmp_path / name
        path.write_text(source)
        files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ("compute.so", "qli-md.so", "qsli-md.so", "mq-md.so"):
        path = tmp_path / name
        path.write_bytes(b"unused library test double")
        files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "version": 1,
        "status": status,
        "cube_core_num": 32,
        "files": files,
        "operators": OPERATOR_MODULES,
        "kernels": [
            {
                "op": op,
                "key": key,
                "library": "compute.so",
                "symbol": "fixture",
                "contract_digest": "0" * 64,
                "slot_count": 1,
                "frame_bytes": 64,
            }
            for op, key in REQUIRED_PROFILES
        ],
        "metadata": {
            op: {
                "library": f"{op}-md.so",
                "symbol": "fixture_metadata",
                "fields": [{"name": name, "ctype": kind} for name, kind in fields],
            }
            for op, fields in METADATA_FIELDS.items()
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path / "mq.py"


def test_bundle_loads_host_api_without_a_compiler_and_rejects_missing_profile(tmp_path):
    write_bundle(tmp_path)
    bundle = A5NativeBundle(str(tmp_path))
    compute, metadata = bundle.get_ops("mixed_quant_sparse_flash_mla")
    assert compute() == "compute" and metadata() == "metadata"
    assert bundle.get_ops("mixed_quant_sparse_flash_mla") == (compute, metadata)
    assert "head_partition.py" in bundle.snapshot()["modules"]
    query = torch.empty(1, 128, 512, dtype=torch.bfloat16)
    assert compute(query, quant_mode=1, layout_q="TND", layout_kv="PA_BBND") == "compute"
    public = bundle.snapshot()["public_apis"]["mixed_quant_sparse_flash_mla"]
    assert public["query_shape"] == (1, 128, 512)
    assert public["calls_returned"] == 2
    with pytest.raises(NotImplementedError, match="No precompiled"):
        bundle.kernel("mq", [True, False, False, False, "off"])


def test_bundle_rejects_missing_model_profile_before_loading_any_library(tmp_path, monkeypatch):
    write_bundle(tmp_path)
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["kernels"].pop()
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(ctypes, "CDLL", lambda *args: pytest.fail("incomplete bundle loaded a library"))
    with pytest.raises(ValueError, match="five required"):
        A5NativeBundle(str(tmp_path))


def test_bundle_rejects_swapped_metadata_contract_before_host_import(tmp_path):
    write_bundle(tmp_path)
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["metadata"]["qli"]["fields"] = manifest["metadata"]["qsli"]["fields"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="field order/width"):
        A5NativeBundle(str(tmp_path))


def test_bundle_rejects_unassembled_or_mutated_sources(tmp_path):
    host = write_bundle(tmp_path, status="DRAFT")
    with pytest.raises(ValueError, match="READY"):
        A5NativeBundle(str(tmp_path))
    write_bundle(tmp_path)
    host.write_text("raise RuntimeError('mutated source must not execute')\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        A5NativeBundle(str(tmp_path))


def test_bundle_selection_does_not_fall_back_to_external_package(monkeypatch):
    from vllm_ascend import envs
    from vllm_ascend.ops import packaged_attention

    packaged_attention._get_dsl_ops.cache_clear()
    monkeypatch.setattr(envs, "VLLM_ASCEND_A5_ATTENTION_BUNDLE", "/task/private/bundle")
    monkeypatch.setattr(packaged_attention, "import_module", lambda *args: pytest.fail("unexpected fallback"))

    def missing_bundle(directory):
        raise ValueError("incomplete native bundle")

    monkeypatch.setattr(packaged_attention, "_get_native_bundle", missing_bundle)
    with pytest.raises(ValueError, match="incomplete native bundle"):
        packaged_attention._get_dsl_ops("quant_lightning_indexer")
    packaged_attention._get_dsl_ops.cache_clear()
