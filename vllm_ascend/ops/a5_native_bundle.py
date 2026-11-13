# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load precompiled A5 attention artifacts without a DSL or runtime compiler.

The bundle owns immutable host sources and libraries; this worker owns loaded
modules and ctypes handles. A missing exact profile is an error. Importing this
module or inspecting a bundle does not initialize a device or launch a kernel.
"""

from __future__ import annotations

import ctypes
import hashlib
import importlib.util
import json
import os
import struct
import sys
import threading
from functools import wraps
from pathlib import Path
from types import ModuleType, SimpleNamespace

import torch

from vllm_ascend.ops import a5_bundle_manifest as bundle_contract

SCHEMA_MAGIC = 0x43425332
FRAME_MAGIC = 0x43424639
HOST_ABI_VERSION = 9
SCHEMA_VERSION = 2
FRAME_HEADER_BYTES = 56
SUPPORTED_OPERATORS = (
    "quant_lightning_indexer",
    "quant_sparse_lightning_indexer",
    "mixed_quant_sparse_flash_mla",
)
TENSOR_DTYPES = {
    "torch.int8": (0, 8),
    "torch.int32": (0, 32),
    "torch.int64": (0, 64),
    "torch.uint8": (1, 8),
    "torch.uint16": (1, 16),
    "torch.float32": (2, 32),
    "torch.bfloat16": (4, 16),
    "torch.float8_e8m0fnu": (14, 8),
}
SCALAR_FORMATS = {(0, 32): "i", (0, 64): "q", (1, 32): "I", (1, 64): "Q", (2, 32): "f"}
METADATA_CTYPES = {
    "c_uint32": ctypes.c_uint32,
    "c_int32": ctypes.c_int32,
    "c_uint64": ctypes.c_uint64,
    "c_int64": ctypes.c_int64,
}


class HostSlot(ctypes.Structure):
    _fields_ = (
        [
            (name, ctypes.c_uint32)
            for name in (
                "kind",
                "flags",
                "dtype_code",
                "dtype_bits",
                "dtype_lanes",
                "encoding",
                "logical_rank",
                "physical_rank",
            )
        ]
        + [
            (name, ctypes.c_uint64)
            for name in (
                "offset",
                "size",
                "alignment",
                "pointer_offset",
                "shape_offset",
                "stride_offset",
                "physical_shape_offset",
                "physical_stride_offset",
            )
        ]
        + [("abi_type_digest", ctypes.c_uint8 * 32)]
    )


class HostSchema(ctypes.Structure):
    _fields_ = [
        ("magic", ctypes.c_uint32),
        ("host_abi_version", ctypes.c_uint16),
        ("schema_version", ctypes.c_uint16),
        ("slot_count", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("frame_bytes", ctypes.c_uint64),
        ("frame_alignment", ctypes.c_uint64),
        ("contract_digest", ctypes.c_uint8 * 32),
        ("slots", ctypes.POINTER(HostSlot)),
    ]


def _key(values) -> str:
    return json.dumps(values, separators=(",", ":"), allow_nan=False)


def current_raw_stream(device=None) -> int:
    return int(torch.npu.current_stream(device).npu_stream)


class NativeKernel:
    """A reentrant ABI9/schema2 launcher using the caller's current stream."""

    def __init__(self, path: Path, record: dict):
        self.library = ctypes.CDLL(str(path))
        self.path = str(path)
        self.symbol = record["symbol"]
        self.launch_count = 0
        self.last_stream = None
        self._counter_lock = threading.Lock()
        symbol = record["symbol"]
        query = getattr(self.library, symbol + "__cannbot_schema_v2")
        query.argtypes = []
        query.restype = ctypes.POINTER(HostSchema)
        schema_ptr = query()
        if not schema_ptr:
            raise ValueError(f"Null native schema: {path}")
        self.schema = schema_ptr.contents
        schema = self.schema
        if (schema.magic, schema.host_abi_version, schema.schema_version, schema.reserved) != (
            SCHEMA_MAGIC,
            HOST_ABI_VERSION,
            SCHEMA_VERSION,
            0,
        ):
            raise ValueError(f"Unsupported native schema: {path}")
        actual = (bytes(schema.contract_digest).hex(), schema.slot_count, schema.frame_bytes)
        expected = (record["contract_digest"], record["slot_count"], record["frame_bytes"])
        if actual != expected:
            raise ValueError(f"Native artifact/schema mismatch: {path}")
        if (
            schema.frame_bytes < FRAME_HEADER_BYTES
            or schema.frame_alignment == 0
            or schema.frame_alignment & (schema.frame_alignment - 1)
        ):
            raise ValueError("Invalid native frame geometry")
        for index in range(schema.slot_count):
            slot = schema.slots[index]
            if slot.offset < FRAME_HEADER_BYTES or slot.offset + slot.size > schema.frame_bytes:
                raise ValueError(f"Native slot {index} lies outside its frame")
            if slot.kind not in (1, 2) or slot.encoding != 0 or slot.dtype_lanes != 1:
                raise ValueError(f"Unsupported native slot {index}")
        self.invoke = getattr(self.library, symbol + "__cannbot_invoke_v9")
        self.invoke.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64]
        self.invoke.restype = ctypes.c_int32

    def frame(self, values):
        schema = self.schema
        if len(values) != schema.slot_count:
            raise ValueError(f"Expected {schema.slot_count} native arguments, got {len(values)}")
        # Per-call storage avoids races between workers/streams or concurrent
        # host invocations of the same cached library handle.
        raw = ctypes.create_string_buffer(schema.frame_bytes + schema.frame_alignment)
        offset = (-ctypes.addressof(raw)) % schema.frame_alignment
        frame = (ctypes.c_char * schema.frame_bytes).from_buffer(raw, offset)
        header = struct.pack(
            "<IHHIIQ32s",
            FRAME_MAGIC,
            HOST_ABI_VERSION,
            SCHEMA_VERSION,
            schema.slot_count,
            0,
            schema.frame_bytes,
            bytes(schema.contract_digest),
        )
        ctypes.memmove(frame, header, len(header))
        for index, value in enumerate(values):
            slot = schema.slots[index]

            def write(relative_offset, payload, slot=slot, index=index):
                if relative_offset + len(payload) > slot.size:
                    raise ValueError(f"Native slot {index} payload overflow")
                ctypes.memmove(ctypes.addressof(frame) + slot.offset + relative_offset, payload, len(payload))

            if slot.kind == 1:
                if not isinstance(value, torch.Tensor):
                    raise TypeError(f"Native tensor slot {index} requires a tensor")
                if TENSOR_DTYPES.get(str(value.dtype)) != (slot.dtype_code, slot.dtype_bits):
                    raise TypeError(f"Native tensor slot {index} dtype mismatch: {value.dtype}")
                shape, stride = tuple(value.shape), tuple(value.stride())
                if (
                    len(shape) != slot.logical_rank
                    or slot.logical_rank != slot.physical_rank
                    or slot.shape_offset != slot.physical_shape_offset
                    or slot.stride_offset != slot.physical_stride_offset
                    or any(dimension < 0 for dimension in (*shape, *stride))
                ):
                    raise ValueError(f"Native tensor slot {index} shape/stride mismatch")
                write(slot.pointer_offset, struct.pack("<Q", value.data_ptr()))
                write(slot.shape_offset, struct.pack("<" + "q" * len(shape), *shape))
                write(slot.stride_offset, struct.pack("<" + "q" * len(stride), *stride))
            else:
                format_code = SCALAR_FORMATS.get((slot.dtype_code, slot.dtype_bits))
                if format_code is None or slot.size != slot.dtype_bits // 8:
                    raise ValueError(f"Unsupported native scalar slot {index}")
                if isinstance(value, torch.Tensor):
                    raise TypeError("Native scalar arguments must not read device tensors")
                write(0, struct.pack("<" + format_code, value))
        return raw, frame

    def __call__(self, *values):
        tensors = [value for value in values if isinstance(value, torch.Tensor)]
        if not tensors or tensors[0].device.type != "npu":
            raise ValueError("Native attention requires NPU tensors")
        device = tensors[0].device
        if any(tensor.device != device for tensor in tensors):
            raise ValueError("Native attention tensors must share one NPU")
        raw, frame = self.frame(values)
        with torch.npu.device(device):
            stream = torch.npu.current_stream(device)
            # Input allocations can originate on another stream. Temporary
            # scratch allocated by host wrappers stays on this consumer stream.
            for tensor in tensors:
                tensor.record_stream(stream)
            code = self.invoke(int(stream.npu_stream), frame, len(frame))
        if code:
            raise RuntimeError(f"Native attention invoke failed: rc={code}")
        with self._counter_lock:
            self.launch_count += 1
            self.last_stream = int(stream.npu_stream)
        # Keep raw/frame alive through the synchronous host launch call.
        del frame, raw


class MetadataKernel:
    """Typed binding to the generated interface-mode AICPU launch shim."""

    def __init__(self, path: Path, record: dict):
        self.library = ctypes.CDLL(str(path))
        self.path = str(path)
        self.launch_count = 0
        self.last_stream = None
        self._counter_lock = threading.Lock()
        self.fields = tuple(
            (field["name"], field["ctype"]) for field in bundle_contract.canonical_fields(record["fields"])
        )
        if len({name for name, _ in self.fields}) != len(self.fields):
            raise ValueError("Duplicate metadata argument names")
        self.function = getattr(self.library, record["symbol"])
        self.function.argtypes = [ctypes.c_void_p] + [METADATA_CTYPES[kind] for _, kind in self.fields]
        self.function.restype = None

    def launch(self, stream, **values):
        if set(values) != {name for name, _ in self.fields}:
            raise ValueError("Metadata arguments do not match the frozen shim fields")
        for name, kind in self.fields:
            value = values[name]
            if not isinstance(value, int):
                raise TypeError(f"Metadata {name} must be a host integer or pointer")
            bits = ctypes.sizeof(METADATA_CTYPES[kind]) * 8
            signed = kind.startswith("c_int")
            lower, upper = (-(1 << (bits - 1)), 1 << (bits - 1)) if signed else (0, 1 << bits)
            if not lower <= value < upper:
                raise ValueError(f"Metadata {name} is outside {kind}")
        self.function(stream, *(values[name] for name, _ in self.fields))
        with self._counter_lock:
            self.launch_count += 1
            self.last_stream = int(stream)


class A5NativeBundle:
    """One immutable bundle per worker; no JIT, fallback or hot-path polling."""

    def __init__(self, directory: str):
        self.root = Path(directory).expanduser().resolve(strict=True)
        manifest_path = self.root / "manifest.json"
        manifest_bytes = manifest_path.read_bytes()
        self.manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        self.manifest = bundle_contract.validate_bundle_manifest(json.loads(manifest_bytes))
        self.runtime_sources = {
            str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), Path(bundle_contract.__file__))
        }
        self.files = self.manifest["files"]
        for relative_path in self.files:
            self.artifact(relative_path)
        self._kernels = {}
        for record in self.manifest["kernels"]:
            key = (record["op"], _key(record["key"]))
            if key in self._kernels:
                raise ValueError(f"Duplicate native attention profile: {key}")
            self._kernels[key] = record
        self._loaded_kernels = {}
        self._metadata = {}
        self._modules = {}
        self._apis = {}
        self._api_stats = {}
        self._lock = threading.RLock()
        # A unique package namespace supports the private relative import of
        # head_partition, without replacing the process-wide external ops package.
        identity = str(self.root) + ":" + self.manifest_sha256
        self.namespace = "_vllm_ascend_a5_" + hashlib.sha256(identity.encode()).hexdigest()
        package = ModuleType(self.namespace)
        package.__path__ = [str(self.root)]
        sys.modules[self.namespace] = package

    def artifact(self, relative_path: str) -> Path:
        if relative_path not in self.files:
            raise ValueError(f"Unbound native artifact: {relative_path}")
        path = (self.root / relative_path).resolve(strict=True)
        if not path.is_relative_to(self.root):
            raise ValueError("Native artifact escapes its bundle directory")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != self.files[relative_path]:
            raise ValueError(f"Native artifact hash mismatch: {relative_path}")
        return path

    def platform_info(self, stream=None):
        device = stream.device if stream is not None else None
        cores = int(torch.npu.get_device_properties(device).cube_core_num)
        if cores != self.manifest["cube_core_num"]:
            raise ValueError(f"Bundle requires {self.manifest['cube_core_num']} Cube cores, device has {cores}")
        return SimpleNamespace(cube_core_num=cores)

    def kernel(self, op: str, key):
        dispatch_key = (op, _key(key))
        with self._lock:
            if dispatch_key not in self._kernels:
                raise NotImplementedError(f"No precompiled native attention profile: {dispatch_key}")
            if dispatch_key not in self._loaded_kernels:
                record = self._kernels[dispatch_key]
                self._loaded_kernels[dispatch_key] = NativeKernel(self.artifact(record["library"]), record)
            return self._loaded_kernels[dispatch_key]

    def metadata(self, op: str):
        with self._lock:
            if op not in self._metadata:
                record = self.manifest["metadata"][op]
                self._metadata[op] = MetadataKernel(self.artifact(record["library"]), record)
            return self._metadata[op]

    def _module(self, relative_path: str):
        with self._lock:
            if relative_path not in self._modules:
                path = self.artifact(relative_path)
                name = self.namespace + "." + path.stem
                spec = importlib.util.spec_from_file_location(name, path)
                if spec is None or spec.loader is None:
                    raise ImportError(f"Cannot load native host module: {path}")
                module = importlib.util.module_from_spec(spec)
                module._bundle = self
                sys.modules[name] = module
                try:
                    spec.loader.exec_module(module)
                except BaseException:
                    sys.modules.pop(name, None)
                    raise
                self._modules[relative_path] = module
            return self._modules[relative_path]

    def get_ops(self, name: str):
        if name not in SUPPORTED_OPERATORS:
            raise ValueError(f"Unsupported bundled attention operator: {name}")
        if name == "mixed_quant_sparse_flash_mla":
            # Its public128 function imports .head_partition at call time.
            # Preload that dependency through the same verified module path,
            # so the relative import cannot bypass host-source identity checks.
            self._module("head_partition.py")
        record = self.manifest["operators"][name]
        compute = getattr(self._module(record["compute"]), name)
        metadata = getattr(self._module(record["metadata"]), name + "_metadata")
        if not callable(compute) or not callable(metadata):
            raise TypeError(f"Bundle does not export callable APIs for {name}")
        return self._record_api(name, compute, is_metadata=False), self._record_api(
            name + "_metadata", metadata, is_metadata=True
        )

    def _record_api(self, name, function, *, is_metadata):
        with self._lock:
            if name in self._apis:
                return self._apis[name]

            @wraps(function)
            def call(*args, **kwargs):
                result = function(*args, **kwargs)
                first = args[0] if args else kwargs.get("q")
                record = {
                    "kind": "metadata" if is_metadata else "compute",
                    # Public query heads remain visible even when MQ128 is
                    # implemented internally by two private64 launches.
                    "query_shape": tuple(first.shape) if not is_metadata and isinstance(first, torch.Tensor) else None,
                }
                for attribute in (
                    "num_heads_q",
                    "topk",
                    "quant_mode",
                    "mask_mode",
                    "cmp_ratio",
                    "candidate_topk_blocks",
                    "candidate_block_size",
                    "has_cmp_kv",
                    "layout_q",
                    "layout_k",
                    "layout_kv",
                    "return_softmax_lse",
                ):
                    value = kwargs.get(attribute)
                    if value is None or type(value) in (bool, int, float, str):
                        record[attribute] = value
                with self._lock:
                    previous = self._api_stats.get(name, {})
                    record["calls_returned"] = previous.get("calls_returned", 0) + 1
                    self._api_stats[name] = record
                return result

            self._apis[name] = call
            return call

    def snapshot(self) -> dict:
        """Read bounded host-side identity/counters without querying a device."""
        with self._lock:
            kernels = []
            for key, kernel in self._loaded_kernels.items():
                with kernel._counter_lock:
                    record = self._kernels[key]
                    kernels.append(
                        {
                            "op": key[0],
                            "key": record["key"],
                            "library": kernel.path,
                            "library_sha256": self.files[record["library"]],
                            "symbol": kernel.symbol,
                            "contract_digest": record["contract_digest"],
                            "host_launch_calls_returned": kernel.launch_count,
                            "last_stream": kernel.last_stream,
                        }
                    )
            metadata = []
            for op, shim in self._metadata.items():
                with shim._counter_lock:
                    record = self.manifest["metadata"][op]
                    metadata.append(
                        {
                            "op": op,
                            "library": shim.path,
                            "library_sha256": self.files[record["library"]],
                            "host_launch_calls_returned": shim.launch_count,
                            "last_stream": shim.last_stream,
                        }
                    )
            return {
                "manifest_sha256": self.manifest_sha256,
                "pid": os.getpid(),
                "runtime_sources": dict(self.runtime_sources),
                "modules": {name: self.files[name] for name in self._modules},
                "public_apis": {name: dict(record) for name, record in self._api_stats.items()},
                "kernels": kernels,
                "metadata": metadata,
                "meaning": "host API calls returned; device completion and numerical correctness are not established",
            }
