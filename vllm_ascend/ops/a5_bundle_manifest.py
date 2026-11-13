# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Environment-independent contract for the initial A5 attention bundle.

This module uses only the standard library. The offline assembler and worker
loader share it so incomplete profiles or incompatible shim fields fail before
host-module imports, library loading or device work.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import PurePosixPath

OPERATOR_MODULES = {
    "quant_lightning_indexer": {"compute": "qli.py", "metadata": "qli_metadata.py"},
    "quant_sparse_lightning_indexer": {"compute": "qsli.py", "metadata": "qsli_metadata.py"},
    "mixed_quant_sparse_flash_mla": {"compute": "mq.py", "metadata": "mq_metadata.py"},
}
REQUIRED_HOST_FILES = frozenset(
    ("qli.py", "qsli.py", "mq.py", "qli_metadata.py", "qsli_metadata.py", "mq_metadata.py", "head_partition.py")
)
REQUIRED_PROFILES = (
    ("qli", (1024, False, 1, 3, 2, 128, 8, 64, 4, False, False)),
    ("qli", (1024, True, 2048, 3, 1, 128, 8, 64, 4, False, False)),
    ("qsli", (16384, 1024, 2048, 3, 1, False, 128, 8, 64)),
    ("mq", (False, True, False, False, "off")),
    ("mq", (True, True, True, False, "off")),
)
PROFILE_TYPES = {
    "qli": (int, bool, int, int, int, int, int, int, int, bool, bool),
    "qsli": (int, int, int, int, int, bool, int, int, int),
    "mq": (bool, bool, bool, bool, str),
}
PROFILE_ARGUMENTS = {
    "qli": (
        "topk",
        "candidate_enabled",
        "candidate_topk",
        "mask_mode",
        "cmp_ratio",
        "pa_block_size",
        "split_count",
        "n1",
        "query_tile_rows",
        "streaming_mode",
        "tnd_layout",
    ),
    "qsli": (
        "tokens",
        "topk",
        "candidate_capacity",
        "mask_mode",
        "cmp_ratio",
        "has_output_offset",
        "pa_block_size",
        "splits",
        "n1",
    ),
    "mq": ("has_cmp", "has_ori_length", "has_cmp_length", "has_lse", "addr_mode"),
}
# AICPU sources are compiled for LP64 Linux/aarch64. ctypes names emitted by
# fields.json can be platform aliases (U32 -> c_uint, I64 -> c_long). Convert
# them to explicit widths rather than using the assembling host's C long size.
CTYPE_ALIASES = {
    "c_int": "c_int32",
    "c_int32": "c_int32",
    "c_uint": "c_uint32",
    "c_uint32": "c_uint32",
    "c_long": "c_int64",
    "c_longlong": "c_int64",
    "c_int64": "c_int64",
    "c_ulong": "c_uint64",
    "c_ulonglong": "c_uint64",
    "c_uint64": "c_uint64",
}
INDEX_POINTERS = ("cu", "used_q", "used_k", "residual", "candidate_length", "output_offset", "output", "scratch")
INDEX_FLAGS = ("has_cu", "has_q", "has_k", "has_residual", "sparse", "mask", "ratio")
INDEX_COUNTS = ("total_q", "batch", "max_tasks", "capacity", "splits", "workers", "groups", "query_rows", "heads")
QSLI_FIELDS = (
    tuple((name, "c_uint64") for name in INDEX_POINTERS)
    + tuple((name, "c_uint32") for name in INDEX_FLAGS)
    + tuple((name, "c_int64") for name in INDEX_COUNTS)
    + (("ld", "c_uint32"), ("has_offset", "c_uint32"))
)
QLI_FIELDS = (
    QSLI_FIELDS[:3] + (("cu_k", "c_uint64"),) + QSLI_FIELDS[3:11] + (("has_cu_k", "c_uint32"),) + QSLI_FIELDS[11:]
)
METADATA_FIELDS = {
    "qli": QLI_FIELDS,
    "qsli": QSLI_FIELDS,
    "mq": (
        ("win_len", "c_uint64"),
        ("cmp_len", "c_uint64"),
        ("metadata", "c_uint64"),
        ("rows", "c_int32"),
        ("blocks", "c_int32"),
        ("has_cmp", "c_int32"),
        ("cu_q", "c_uint64"),
        ("batch", "c_int32"),
        ("support_fd", "c_int32"),
    ),
}


def profile_key(op: str, values) -> tuple[str, str]:
    return op, json.dumps(values, separators=(",", ":"), allow_nan=False)


def canonical_fields(fields: list[dict]) -> list[dict]:
    result = []
    names = set()
    for field in fields:
        name, kind = field["name"], field["ctype"]
        if name in names or kind not in CTYPE_ALIASES:
            raise ValueError(f"Duplicate field or unsupported metadata C type: {name}/{kind}")
        names.add(name)
        result.append({"name": name, "ctype": CTYPE_ALIASES[kind]})
    return result


def _relative_path(value: str) -> None:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"Invalid bundle-relative path: {value!r}")
    path = PurePosixPath(value)
    if not path.parts or path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise ValueError(f"Noncanonical bundle-relative path: {value!r}")


def _sha256(value: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("A frozen artifact requires a lowercase SHA256 digest")


def validate_bundle_manifest(manifest: dict) -> dict:
    """Return a normalized private copy, without reading files or a device."""
    result = copy.deepcopy(manifest)
    if result.get("version") != 1 or result.get("status") != "READY":
        raise ValueError("Native attention bundle is not an assembled READY bundle")
    if type(result.get("cube_core_num")) is not int or result["cube_core_num"] != 32:
        raise ValueError("Initial native bundle requires the 32-Cube-core workspace contract")
    files = result["files"]
    if not files.keys() >= REQUIRED_HOST_FILES:
        raise ValueError("Native bundle is missing required host modules")
    for path, digest in files.items():
        _relative_path(path)
        _sha256(digest)
    if result["operators"] != OPERATOR_MODULES:
        raise ValueError("Native bundle must export the three matching compute/metadata module pairs")
    keys = set()
    libraries = set()
    for record in result["kernels"]:
        op, values = record["op"], record["key"]
        types = PROFILE_TYPES.get(op)
        if types is None or not isinstance(values, (list, tuple)) or len(values) != len(types):
            raise ValueError(f"Invalid native profile key: {op}/{values}")
        if any(type(value) is not expected for value, expected in zip(values, types)):
            raise ValueError(f"Native profile key has an incorrect scalar type: {op}/{values}")
        key = profile_key(op, values)
        if key in keys:
            raise ValueError(f"Duplicate native attention profile: {key}")
        keys.add(key)
        _sha256(record["contract_digest"])
        if re.fullmatch(r"[A-Za-z_]\w*", record["symbol"]) is None:
            raise ValueError("Invalid native schema symbol")
        if type(record["slot_count"]) is not int or record["slot_count"] <= 0:
            raise ValueError("Native schema must have a positive slot count")
        if type(record["frame_bytes"]) is not int or record["frame_bytes"] <= 56:
            raise ValueError("Native schema frame must include header and arguments")
        libraries.add(record["library"])
    if any(profile_key(op, key) not in keys for op, key in REQUIRED_PROFILES):
        raise ValueError("Native bundle does not cover the five required model profiles")
    if set(result["metadata"]) != set(METADATA_FIELDS):
        raise ValueError("Each native operator requires its own metadata shim")
    metadata_libraries = set()
    for op, record in result["metadata"].items():
        fields = canonical_fields(record["fields"])
        if tuple((field["name"], field["ctype"]) for field in fields) != METADATA_FIELDS[op]:
            raise ValueError(f"Metadata field order/width does not match the {op} host API")
        if re.fullmatch(r"[A-Za-z_]\w*", record["symbol"]) is None:
            raise ValueError("Invalid native metadata symbol")
        record["fields"] = fields
        metadata_libraries.add(record["library"])
    if len(metadata_libraries) != len(METADATA_FIELDS):
        raise ValueError("QLI, QSLI and MQ require distinct metadata libraries")
    libraries.update(metadata_libraries)
    for path in libraries:
        _relative_path(path)
        if path not in files or path in REQUIRED_HOST_FILES:
            raise ValueError(f"Native library is not separately bound to a file/hash: {path}")
    return result
