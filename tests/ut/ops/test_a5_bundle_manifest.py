# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generated metadata C type aliases must retain their signedness/width."""

import pytest

from vllm_ascend.ops.a5_bundle_manifest import canonical_fields


def test_lp64_generated_fields_are_fixed_width_not_host_long_types():
    original = [
        {"name": "pointer", "ctype": "c_uint64"},
        {"name": "flag", "ctype": "c_uint"},
        {"name": "row_count", "ctype": "c_long"},
        {"name": "signed_int", "ctype": "c_int"},
    ]
    fields = canonical_fields(original)
    assert [field["ctype"] for field in fields] == ["c_uint64", "c_uint32", "c_int64", "c_int32"]
    assert original[1]["ctype"] == "c_uint"  # Source descriptor remains intact.


@pytest.mark.parametrize(
    "fields",
    [
        [{"name": "first", "ctype": "c_int32"}, {"name": "first", "ctype": "c_uint32"}],
        [{"name": "opaque", "ctype": "c_void_p"}],
    ],
)
def test_rejects_duplicate_or_untyped_shim_fields(fields):
    with pytest.raises(ValueError, match="Duplicate field or unsupported"):
        canonical_fields(fields)
