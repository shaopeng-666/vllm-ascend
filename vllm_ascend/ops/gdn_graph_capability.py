#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from contextlib import contextmanager
from contextvars import ContextVar
from enum import IntFlag, auto
from typing import Literal


class GDNPrefillGraphCapability(IntFlag):
    """Replay contracts supported by the selected GDN prefill implementation."""

    EXACT_SINGLE_REQUEST = auto()
    PADDED_DYNAMIC_LENGTHS = auto()
    MULTI_REQUEST = auto()


_gdn_prefill_graph_capture_active = ContextVar(
    "gdn_prefill_graph_capture_active",
    default=False,
)


def is_gdn_prefill_graph_capture_active() -> bool:
    return _gdn_prefill_graph_capture_active.get()


@contextmanager
def gdn_prefill_graph_capture_scope():
    """Mark the lexical GPUModelRunner capture scope for Python custom ops."""
    token = _gdn_prefill_graph_capture_active.set(True)
    try:
        yield
    finally:
        _gdn_prefill_graph_capture_active.reset(token)


def is_exact_gdn_prefill_graph_contract(
    *,
    num_reqs: int,
    num_actual_tokens: int,
    graph_num_tokens: int,
    min_num_scheduled_tokens: int,
    max_num_scheduled_tokens: int,
) -> bool:
    """Whether a graph bucket has immutable singleton prefill metadata."""
    return (
        num_reqs == 1
        and num_actual_tokens == graph_num_tokens
        and min_num_scheduled_tokens == graph_num_tokens
        and max_num_scheduled_tokens == graph_num_tokens
    )


def get_gdn_prefill_graph_capability(
    configured_backend: Literal["auto", "native", "fla_npu"],
    *,
    native_dynamic_lengths_available: bool,
    capture_metadata_is_exact: bool = False,
    fla_runtime_supported: bool = True,
    fla_graph_capture_supported: bool = True,
) -> GDNPrefillGraphCapability:
    if (
        configured_backend == "fla_npu"
        and fla_runtime_supported
        and capture_metadata_is_exact
        and fla_graph_capture_supported
    ):
        return GDNPrefillGraphCapability.EXACT_SINGLE_REQUEST

    capability = GDNPrefillGraphCapability(0)
    if native_dynamic_lengths_available:
        capability |= (
            GDNPrefillGraphCapability.EXACT_SINGLE_REQUEST
            | GDNPrefillGraphCapability.PADDED_DYNAMIC_LENGTHS
            | GDNPrefillGraphCapability.MULTI_REQUEST
        )
    elif configured_backend != "fla_npu":
        capability |= GDNPrefillGraphCapability.EXACT_SINGLE_REQUEST
    return capability


def select_gdn_prefill_implementation(
    configured_backend: Literal["auto", "native", "fla_npu"],
    *,
    is_graph_capture: bool,
    graph_metadata_is_exact: bool,
    native_dynamic_lengths_available: bool,
    fla_runtime_supported: bool = True,
    fla_graph_capture_supported: bool = True,
) -> Literal["triton", "native", "fla_npu"]:
    """Select an implementation without putting host metadata in a replay graph.

    FLA NPU remains the requested implementation only on hardware that supports
    its high-level prepare path. On older hardware every execution mode falls
    back to the native operator because the current FLA contract requires A5
    features (QK L2 normalization and state-v-first). An ACL graph can retain
    FLA only when the captured host metadata is also an exact immutable
    singleton. If no native implementation is available, use Triton.
    """
    if configured_backend == "fla_npu":
        if not fla_runtime_supported:
            if native_dynamic_lengths_available:
                return "native"
            return "triton"
        if is_graph_capture:
            if fla_graph_capture_supported and graph_metadata_is_exact:
                return "fla_npu"
            if native_dynamic_lengths_available:
                return "native"
            return "triton"
        return "fla_npu"
    if native_dynamic_lengths_available and configured_backend in {"auto", "native"}:
        return "native"
    return "triton"


def select_gdn_prefill_graph_backend(
    configured_backend: Literal["auto", "native", "fla_npu"],
    *,
    runtime_mode: object,
    is_ascend_950: bool,
    metadata_immutable: bool,
    native_dynamic_lengths_available: bool,
    capture_enabled: bool = False,
) -> Literal["triton", "native", "fla_npu"] | None:
    """Resolve the typed backend marker carried by FULL graph metadata.

    ``for_cudagraph_capture`` is a stream-state detail and is false while a
    graph's FX warmup metadata is built.  The forward context's explicit
    runtime mode is the stable contract shared by dummy warmup, capture, and
    replay.  A real eager prefill has runtime mode NONE even when the service
    is configured for FULL_DECODE_ONLY.
    """
    runtime_mode_name = getattr(runtime_mode, "name", str(runtime_mode))
    if not capture_enabled and (runtime_mode is None or runtime_mode_name == "NONE"):
        return None
    return select_gdn_prefill_implementation(
        configured_backend,
        is_graph_capture=True,
        graph_metadata_is_exact=metadata_immutable,
        native_dynamic_lengths_available=native_dynamic_lengths_available,
        fla_runtime_supported=is_ascend_950,
        fla_graph_capture_supported=is_ascend_950,
    )
