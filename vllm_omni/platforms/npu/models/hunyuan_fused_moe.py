# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import os
import torch
import torch.distributed as dist
import vllm.forward_context as _vllm_fc
from vllm.forward_context import get_forward_context
from vllm.config import VllmConfig
from vllm.distributed import get_ep_group
from vllm.distributed.parallel_state import get_tensor_model_parallel_world_size
from vllm.distributed.parallel_state import (
    init_model_parallel_group as vllm_init_model_parallel_group,
)
from vllm.distributed import get_ep_group

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.ops.fused_moe.fused_moe import AscendSharedFusedMoE, AscendFusedMoE
from vllm_ascend.ops.fused_moe.moe_comm_method import _MoECommMethods
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type


from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType
from vllm_ascend.flash_common3_context import get_flash_common3_context, set_flash_common3_context
from vllm_ascend.ops.fused_moe.fused_moe import FusedMoEEvents

from vllm_omni.diffusion.distributed.parallel_state import (
    get_data_parallel_world_size,
    get_world_group,
)
from vllm_omni.diffusion.forward_context import get_forward_context as omni_get_ctx


def _ensure_forward_context_attr(name: str, annotation: Any, default: Any) -> None:
    if name not in _vllm_fc.ForwardContext.__annotations__:
        _vllm_fc.ForwardContext.__annotations__[name] = annotation
    if not hasattr(_vllm_fc.ForwardContext, name):
        setattr(_vllm_fc.ForwardContext, name, default)


def _set_hunyuan_fused_moe_forward_context(num_tokens: int) -> None:
    if not _vllm_fc.is_forward_context_available():
        return

    forward_context = _vllm_fc.get_forward_context()
    forward_context.num_tokens = num_tokens
    forward_context.moe_comm_type = _select_moe_comm_method(vllm_config=omni_get_ctx().vllm_config)
    forward_context.moe_comm_method = _MoECommMethods.get(forward_context.moe_comm_type)


def _init_mc2_group_for_diffusion(
    world_size: int,
    data_parallel_size: int,
    tensor_parallel_size: int,
    backend: str,
    local_rank: int,
) -> None:
    import vllm_ascend.distributed.parallel_state as vllm_ascend_parallel_state

    if getattr(vllm_ascend_parallel_state, "_MC2", None) is not None:
        return
    all_ranks = torch.arange(world_size).reshape(-1, data_parallel_size * tensor_parallel_size)
    group_ranks = all_ranks.unbind(0)
    group_ranks = [x.tolist() for x in group_ranks]

    vllm_ascend_parallel_state._MC2 = vllm_init_model_parallel_group(
        group_ranks,
        local_rank,
        backend,
        group_name="mc2",
    )


def _select_moe_comm_method(vllm_config: VllmConfig) -> MoECommType | None:
    soc_version = get_ascend_device_type()
    if not vllm_config.parallel_config.enable_expert_parallel or get_ep_group().world_size == 1:
        moe_comm_type = MoECommType.ALLGATHER
    elif soc_version in {AscendDeviceType.A2}:
        moe_comm_type = MoECommType.ALLGATHER
    elif soc_version in {AscendDeviceType.A3}:
        moe_comm_type = MoECommType.ALLTOALL
    elif soc_version in {AscendDeviceType._310P}:
        moe_comm_type = MoECommType.ALLGATHER
    elif soc_version in {AscendDeviceType.A5}:
        moe_comm_type = MoECommType.ALLTOALL
    else:
        raise ValueError(f"Unsupported soc_version: {soc_version}")
    return moe_comm_type


def prepare_hunyuan_fused_moe_runtime() -> None:
    world_size = torch.distributed.get_world_size()
    data_parallel_size = get_data_parallel_world_size()
    tensor_parallel_size = get_tensor_model_parallel_world_size()
    backend = torch.distributed.get_backend(get_world_group().device_group)
    local_rank = get_world_group().local_rank
    _init_mc2_group_for_diffusion(
        world_size=world_size,
        data_parallel_size=data_parallel_size,
        tensor_parallel_size=tensor_parallel_size,
        backend=backend,
        local_rank=local_rank,
    )

    moe_comm_type = _select_moe_comm_method(vllm_config=omni_get_ctx().vllm_config)
    if moe_comm_type == MoECommType.ALLTOALL:
        moe_comm_type_name = "ALLTOALL"
    else:
        moe_comm_type_name = "ALLGATHER"
    _ensure_forward_context_attr("num_tokens", int | None, None)
    _ensure_forward_context_attr("in_profile_run", bool, False)
    _ensure_forward_context_attr("moe_comm_type", MoECommType | None, moe_comm_type)
    _ensure_forward_context_attr("moe_comm_type_name", str | None, moe_comm_type_name)
    _ensure_forward_context_attr("moe_comm_method", Any, _MoECommMethods.get(moe_comm_type))
    _ensure_forward_context_attr("flash_comm_v1_enabled", bool, False)
    
    
class AscendHunyuanSharedFusedMoE(AscendSharedFusedMoE, AscendFusedMoE):
    def __init__(self, *, prefix: str = "", **kwargs: Any) -> None:
        AscendSharedFusedMoE.__init__(self, prefix=prefix, **kwargs)
        self._prefix = prefix
        self._init_hook_handle = self.register_forward_pre_hook(self._initialize_kernel_hook, with_kwargs=True)

    def _initialize_kernel_hook(self, module: Any, args: Any, kwargs: Any) -> None:
        if self.quant_method:
            self.quant_method.process_weights_after_loading(self)
        self._init_hook_handle.remove()

    def forward(self, hidden_states: Any, router_logits: Any) -> Any:
        _set_hunyuan_fused_moe_forward_context(hidden_states.shape[0])
        forward_context = _vllm_fc.get_forward_context()
        ascend_flash_comm_v1_enabled = _EXTRA_CTX.flash_comm_v1_enabled
        _EXTRA_CTX.flash_comm_v1_enabled = forward_context.row_parallel_linear_fc1_enabled
        if self._shared_experts is None:
            fused_out = self.forward_impl(hidden_states=hidden_states, router_logits=router_logits,)
            shared_out = None
            return shared_out, fused_out
        try:
            return self.forward_impl(hidden_states=hidden_states, router_logits=router_logits,)
        finally:
            _EXTRA_CTX.flash_comm_v1_enabled = ascend_flash_comm_v1_enabled
            
    def forward_impl(  # type: ignore[override]
        self, hidden_states: torch.Tensor, router_logits: torch.Tensor
    ):  
        if self.multistream_overlap_gate:
            set_flash_common3_context(shared_experts=self._shared_experts)

        before_routed_experts = torch.npu.current_stream().record_event()
        fused_moe_results = AscendFusedMoE.forward_impl(
            self,
            hidden_states=hidden_states,
            router_logits=router_logits,
            return_with_event=True,
        )
        routed_out = fused_moe_results.routed_out
        if self._shared_experts is None:
            return routed_out
        
        if self.multistream_overlap_gate:
            fc3_context = get_flash_common3_context()
            assert fc3_context is not None
            shared_out = fc3_context.shared_out
        else:
            forward_context = get_forward_context()
            if forward_context.moe_comm_type == MoECommType.ALLTOALL and forward_context.row_parallel_linear_fc1_enabled:
                world_size = dist.get_world_size()
                local_rank, world_size = dist.get_rank(), dist.get_world_size()
                hidden_states_ = torch.zeros([world_size * hidden_states.shape[0], *hidden_states.shape[1:]], device=hidden_states.device, dtype=hidden_states.dtype)
                torch.distributed.all_gather_into_tensor(hidden_states_, hidden_states)
                hidden_states = hidden_states_
                hidden_states = hidden_states.contiguous()
                
                shared_out = self._forward_shared_experts(
                    hidden_states,
                    FusedMoEEvents(
                        before_routed_experts=before_routed_experts,
                        before_dispatch=fused_moe_results.before_dispatch_evt,
                        before_combine=fused_moe_results.before_combine_evt,
                    ),
                )   
                shared_out_chunk = torch.chunk(shared_out, world_size, dim=0)
                shared_out = shared_out_chunk[local_rank]
            else:
                shared_out = self._forward_shared_experts(
                    hidden_states,
                    FusedMoEEvents(
                        before_routed_experts=before_routed_experts,
                        before_dispatch=fused_moe_results.before_dispatch_evt,
                        before_combine=fused_moe_results.before_combine_evt,
                    ),
                )
        return shared_out, routed_out

    def __del__(self):
        import vllm_ascend.distributed.parallel_state as vllm_ascend_parallel_state

        if vllm_ascend_parallel_state._MC2:
            vllm_ascend_parallel_state._MC2.destroy()
        vllm_ascend_parallel_state._MC2 = None
