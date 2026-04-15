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
from vllm.distributed import get_ep_group, tensor_model_parallel_all_reduce

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.ops.fused_moe.fused_moe import AscendSharedFusedMoE
from vllm_ascend.ops.fused_moe.moe_comm_method import _MoECommMethods
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type


from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType
from vllm_ascend.flash_common3_context import get_flash_common3_context, set_flash_common3_context
from vllm_ascend.ops.fused_moe.experts_selector import select_experts
from vllm_ascend.ops.fused_moe.moe_comm_method import AllGatherCommImpl, FusedExpertsResult
from vllm_ascend.utils import (
    npu_stream_switch,
    shared_expert_dp_enabled,
)
from vllm_ascend.ops.fused_moe.fused_moe import FusedMoEEvents, FusedMoEResult

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
    forward_context.flash_comm_v1_enabled = False


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
    _ensure_forward_context_attr("num_tokens", int | None, None)
    _ensure_forward_context_attr("in_profile_run", bool, False)
    _ensure_forward_context_attr("moe_comm_type", MoECommType | None, moe_comm_type)
    _ensure_forward_context_attr("moe_comm_method", Any, _MoECommMethods.get(moe_comm_type))
    _ensure_forward_context_attr("flash_comm_v1_enabled", bool, False)


class AscendHunyuanFusedMoE(AscendFusedMoE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward_impl(  # type: ignore[override]
        self, hidden_states: torch.Tensor, router_logits: torch.Tensor, return_with_event: bool = False
    ) -> torch.Tensor | FusedMoEResult:
        assert self.quant_method is not None

        forward_context = get_forward_context()
        # When static kernels are enabled, the forward pass runs twice (compilation + capture),
        # causing moe_layer_index to overflow. Wrap the index to prevent out-of-bounds errors.
        if self.enable_npugraph_ex_static_kernel:
            moe_layer_index = forward_context.moe_layer_index % (len(forward_context.all_moe_layers))
            forward_context.moe_layer_index = moe_layer_index

        # Load balancing for token distribution among experts in dummy_run
        # TODO: The community only considers load balancing when DP > 1.
        # This approach may overlook some extreme scenarios.
        enable_force_load_balance = _EXTRA_CTX.in_profile_run

        forward_context = get_forward_context()
        if self.multistream_overlap_gate:
            assert AscendFusedMoE.gate_stream is not None
            fc3_context = get_flash_common3_context()
            assert fc3_context is not None
            AscendFusedMoE.gate_stream.wait_stream(torch.npu.current_stream())
            with npu_stream_switch(AscendFusedMoE.gate_stream, enabled=self.multistream_overlap_gate):
                # share_expert
                assert fc3_context.shared_experts is not None
                shared_out = fc3_context.shared_experts(hidden_states)
                # NOTE: This is exactly the opposite of `maybe_all_reduce_tensor_model_parallel`
                moe_comm_type = _EXTRA_CTX.moe_comm_type
                if (
                    moe_comm_type in {MoECommType.ALLTOALL, MoECommType.MC2, MoECommType.FUSED_MC2}
                    and not shared_expert_dp_enabled()
                ):
                    shared_out = tensor_model_parallel_all_reduce(shared_out)
                set_flash_common3_context(shared_out=shared_out)

                topk_weights, topk_ids = select_experts(
                    hidden_states=hidden_states,
                    router_logits=router_logits,
                    top_k=self.top_k,
                    use_grouped_topk=self.use_grouped_topk,
                    renormalize=self.renormalize,
                    topk_group=self.topk_group,
                    num_expert_group=self.num_expert_group,
                    custom_routing_function=self.custom_routing_function,
                    scoring_func=self.scoring_func,
                    routed_scaling_factor=self.routed_scaling_factor,
                    e_score_correction_bias=self.e_score_correction_bias,
                    global_num_experts=self.global_num_experts,
                )

                if isinstance(_EXTRA_CTX.moe_comm_method, AllGatherCommImpl):
                    topk_weights = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(topk_weights, True, True)
                    topk_ids = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(topk_ids, True, True)

                set_flash_common3_context(topk_weights=topk_weights, topk_ids=topk_ids)

        if getattr(forward_context, "moe_comm_type", None) == MoECommType.ALLTOALL:
            replace_allreduce = getattr(forward_context, "row_parallel_linear_fc1_enabled", None)
        else:
            replace_allreduce = _EXTRA_CTX.flash_comm_v1_enabled

        hidden_states, router_logits, mc2_mask, context_metadata = _EXTRA_CTX.moe_comm_method.prepare(
            hidden_states=hidden_states,
            router_logits=router_logits,
            replace_allreduce=replace_allreduce,
            enable_shared_expert_dp=self.enable_shared_expert_dp,
            quant_type=self.quant_type,
        )

        # Make sure the default stream waits for the gate stream to finish.
        if self.multistream_overlap_gate:
            torch.npu.current_stream().wait_stream(AscendFusedMoE.gate_stream)

        if isinstance(hidden_states, tuple):
            hidden_states, pertoken_scale = hidden_states
        else:
            pertoken_scale = None

        # Matrix multiply.
        fused_experts_results: FusedExpertsResult = self.quant_method.apply(
            layer=self,
            x=hidden_states,
            router_logits=router_logits,
            pertoken_scale=pertoken_scale,
            top_k=self.top_k,
            renormalize=self.renormalize,
            use_grouped_topk=self.use_grouped_topk,
            global_num_experts=self.global_num_experts,
            expert_map=self._expert_map,
            topk_group=self.topk_group,
            num_expert_group=self.num_expert_group,
            custom_routing_function=self.custom_routing_function,
            scoring_func=self.scoring_func,
            routed_scaling_factor=self.routed_scaling_factor,
            e_score_correction_bias=self.e_score_correction_bias,
            activation=self.activation,
            apply_router_weight_on_input=self.apply_router_weight_on_input,
            enable_force_load_balance=enable_force_load_balance,
            log2phy=self.log2phy,
            global_redundant_expert_num=self.global_redundant_expert_num,
            mc2_mask=mc2_mask,
        )

        if self.dynamic_eplb:
            expert_tokens = fused_experts_results.expert_tokens
            group_list_type = fused_experts_results.group_list_type
            assert expert_tokens is not None and group_list_type is not None, (
                "expert_tokens and group_list_type should not be None when dynamic_eplb is enabled."
            )
            local_load = (
                expert_tokens
                if group_list_type == 1
                else torch.cat([expert_tokens[:1], expert_tokens[1:] - expert_tokens[:-1]])
            )
            if self.multi_stage:
                cur_iter = torch.remainder(self.load_counter, self.num_iter)
                self.moe_load.index_add_(
                    dim=0, index=cur_iter, source=local_load.to(torch.int32, non_blocking=True).view(1, -1)
                )
                self.load_counter.add_(1)
            else:
                self.moe_load.add_(local_load)
        routed_out = _EXTRA_CTX.moe_comm_method.finalize(
            hidden_states=fused_experts_results.routed_out,
            reduce_results=self.reduce_results,
            context_metadata=context_metadata,
        )

        if return_with_event:
            return FusedMoEResult(
                routed_out=routed_out,
                before_dispatch_evt=fused_experts_results.before_dispatch_evt,
                before_combine_evt=fused_experts_results.before_combine_evt,
            )
        else:
            # The vLLM FusedMoE forward_impl does not return events.
            return routed_out


class AscendHunyuanSharedFusedMoE(AscendSharedFusedMoE, AscendHunyuanFusedMoE):
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
        if self._shared_experts is None:
            fused_out = self.forward_impl(hidden_states=hidden_states, router_logits=router_logits,)
            shared_out = None
            return shared_out, fused_out
        return self.forward_impl(hidden_states=hidden_states, router_logits=router_logits)
    
    def forward_impl(  # type: ignore[override]
        self, hidden_states: torch.Tensor, router_logits: torch.Tensor
    ):  
        if self.multistream_overlap_gate:
            set_flash_common3_context(shared_experts=self._shared_experts)

        before_routed_experts = torch.npu.current_stream().record_event()
        fused_moe_results = AscendHunyuanFusedMoE.forward_impl(
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
