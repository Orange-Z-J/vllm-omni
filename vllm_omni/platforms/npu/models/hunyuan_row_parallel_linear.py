# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import vllm.forward_context as _vllm_fc
from vllm_ascend.ops.linear import AscendRowParallelLinear
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm_ascend.ops.linear_op import SequenceRowParallelOp
from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm.distributed import (
    split_tensor_along_last_dim,
)

import torch
from torch.nn.parameter import Parameter
import os

def _ensure_forward_context_attr(name: str, annotation: Any, default: Any) -> None:
    if name not in _vllm_fc.ForwardContext.__annotations__:
        _vllm_fc.ForwardContext.__annotations__[name] = annotation
    if not hasattr(_vllm_fc.ForwardContext, name):
        setattr(_vllm_fc.ForwardContext, name, default)


def prepare_hunyuan_row_parallel_linear_runtime() -> None:
    _ensure_forward_context_attr("row_parallel_linear_fc1_enabled", bool, False)
    _ensure_forward_context_attr("mmrs_fusion", bool, False)
    _ensure_forward_context_attr("fc1_pad_size", bool, 0)


def _set_hunyuan_row_parallel_linear_forward_context(num_tokens: int) -> None:
    if not _vllm_fc.is_forward_context_available():
        return

    forward_context = _vllm_fc.get_forward_context()
    forward_context.row_parallel_linear_fc1_enabled = bool(int(os.getenv("HY_IMAGE_3_ENABLE_FLASHCOMM1", 0)))
    forward_context.mmrs_fusion = True
    if forward_context.row_parallel_linear_fc1_enabled:
        tp_world_size = get_tensor_model_parallel_world_size()
        fc1_pad_size = (tp_world_size - (num_tokens % tp_world_size)) % tp_world_size
        forward_context.fc1_pad_size = fc1_pad_size

def _set_hunyuan_row_parallel_linear_ascend_forward_context() -> None:
    try:
        flash_comm_v1_enabled = _EXTRA_CTX.flash_comm_v1_enabled
        mmrs_fusion = _EXTRA_CTX.mmrs_fusion
    except AssertionError:
        flash_comm_v1_enabled = False
        mmrs_fusion = False

    try:
        pad_size = _EXTRA_CTX.pad_size
    except AttributeError:
        pad_size = 0

    forward_context = _vllm_fc.get_forward_context()
    _EXTRA_CTX.flash_comm_v1_enabled = forward_context.row_parallel_linear_fc1_enabled
    _EXTRA_CTX.mmrs_fusion = forward_context.mmrs_fusion
    _EXTRA_CTX.pad_size = forward_context.fc1_pad_size
    return flash_comm_v1_enabled, mmrs_fusion, pad_size


class HunyuanSequenceRowParallelOp(SequenceRowParallelOp):
    def __init__(self, layer):
        super().__init__(layer)
        self.unique_prefix = None

    def apply(self, input_):
        output, output_bias = self.apply_impl(input_)

        if not self.return_bias:
            return output
        return output, output_bias
    
    def apply_impl(self, input_: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        if self.input_is_parallel:
            input_parallel = input_
        else:
            splitted_input = split_tensor_along_last_dim(input_, num_partitions=self.tp_size)
            input_parallel = splitted_input[self.tp_rank].contiguous()

        assert self.quant_method is not None
        bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias

        if self.tp_size == 1 or not self.reduce_results:
            output = self.quant_method.apply(self.layer, input_parallel, bias=bias_)
        else:
            output = self.matmul_and_reduce(input_parallel, bias_)

        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias

    def update_attrs(self):
        super().update_attrs()
        self.input_is_parallel = self.layer.input_is_parallel
        self.reduce_results = self.layer.reduce_results


class AscendHunyuanRowParallelLinear(AscendRowParallelLinear):
    def __init__(self, *args: Any, prefix: str = "", **kwargs: Any) -> None:
        super().__init__(*args, prefix=prefix, **kwargs)
        self.unique_prefix=None
        self.custom_op = HunyuanSequenceRowParallelOp(self)
        if self.custom_op is not None:
            self.custom_op.update_attrs()
        self._prefix = prefix
        self._init_hook_handle = self.register_forward_pre_hook(
            self._initialize_kernel_hook,
            with_kwargs=True,
        )

    def _initialize_kernel_hook(self, module: Any, args: Any, kwargs: Any) -> None:
        if self.quant_method:
            self.quant_method.process_weights_after_loading(self)
        self._init_hook_handle.remove()

    def forward(
        self,
        input_: Any,
        **kwargs: Any,
    ) -> Any:
        _set_hunyuan_row_parallel_linear_forward_context(input_.shape[0])
        flash_comm_v1_enabled, mmrs_fusion, pad_size = _set_hunyuan_row_parallel_linear_ascend_forward_context()
        if self.custom_op is not None:
            return self.custom_op.apply(input_)
        try:
            return super().forward(input_, **kwargs)
        finally:
            _EXTRA_CTX.flash_comm_v1_enabled = flash_comm_v1_enabled
            _EXTRA_CTX.mmrs_fusion = mmrs_fusion
            _EXTRA_CTX.pad_size = pad_size