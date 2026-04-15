# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import vllm.forward_context as _vllm_fc
from vllm.model_executor.layers.linear import RowParallelLinear
from vllm.utils.import_utils import resolve_obj_by_qualname

from vllm_omni.platforms import current_omni_platform


class HunyuanRowParallelLinearDefault(RowParallelLinear):
    def __init__(self, *args: Any, prefix: str = "", **kwargs: Any) -> None:
        super().__init__(*args, prefix=prefix, **kwargs)
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
        return super().forward(input_, **kwargs)


class HunyuanRowParallelLinear:
    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        op_name = "hunyuan_row_parallel_linear"
        current_omni_platform.prepare_diffusion_op_runtime(op_name)
        impl = resolve_obj_by_qualname(
            current_omni_platform.get_diffusion_model_impl_qualname(op_name),
        )
        return impl(*args, **kwargs)