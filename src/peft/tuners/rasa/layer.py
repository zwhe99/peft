# Copyright 2023-present the HuggingFace Inc. team.
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
from __future__ import annotations

import math
import warnings
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.utils.imports import is_xpu_available
from torch import svd_lowrank
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.integrations import dequantize_module_weight, gather_params_ctx, get_bnb_param_type
from peft.utils.other import transpose

from .config import RasaConfig

class LastDimWeighting(nn.Module):
    def __init__(self, feature_dim):
        super(LastDimWeighting, self).__init__()
        self.weight = nn.Parameter(torch.ones(feature_dim))

    def forward(self, x):
        return x * self.weight

class RasaLayer(BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("rasa_private_A", "rasa_private_B", "rasa_private_D", "rasa_shared_A", "rasa_shared_B")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("r", "rasa_alpha", "scaling", "rasa_dropout")

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        self.r = {}
        self.effective_r = {}
        self.rasa_k = {}
        self.rasa_alpha = {}
        self.scaling = {}
        self.rasa_dropout = nn.ModuleDict({})
        self.rasa_d_init = {}
        self.rasa_private_A = nn.ModuleDict({})
        self.rasa_private_B = nn.ModuleDict({})
        self.rasa_private_D = nn.ModuleDict({})

        # will be assigned in `update_layer`
        self.rasa_shared_A: nn.ModuleDict = None
        self.rasa_shared_B: nn.ModuleDict = None
        self.module_name: str = None

        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        self._caches: dict[str, Any] = {}
        self.kwargs = kwargs

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, nn.Conv2d):
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
        elif isinstance(base_layer, nn.Embedding):
            in_features, out_features = base_layer.num_embeddings, base_layer.embedding_dim
        elif isinstance(base_layer, Conv1D):
            in_features, out_features = (
                base_layer.weight.ds_shape if hasattr(base_layer.weight, "ds_shape") else base_layer.weight.shape
            )
        else:
            # possibly support user provided custom layer types using dynamic dispatch
            if hasattr(base_layer, "in_features") and hasattr(base_layer, "out_features"):
                in_features, out_features = base_layer.in_features, base_layer.out_features
            else:
                in_features, out_features = None, None
            warnings.warn(
                f"Unsupported layer type '{type(base_layer)}' encountered, proceed at your own risk.", UserWarning
            )

        self.in_features = in_features
        self.out_features = out_features

    def update_layer(
        self, adapter_name, module_name, rasa_shared_A, rasa_shared_B, r, effective_r, rasa_k, rasa_alpha, rasa_dropout, rasa_d_init
    ):
        # This code works for linear layers, override for other layer types
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        assert r - rasa_k > 0, "r - rasa_k should be greater than 0"

        self.r[adapter_name] = r
        self.effective_r[adapter_name] = effective_r
        self.rasa_k[adapter_name] = rasa_k
        self.rasa_alpha[adapter_name] = rasa_alpha
        self.module_name = module_name
        if rasa_dropout > 0.0:
            rasa_dropout_layer = nn.Dropout(p=rasa_dropout)
        else:
            rasa_dropout_layer = nn.Identity()

        self.rasa_dropout.update(nn.ModuleDict({adapter_name: rasa_dropout_layer}))
        self.rasa_d_init[adapter_name] = rasa_d_init

        # Actual trainable parameters
        self.rasa_private_A[adapter_name] = nn.Linear(self.in_features, r - rasa_k, bias=False)
        self.rasa_private_B[adapter_name] = nn.Linear(r - rasa_k, self.out_features, bias=False)
        self.rasa_private_D[adapter_name] = LastDimWeighting(effective_r)
        self.rasa_shared_A = rasa_shared_A # should have been initialized outside
        self.rasa_shared_B = rasa_shared_B # should have been initialized outside
        self.scaling[adapter_name] = 1.0   # rasa uses `rasa_private_D` for scaling
        self.reset_rasa_private_parameters(adapter_name)

        self._move_adapter_to_device_of_base_layer(adapter_name)

        self.set_adapter(self.active_adapters)

    def reset_rasa_private_parameters(self, adapter_name):
        if adapter_name in self.rasa_private_A.keys():
            # initialize A the same way as the default for nn.Linear and B to zero
            # https://github.com/microsoft/LoRA/blob/a0a92e0f26c067cf94747bdbf1ce73793fa44d19/loralib/layers.py#L124
            nn.init.kaiming_uniform_(self.rasa_private_A[adapter_name].weight, a=math.sqrt(5))
            nn.init.zeros_(self.rasa_private_B[adapter_name].weight)

            # initialize D
            r = self.r[adapter_name]
            effective_r = self.effective_r[adapter_name]
            rasa_k = self.rasa_k[adapter_name]
            alpha = self.rasa_alpha[adapter_name]
            if self.rasa_d_init[adapter_name] == "default":
                nn.init.constant_(self.rasa_private_D[adapter_name].weight[:r - rasa_k], (0.5 * alpha) / (r - rasa_k))
                nn.init.constant_(self.rasa_private_D[adapter_name].weight[r - rasa_k:], (0.5 * alpha) / (effective_r - (r - rasa_k)))
            elif self.rasa_d_init[adapter_name] == "keep_scale":
                nn.init.constant_(self.rasa_private_D[adapter_name].weight, alpha / r)
            else:
                raise ValueError(f"Invalid rasa_d_init value: {self.rasa_d_init[adapter_name]}")

    def _cache_store(self, key: str, value: Any) -> None:
        self._caches[key] = value

    def _cache_pop(self, key: str) -> Any:
        value = self._caches.pop(key)
        return value

    def set_scale(self, adapter, scale):
        if adapter not in self.scaling:
            # Ignore the case where the adapter is not in the layer
            return
        self.scaling[adapter] = scale

    def scale_layer(self, scale: float) -> None:
        if scale == 1:
            return

        for active_adapter in self.active_adapters:
            if active_adapter not in self.rasa_private_A.keys():
                continue

            self.scaling[active_adapter] *= scale

    def unscale_layer(self, scale=None) -> None:
        for active_adapter in self.active_adapters:
            if active_adapter not in self.rasa_private_A.keys():
                continue

            if scale is None:
                self.scaling[active_adapter] = 1.0
            else:
                self.scaling[active_adapter] /= scale

    def _check_forward_args(self, x, *args, **kwargs):
        """Check if the arguments are compatible with the configs and state of the model"""
        adapter_names = kwargs.get("adapter_names", None)
        if adapter_names is None:
            return

        if len(x) != len(adapter_names):
            msg = (
                "Length of `adapter_names` should be the same as the number of inputs, but got "
                f"{len(adapter_names)} and {len(x)} respectively."
            )
            raise ValueError(msg)

        if self.merged:
            # It is unclear what would be the right thing to do if users pass adapter_names and there are merged
            # adapters. Therefore, it is better to raise an error in this case.
            msg = "Cannot pass `adapter_names` when there are merged adapters, please call `unmerge_adapter` first."
            raise ValueError(msg)


# Below code is based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# and modified to work with PyTorch FSDP


#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


class Linear(nn.Module, RasaLayer):
    # Rasa implemented in a dense layer
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        module_name: str,
        rasa_shared_A: nn.ModuleDict,
        rasa_shared_B: nn.ModuleDict,
        r: int = 0,
        effective_r: int = 0,
        rasa_k: int = 0,
        rasa_alpha: int = 1,
        rasa_dropout: float = 0.0,
        rasa_d_init: str = "default",
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        RasaLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            module_name,
            rasa_shared_A,
            rasa_shared_B,
            r,
            effective_r,
            rasa_k,
            rasa_alpha,
            rasa_dropout,
            rasa_d_init,
        )
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.rasa_private_A.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    delta_weight = self.get_delta_weight(active_adapter)
                    orig_weights += delta_weight

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights
                else:
                    delta_weight = self.get_delta_weight(active_adapter)
                    base_layer.weight.data += delta_weight

                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.rasa_private_A.keys():
                weight = self.get_base_layer().weight
                delta_weight = self.get_delta_weight(active_adapter)
                weight.data -= delta_weight

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.rasa_private_B[adapter].weight.device
        dtype = self.rasa_private_B[adapter].weight.dtype

        # In case users wants to merge the adapter weights that are in
        # (b)float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # (b)float16 because some CPUs have slow bf16/fp16 matmuls.
        cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)

        weight_private_A = self.rasa_private_A[adapter].weight
        weight_private_B = self.rasa_private_B[adapter].weight
        weight_private_D = self.rasa_private_D[adapter].weight
        weight_shared_A = self.rasa_shared_A[adapter][self.module_name].weight
        weight_shared_B = self.rasa_shared_B[adapter][self.module_name].weight

        if cast_to_fp32:
            weight_private_A = weight_private_A.float()
            weight_private_B = weight_private_B.float()
            weight_private_D = weight_private_D.float()
            weight_shared_A = weight_shared_A.float()
            weight_shared_B = weight_shared_B.float()

        output_tensor = transpose(
            (torch.cat([weight_private_B, weight_shared_B], dim=1) * weight_private_D) @ torch.cat([weight_private_A, weight_shared_A], dim=0),
            self.fan_in_fan_out
        ) * self.scaling[adapter]

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.rasa_private_A[adapter].weight.data = weight_private_A.to(dtype)
            self.rasa_private_B[adapter].weight.data = weight_private_B.to(dtype)
            self.rasa_private_D[adapter].weight.data = weight_private_D.to(dtype)
            self.rasa_shared_A[adapter][self.module_name].weight.data = weight_shared_A.to(dtype)
            self.rasa_shared_B[adapter][self.module_name].weight.data = weight_shared_B.to(dtype)

        return output_tensor

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            raise NotImplementedError("Mixed batch forward is not supported for Rasa.")
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype
            for active_adapter in self.active_adapters:
                if active_adapter not in self.rasa_private_A.keys():
                    continue
                rasa_private_A = self.rasa_private_A[active_adapter]
                rasa_private_B = self.rasa_private_B[active_adapter]
                rasa_private_D = self.rasa_private_D[active_adapter]
                rasa_shared_A = self.rasa_shared_A[active_adapter][self.module_name]
                rasa_shared_B = self.rasa_shared_B[active_adapter][self.module_name]

                dropout = self.rasa_dropout[active_adapter]
                scaling = self.scaling[active_adapter]

                x = x.to(rasa_private_A.weight.dtype)

                result = result + F.linear(
                    rasa_private_D(
                        F.linear(
                            dropout(x),
                            torch.cat([rasa_private_A.weight, rasa_shared_A.weight], dim=0),
                        )
                    ),
                    torch.cat([rasa_private_B.weight, rasa_shared_B.weight], dim=1)
                ) * scaling

            result = result.to(torch_result_dtype)

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "rasa." + rep


class Embedding(nn.Module, RasaLayer):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Embedding is not supported for Rasa.")

class Conv2d(nn.Module, RasaLayer):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Conv2d is not supported for Rasa.")

def dispatch_default(
    target: torch.nn.Module,
    adapter_name: str,
    rasa_config: RasaConfig,
    **kwargs,
) -> Optional[torch.nn.Module]:
    new_module = None

    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    if isinstance(target_base_layer, torch.nn.Embedding):
        embedding_kwargs = kwargs.copy()
        embedding_kwargs.pop("fan_in_fan_out", None)
        new_module = Embedding(target, adapter_name, **embedding_kwargs)
    elif isinstance(target_base_layer, torch.nn.Conv2d):
        new_module = Conv2d(target, adapter_name, **kwargs)
    elif isinstance(target_base_layer, torch.nn.Linear):
        if kwargs["fan_in_fan_out"]:
            warnings.warn(
                "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                "Setting fan_in_fan_out to False."
            )
            kwargs["fan_in_fan_out"] = rasa_config.fan_in_fan_out = False
        new_module = Linear(target, adapter_name, **kwargs)
    elif isinstance(target_base_layer, Conv1D):
        if not kwargs["fan_in_fan_out"]:
            warnings.warn(
                "fan_in_fan_out is set to False but the target module is `Conv1D`. " "Setting fan_in_fan_out to True."
            )
            kwargs["fan_in_fan_out"] = rasa_config.fan_in_fan_out = True
        new_module = Linear(target, adapter_name, is_target_conv_1d_layer=True, **kwargs)

    return new_module
