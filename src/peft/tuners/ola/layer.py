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

import warnings
from typing import List, Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.other import transpose


class OlaLayer(BaseTunerLayer):
    # List all names of layers that may contain adapter weights
    adapter_layer_names = ("ola_A", "ola_B")
    other_param_names = ("r", "ola_alpha", "scaling", "ola_dropout", "ola_indices_A", "ola_indices_B")

    def __init__(self, base_layer: nn.Module, **kwargs):
        self.base_layer = base_layer
        self.r = {}
        self.ola_alpha = {}
        self.scaling = {}
        self.ola_dropout = nn.ModuleDict({})

        # Stores a reference to the ola_A/B ModuleDict.
        # Set to `None` otherwise to avoid computation with random weights
        self.ola_A: Optional[nn.ModuleDict] = None
        self.ola_B: Optional[nn.ModuleDict] = None
        self.ola_indices_A: Optional[dict] = None
        self.ola_indices_B: Optional[dict] = None

        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, Conv1D):
            in_features, out_features = (
                base_layer.weight.ds_shape if hasattr(base_layer.weight, "ds_shape") else base_layer.weight.shape
            )

        self.in_features = in_features
        self.out_features = out_features
        self.kwargs = kwargs

    def _move_adapter_to_device_of_base_layer(self, adapter_name: str, device: Optional[torch.device] = None) -> None:
        """
        Move the adapter of the given name to the device of the base layer.
        """
        from peft.tuners.vera.buffer_dict import BufferDict

        if device is None:
            # check weight and qweight (for GPTQ)
            for weight_name in ("weight", "qweight"):
                weight = getattr(self.get_base_layer(), weight_name, None)
                if weight is not None:
                    device = weight.device
                    dtype = weight.dtype
                    break
            else:
                # no break encountered: could not determine the device
                return

        # loop through all potential adapter layers and move them to the device of the base layer; be careful to only
        # move this specific adapter to the device, as the other adapters could be on different devices
        # see #1639
        for adapter_layer_name in self.adapter_layer_names + self.other_param_names:
            adapter_layer = getattr(self, adapter_layer_name, None)
            if not isinstance(adapter_layer, (nn.ModuleDict, nn.ParameterDict, BufferDict, dict)):
                continue
            if adapter_name not in adapter_layer:
                continue
            if adapter_layer_name in ["r", "ola_alpha", "scaling", "ola_dropout", "ola_indices_A", "ola_indices_B"]:
                continue
            if weight.dtype.is_floating_point or weight.dtype.is_complex:
                adapter_layer[adapter_name] = adapter_layer[adapter_name].to(device, dtype=dtype)
            else:
                adapter_layer[adapter_name] = adapter_layer[adapter_name].to(device)

    @property
    def merged(self) -> bool:
        return bool(self.merged_adapters)

    def update_layer(
        self,
        adapter_name,
        layer_id,
        module_name,
        ola_indices_A,
        ola_indices_B,
        ola_A: nn.ModuleDict,
        ola_B: nn.ModuleDict,
        r,
        effective_r,
        ola_use_scaling,
        ola_share_scaling,
        ola_alpha,
        ola_dropout,
    ):
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")
        if effective_r <= 0:
            raise ValueError(f"`effective_r` should be a positive integer value but the value passed is {effective_r}")
        self.r[adapter_name] = r
        self.ola_alpha[adapter_name] = ola_alpha
        if ola_dropout > 0.0:
            ola_dropout_layer = nn.Dropout(p=ola_dropout)
        else:
            ola_dropout_layer = nn.Identity()

        self.ola_dropout.update(nn.ModuleDict({adapter_name: ola_dropout_layer}))

        local_effective_r = ola_indices_A[adapter_name][layer_id][module_name][1] - ola_indices_A[adapter_name][layer_id][module_name][0]
        if ola_use_scaling:
            if ola_share_scaling:
                self.scaling[adapter_name] = ola_alpha / effective_r
            else:
                self.scaling[adapter_name] = ola_alpha / local_effective_r
        else:
            self.scaling[adapter_name] = 1.0
        self.layer_id = layer_id
        self.module_name = module_name
        self.ola_indices_A = ola_indices_A
        self.ola_indices_B = ola_indices_B
        self.ola_A = ola_A
        self.ola_B = ola_B

        if adapter_name not in ola_A:
            raise NotImplementedError("Adapter name not found in the provided ola_A")

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

class Linear(nn.Linear, OlaLayer):
    # Ola implemented in a dense layer
    def __init__(
        self,
        base_layer,
        layer_id,
        module_name,
        ola_indices_A,
        ola_indices_B,
        ola_A: nn.ModuleDict,
        ola_B: nn.ModuleDict,
        adapter_name: str,
        r: int = 0,
        effective_r: int = 0,
        ola_use_scaling: bool = True,
        ola_share_scaling: bool = False,
        ola_alpha: int = 1,
        ola_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        **kwargs,
    ) -> None:
        # this gets the init from nn.Linear's super perspective, i.e. nn.Module.__init__, which should always be called
        super(nn.Linear, self).__init__()
        OlaLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name
        self.update_layer(adapter_name, layer_id, module_name, ola_indices_A, ola_indices_B, ola_A, ola_B, r, effective_r, ola_use_scaling, ola_share_scaling, ola_alpha, ola_dropout)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
        raise NotImplementedError("Merging is not supported for OlaLayer yet.")

    def unmerge(self) -> None:
        raise NotImplementedError("Unmerging is not supported for OlaLayer yet.")

    def get_delta_weight(self, adapter) -> torch.Tensor:
        raise NotImplementedError("Getting delta weight is not supported for OlaLayer yet.")

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            for active_adapter in self.active_adapters:
                if active_adapter not in self.ola_A.keys():
                    continue

                if getattr(self, "ola_A_weight", None) is None:
                    ola_A = self.ola_A[active_adapter][self.module_name]
                    ola_B = self.ola_B[active_adapter][self.module_name]
                    A_start, A_end = self.ola_indices_A[active_adapter][self.layer_id][self.module_name]
                    B_start, B_end = self.ola_indices_B[active_adapter][self.layer_id][self.module_name]
                    self.ola_A_weight = ola_A.weight[A_start: A_end, :]
                    self.ola_B_weight = ola_B.weight[:, B_start: B_end]

                dropout = self.ola_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = x.to(self.ola_A_weight.dtype)
                result = result + F.linear(F.linear(dropout(x), self.ola_A_weight), self.ola_B_weight) * scaling

        result = result.to(previous_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "ola." + rep
