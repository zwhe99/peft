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

import re
import math
import warnings
from dataclasses import asdict
from enum import Enum
from typing import Optional, Union
from collections import defaultdict

import torch
import torch.nn as nn
from torch.nn.init import _calculate_correct_fan
from tqdm import tqdm
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer, check_target_module_exists
from peft.utils import (
    TRANSFORMERS_MODELS_TO_ORA_TARGET_MODULES_MAPPING,
    ModulesToSaveWrapper,
    _get_submodules,
)

from ..tuners_utils import _maybe_include_all_linear_layers
from .config import OraConfig
from .layer import Linear, OraLayer

class OraModel(BaseTuner):
    """
    Creates Vector-based Random Matrix Adaptation (Ora) model from a pretrained transformers model.

    Args:
        model ([`~transformers.PreTrainedModel`]): The model to be adapted.
        config ([`OraConfig`]): The configuration of the Ora model.
        adapter_name (`str`): The name of the adapter, defaults to `"default"`.

    Returns:
        `torch.nn.Module`: The Ora model.

    Example:

        ```py
        >>> from transformers import AutoModelForCausalLM
        >>> from peft import OraConfig, get_peft_model

        >>> base_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
        >>> config = OraConfig(r=128)
        >>> model = get_peft_model(base_model, config)
        ```

    **Attributes**:
        - **model** ([`~transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`OraConfig`]): The configuration of the Ora model.
    """

    prefix: str = "ora_"

    def __init__(self, model, config, adapter_name) -> None:
        super().__init__(model, config, adapter_name)

    def _scan_module(self, config) -> dict[str, dict[str, dict[str, Union[tuple[int, int], list[int], int]]]]:
        """
        Scans the model's modules and groups them based on their type and shape.

        Args:
            config: The configuration object.

        Returns:
            A dictionary containing the grouped modules. Each module type is a key in the dictionary,
            and the corresponding value is a dictionary with the following keys:
                - "shape": The shape of the modules.
                - "layer_ids": A list of layer IDs for the modules.
                - "num_layers": The number of layers for the module type.

        Raises:
            ValueError: If no layers types compatible with Ora were found.
            AssertionError: If there is a shape mismatch for any module type across layers.
        """
        model_config = getattr(self.model, "config", {"model_type": "custom"})
        if hasattr(model_config, "to_dict"):
            model_config = model_config.to_dict()

        peft_config = self._prepare_adapter_config(config, model_config)
        peft_config = _maybe_include_all_linear_layers(peft_config, self.model)

        module2shape = {}
        for key, module in self.model.named_modules():
            if not self._check_target_module_exists(peft_config, key):
                continue

            if isinstance(module, (nn.Linear, Conv1D)):
                module_shape = tuple(module.weight.shape)
                if isinstance(module, Conv1D):
                    module_shape = module_shape[::-1]
            else:
                continue

            module2shape[key] = module_shape

        if not module2shape:
            msg = "No layers types compatible with Ora were found. Please check `peft_config.target_modules`."
            raise ValueError(msg)

        # Group modules across layers
        grouped_dict = defaultdict(list)
        pattern = re.compile(r'layers\.(\d+)\.(.+)')

        for key, value in module2shape.items():
            match = pattern.search(key)
            if match:
                layer_id = match.group(1)
                module_name = match.group(2).replace('.', '__')
                grouped_dict[module_name].append((layer_id, value))

        grouped_dict = dict(grouped_dict)

        # Assert each type of module has the same shape across layers
        for key, value in grouped_dict.items():
            assert all([v[1] == value[0][1] for v in value]), f"Shape mismatch for {key} layers: {value}"

        # Add the number of layers for each module type
        for key in grouped_dict.keys():
            grouped_dict[key] = {
                "shape": grouped_dict[key][0][1],
                "layer_ids": [int(v[0]) for v in grouped_dict[key]],
                "num_layers": len(grouped_dict[key]),
            }

        return grouped_dict

    def _init_ora_A_ora_B(self, config: OraConfig, adapter_name: str) -> None:
        self.ora_A = nn.ModuleDict({})
        self.ora_B = nn.ModuleDict({})
        self.ora_indices_A = {}
        self.ora_indices_B = {}

        module2shape = self._scan_module(config)
        for key, value in module2shape.items():
            in_dim = value["shape"][1]
            out_dim = value["shape"][0]
            num_layers = value["num_layers"]
            layer_ids = value["layer_ids"]
            assert (config.r * num_layers) % config.effective_r == 0, "effective_r should be a factor of r * num_layers"

            para_buget = (in_dim * config.r + config.r * out_dim) * num_layers
            in_and_out_buget = para_buget // config.effective_r

            in_buget = int(in_and_out_buget * (in_dim / (in_dim + out_dim)))
            out_buget = in_and_out_buget - in_buget

            if adapter_name not in self.ora_A:
                self.ora_A[adapter_name] = nn.ModuleDict({})

            if adapter_name not in self.ora_B:
                self.ora_B[adapter_name] = nn.ModuleDict({})

            self.ora_A[adapter_name][key] = nn.Linear(in_buget, config.effective_r, bias=False)
            self.ora_B[adapter_name][key] = nn.Linear(config.effective_r, out_buget, bias=False)

            nn.init.kaiming_uniform_(self.ora_A[adapter_name][key].weight, a=math.sqrt(5))
            nn.init.zeros_(self.ora_B[adapter_name][key].weight)

            offset_A = (in_buget - in_dim) // (num_layers - 1)
            offset_B = (out_buget - out_dim) // (num_layers - 1)

            if adapter_name not in self.ora_indices_A:
                self.ora_indices_A[adapter_name] = {}
            if adapter_name not in self.ora_indices_B:
                self.ora_indices_B[adapter_name] = {}

            for idx, layer_ids in enumerate(layer_ids):
                ora_indices_A = list(range(idx * offset_A, idx * offset_A + in_dim))
                ora_indices_B = list(range(idx * offset_B, idx * offset_B + out_dim))

                if layer_ids not in self.ora_indices_A[adapter_name]:
                    self.ora_indices_A[adapter_name][layer_ids] = {}
                if layer_ids not in self.ora_indices_B[adapter_name]:
                    self.ora_indices_B[adapter_name][layer_ids] = {}

                self.ora_indices_A[adapter_name][layer_ids][key] = ora_indices_A
                self.ora_indices_B[adapter_name][layer_ids][key] = ora_indices_B

    def _pre_injection_hook(self, model: nn.Module, config: OraConfig, adapter_name: str) -> None:
        self._init_ora_A_ora_B(config, adapter_name)

    def _check_new_adapter_config(self, config: OraConfig) -> None:
        """
        A helper method to check the config when a new adapter is being added.

        Raise a ValueError if there is something wrong with the config or if it conflicts with existing adapters.

        """
        # the below todo is copied from LoRA
        # TODO: there should be a check if any of the existing adapters actually has bias != "none", or else the check
        # does not fully correspond to the error message.
        if (len(self.peft_config) > 1) and (config.bias != "none"):
            raise ValueError(
                f"{self.__class__.__name__} supports only 1 adapter with bias. When using multiple adapters, "
                "set bias to 'none' for all adapters."
            )

        for existing_config in self.peft_config.values():
            if existing_config is config:
                # skip the current config
                continue

            if existing_config.projection_prng_key != config.projection_prng_key:
                raise ValueError(
                    f"Ora PRNG initialisation key must be the same for all adapters. Got {config.projection_prng_key=} but "
                    f"previous config had {existing_config.projection_prng_key}."
                )

    @staticmethod
    def _check_target_module_exists(ora_config, key):
        return check_target_module_exists(ora_config, key)

    def _create_and_replace(
        self,
        ora_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
        **optional_kwargs,
    ):
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        pattern = re.compile(r'layers\.(\d+)\.(.+)')
        match = pattern.search(current_key)
        if match:
            layer_id = int(match.group(1))
            module_name = match.group(2).replace('.', '__')
        else:
            raise ValueError("Invalid target module type")

        r = ora_config.r
        bias = hasattr(target, "bias") and target.bias is not None
        kwargs = {
            "r": r,
            "ora_dropout": ora_config.ora_dropout,
            "fan_in_fan_out": ora_config.fan_in_fan_out,
            "init_weights": ora_config.init_weights,
        }
        kwargs["bias"] = bias

        if isinstance(target, Linear):
            target.update_layer(
                adapter_name,
                layer_id,
                module_name,
                self.ora_indices_A,
                self.ora_indices_B,
                self.ora_A,
                self.ora_B,
                r,
                ora_config.ora_dropout,
                ora_config.init_weights,
                d_initial=ora_config.d_initial,
            )
        else:
            new_module = self._create_new_module(ora_config, layer_id, module_name, self.ora_indices_A, self.ora_indices_B, self.ora_A, self.ora_B, adapter_name, target, **kwargs)
            if adapter_name not in self.active_adapter:
                # adding an additional adapter: it is not automatically trainable
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _replace_module(parent, child_name, new_module, child):
        setattr(parent, child_name, new_module)
        # It's not necessary to set requires_grad here, as that is handled by
        # _mark_only_adapters_as_trainable

        # child layer wraps the original module, unpack it
        if hasattr(child, "base_layer"):
            child = child.base_layer

        if not hasattr(new_module, "base_layer"):
            new_module.weight = child.weight
            if hasattr(child, "bias"):
                new_module.bias = child.bias

        if getattr(child, "state", None) is not None:
            if hasattr(new_module, "base_layer"):
                new_module.base_layer.state = child.state
            else:
                new_module.state = child.state
            new_module.to(child.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if "ora_" in name:
                module.to(child.weight.device)

    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if self.prefix not in n:
                p.requires_grad = False

        for active_adapter in self.active_adapters:
            bias = self.peft_config[active_adapter].bias
            if bias == "none":
                continue

            if bias == "all":
                for n, p in model.named_parameters():
                    if "bias" in n:
                        p.requires_grad = True
            elif bias == "vera_only":
                for m in model.modules():
                    if isinstance(m, OraLayer) and hasattr(m, "bias") and m.bias is not None:
                        m.bias.requires_grad = True
            else:
                raise NotImplementedError(f"Requested bias: {bias}, is not implemented.")


    @staticmethod
    def _create_new_module(ora_config, layer_id, module_name, ora_indices_A, ora_indices_B, ora_A, ora_B, adapter_name, target, **kwargs):
        bias = kwargs.pop("bias", False)

        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if isinstance(target_base_layer, torch.nn.Linear):
            if kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                    "Setting fan_in_fan_out to False."
                )
                kwargs["fan_in_fan_out"] = ora_config.fan_in_fan_out = False
        elif isinstance(target_base_layer, Conv1D):
            kwargs["is_target_conv_1d_layer"] = True
            if not kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to False but the target module is `Conv1D`. "
                    "Setting fan_in_fan_out to True."
                )
                kwargs["fan_in_fan_out"] = ora_config.fan_in_fan_out = True
        else:
            raise ValueError(
                f"Target module {target} is not supported. Currently, only the following modules are supported: "
                "`torch.nn.Linear`, `transformers.pytorch_utils.Conv1D`."
            )
        new_module = Linear(
            target,
            layer_id,
            module_name,
            ora_indices_A,
            ora_indices_B,
            ora_A,
            ora_B,
            adapter_name,
            bias=bias,
            **kwargs,
        )

        return new_module

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    def get_peft_config_as_dict(self, inference: bool = False):
        config_dict = {}
        for key, value in self.peft_config.items():
            config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(value).items()}
            if inference:
                config["inference_mode"] = True
        config_dict[key] = config
        return config

    def _set_adapter_layers(self, enabled=True):
        for module in self.model.modules():
            if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper)):
                module.enable_adapters(enabled)

    def enable_adapter_layers(self):
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self):
        for active_adapter in self.active_adapters:
            val = self.peft_config[active_adapter].bias
            if val != "none":
                msg = (
                    f"Careful, disabling adapter layers with bias configured to be '{val}' does not produce the same "
                    "output as the the base model would without adaption."
                )
                warnings.warn(msg)
        self._set_adapter_layers(enabled=False)

    def set_adapter(self, adapter_name):
        for module in self.model.modules():
            if isinstance(module, OraLayer):
                if module.merged:
                    warnings.warn("Adapter cannot be set when the model is merged. Unmerging the model first.")
                    module.unmerge()
                module.set_adapter(adapter_name)
        self.active_adapter = adapter_name

    @staticmethod
    def _prepare_adapter_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_ORA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = set(
                TRANSFORMERS_MODELS_TO_ORA_TARGET_MODULES_MAPPING[model_config["model_type"]]
            )
        return peft_config

    def _unload_and_optionally_merge(
        self,
        merge=True,
        progressbar: bool = False,
        safe_merge: bool = False,
        adapter_names: Optional[list[str]] = None,
    ):
        # we cannot use self.prefix as we want to include non-trainable ora parameters
        key_list = [key for key, _ in self.model.named_modules() if "ora" not in key]
        desc = "Unloading " + ("and merging " if merge else "") + "model"
        for key in tqdm(key_list, disable=not progressbar, desc=desc):
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue

            if hasattr(target, "base_layer"):
                if merge:
                    target.merge(safe_merge=safe_merge, adapter_names=adapter_names)

                self._replace_module(parent, target_name, target.get_base_layer(), target)
            elif isinstance(target, ModulesToSaveWrapper):
                # save any additional trainable modules part of `modules_to_save`
                setattr(parent, target_name, target.modules_to_save[target.active_adapter])

        return self.model

    def delete_adapter(self, adapter_name: str):
        """
        Deletes an existing adapter.

        Args:
            adapter_name (str): Name of the adapter to be deleted.
        """
        if adapter_name not in list(self.peft_config.keys()):
            raise ValueError(f"Adapter {adapter_name} does not exist")
        del self.peft_config[adapter_name]

        # we cannot use self.prefix as we want to include non-trainable ora parameters
        key_list = [key for key, _ in self.model.named_modules() if "ora" not in key]
        new_adapter = None
        for key in key_list:
            _, target, _ = _get_submodules(self.model, key)
            if isinstance(target, OraLayer):
                target.delete_adapter(adapter_name)
                if new_adapter is None:
                    new_adapter = target.active_adapter[:]

        self.active_adapter = new_adapter or []

    def merge_and_unload(
        self, progressbar: bool = False, safe_merge: bool = False, adapter_names: Optional[list[str]] = None
    ):
        r"""
        This method merges the Ora layers into the base model. This is needed if someone wants to use the base model
        as a standalone model.

        Args:
            progressbar (`bool`):
                whether to show a progressbar indicating the unload and merge process
            safe_merge (`bool`):
                whether to activate the safe merging check to check if there is any potential Nan in the adapter
                weights
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.

        Example:

        ```py
        >>> from transformers import AutoModelForCausalLM
        >>> from peft import PeftModel

        >>> base_model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-40b")
        >>> peft_model_id = "smangrul/falcon-40B-int4-peft-lora-sfttrainer-sample"
        >>> model = PeftModel.from_pretrained(base_model, peft_model_id)
        >>> merged_model = model.merge_and_unload()
        ```
        """
        return self._unload_and_optionally_merge(
            progressbar=progressbar, safe_merge=safe_merge, adapter_names=adapter_names
        )

    def unload(self):
        """
        Gets back the base model by removing all the Ora modules without merging. This gives back the original base
        model.
        """
        return self._unload_and_optionally_merge(merge=False)
