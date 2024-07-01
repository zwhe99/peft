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
from dataclasses import dataclass, field
from typing import List, Optional, Union

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class OlaConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`OlaModel`].

    Paper: https://arxiv.org/abs/2310.11454.

    Args:
        r (`int`, *optional*, defaults to `256`):
            Ola parameter dimension ("rank"). Choose higher values than LoRA ranks here, since Ola uses far fewer
            parameters than LoRA (see Table 1).
        target_modules (`Union[List[str], str]`):
            The names of the modules to apply Ola to. Only linear layers are supported.
        projection_prng_key (`int`):
            Ola PRNG init key. Used for initialising ola_A and ola_B for new models or when loading a checkpoint
            that did not include these projections. Defaults to `0`.
        save_projection (`bool`):
            Whether to save the ola_A / ola_B projections in the state dict alongside per layer lambda_b / lambda_d
            weights. This will increase the size of the checkpoint, but guarantee that we can reload the checkpoint on
            all system configurations. Defaults to `True`.
        ola_dropout (`float`):
            The dropout probability for Ola layers.
        fan_in_fan_out (`bool`):
            Set this to True if the layer to replace stores weight like (fan_in, fan_out). For example, gpt-2 uses
            `Conv1D` which stores weights like (fan_in, fan_out) and hence this should be set to `True`.
        bias (`str`):
            Bias type for Ola. Can be 'none', 'all' or 'ola_only'. If 'all' or 'ola_only', the corresponding biases
            will be updated during training. Be aware that this means that, even when disabling the adapters, the model
            will not produce the same output as the base model would have without adaptation.
        modules_to_save (`List[str]`):
            List of modules apart from Ola layers to be set as trainable and saved in the final checkpoint.
        layers_to_transform (`Union[List[int],int]`):
            The layer indexes to transform, if this argument is specified, it will apply the Ola transformations on
            the layer indexes that are specified in this list. If a single integer is passed, it will apply the Ola
            transformations on the layer at this index.
        layers_pattern (`str`):
            The layer pattern name, used only if `layers_to_transform` is different from `None` and if the layer
            pattern is not in the common layers pattern.
    """

    r: int = field(default=256, metadata={"help": "Ola attention dimension"})
    effective_r: int = field(default=256, metadata={"help": "Ola attention dimension"})

    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "List of module names or regex expression of the module names to replace with Ola."
                "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'. "
                "Only linear layers are supported."
            )
        },
    )
    ola_use_scaling: bool = field(default=True, metadata={"help": "Use scaling in Ola"})
    ola_share_scaling: bool = field(default=False, metadata={"help": "Share scaling factor in Ola"})
    ola_alpha: int = field(default=8, metadata={"help": "Ola alpha"})
    ola_dropout: float = field(default=0.0, metadata={"help": "Ola dropout"})
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    bias: str = field(default="none", metadata={"help": "Bias type for Ola. Can be 'none', 'all' or 'ola_only'"})
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": (
                "List of modules apart from Ola layers to be set as trainable and saved in the final checkpoint. For"
                " example, in Sequence Classification or Token Classification tasks, the final layer"
                " `classifier/score` are randomly initialized and as such need to be trainable and saved."
            )
        },
    )
    layers_to_transform: Optional[Union[List[int], int]] = field(
        default=None,
        metadata={
            "help": (
                "The layer indexes to transform, is this argument is specified, PEFT will transform only the layers"
                " indexes that are specified inside this list. If a single integer is passed, PEFT will transform only"
                " the layer at this index."
            )
        },
    )
    layers_pattern: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The layer pattern name, used only if `layers_to_transform` is different to None and if the layer"
                " pattern is not in the common layers pattern."
            )
        },
    )

    def __post_init__(self):
        self.peft_type = PeftType.OLA
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )