# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""LLaMA model configuration"""

from transformers.models.llama.configuration_llama import LlamaConfig


class LlamaLadderConfig(LlamaConfig):

    model_type = "llamaLadder"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        ladder_layers=None,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )
        if ladder_layers is None:
            self.ladder_layers = []
        elif isinstance(ladder_layers, int):
            self.ladder_layers = list(range(self.num_hidden_layers - ladder_layers, self.num_hidden_layers))
        elif isinstance(ladder_layers, list):
            self.ladder_layers = ladder_layers
        else:
            raise ValueError(f"Invalid ladder_layers type: {type(ladder_layers)}")

        print(f"Ladder layers: {self.ladder_layers}")
