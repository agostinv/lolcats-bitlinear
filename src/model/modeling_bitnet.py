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
"""
Thin wrappers and replacement classes for BitNetForCausalLM
"""
from typing import Optional, Tuple, List, Union

import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling_bitnet import BitNetModel, BitNetForCausalLM
from masking_utils import create_causal_mask

from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.cache_utils import Cache, DynamicCache
from transformers.utils import (
    add_start_docstrings_to_model_forward,
    logging,
)

from .convert_model import get_attention_cache

logger = logging.get_logger(__name__)


# Modified from transformers.models.llama.modeling_llama.LlamaModel (v4.43)
class LolcatsBitnetModel(BitNetModel):
    """
    Wrapper for BitNet / Llama-like transformer model

    Modified from a KeyFormer version of BitNetModel
    -> Only difference is using KV state for past_key_values instead of cache
    """

    def __init__(self, *args: any, **kwargs: any):
        super().__init__(*args, **kwargs)
        self.layerwise_cpu = False

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            if past_key_values is None or isinstance(
                past_key_values, DynamicCache
            ):  # Determine and setup our KV cache or state
                attention_type = getattr(
                    self.layers[0].self_attn, "attention_type", None
                )
                past_key_values = get_attention_cache(attention_type)
            else:
                past_key_values.get_usable_length(seq_length)

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position: torch.Tensor = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_attentions = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            hidden_states = outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (outputs[1],)

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            attentions=all_attentions,
        )


class LolcatsBitNetForCausalLM(BitNetForCausalLM):
    """
    Wrapper for BitNet-like autoregressive language model
    """

    def __init__(self, config):
        # Adapt config to LlamaConfig
        if getattr(config, "attention_bias", None) is None:
            config.attention_bias = False
        if getattr(config, "rope_scaling", None) is None:
            config.rope_scaling = None
        if getattr(config, "pretraining_tp", None) is None:
            config.pretraining_tp = 1
        super().__init__(config)
        self.model = LolcatsBitNetModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self, *args: any, labels: Optional[torch.LongTensor] = None, **kwargs: any
    ):
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        # outputs = self.model(*args, **kwargs)
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            cache_position=cache_position,
            **kwargs,
        )  # pylint: disable=E1102

        hidden_states = outputs.last_hidden_state
        # if False:  # getattr(self.model.layers[0].self_attn, 'train_attention', False):
        #     logits = None  # MZ 8/25: Sorry, was trying stuff
        # regular training
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(
                self.vocab_size // self.config.pretraining_tp, dim=0
            )
            logits = [
                F.linear(hidden_states, lm_head_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            logits = torch.cat(logits, dim=-1)
        else:
            # another strange precision issue
            hidden_states = hidden_states.to(torch.bfloat16)
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
