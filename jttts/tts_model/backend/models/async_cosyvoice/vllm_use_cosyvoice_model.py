# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/qwen2/modeling_qwen2.py
# Copyright 2024 The Qwen team.
# Copyright 2023 The vLLM team.
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
"""Inference-only Qwen2 model compatible with HuggingFace weights."""
from typing import Iterable, List, Optional, Set, Tuple, Union, Iterator, overload, TypedDict, Mapping, Any
from typing_extensions import TypeVar

import torch
from torch import nn
import pdb
from vllm.attention import AttentionMetadata
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from vllm.model_executor.models.interfaces import T, SupportsLoRA, SupportsPP
from vllm.model_executor.models.qwen2 import Qwen2Model

from vllm.model_executor.models.utils import AutoWeightsLoader, maybe_prefix, merge_multimodal_embeddings
import math
from vllm.model_executor.models.utils import default_weight_loader

logger = init_logger(__name__)

IGNORE_ID = -1
import logging as log
log.basicConfig(level=log.INFO,format='[ %(levelname)s %(asctime)s %(filename)s:%(lineno)d ] %(message)s')

import uuid
import hashlib


# *****************************************************************************************************

class CosyVoice3Model(nn.Module, SupportsLoRA, SupportsPP):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        # RAS Sampling Configuration
        self.use_ras = True
        self.ras_win_size = 10
        self.ras_tau_r = 0.1


        self.config = config
        self.lora_config = lora_config
        self.quant_config = quant_config

        self.llm_input_size = 896
        self.llm_output_size = 896
        self.speech_token_size = 6561  # 原始语音 token 数量
        self.extended_speech_vocab_size = self.speech_token_size + 200  # CosyVoice3 总语音 vocab
        self.llm_token_size = config.vocab_size  # Qwen2 的文本 vocab size

        # 特殊 token 定义（全部在 extended 语音空间内）
        self.sos_token_id = self.speech_token_size + 0
        self.eos_token_id = self.speech_token_size + 1
        self.task_token_id = self.speech_token_size + 2
        self.fill_token_id = self.speech_token_size + 3
        self.ras_pad_token_id = -1  # 明确的填充标识，不在语音token范围内

        self.allow_patterns_overrides = ["llm.*", "speech_embedding.*"]


        # Qwen2 LLM backbone
        self.model = Qwen2Model(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))

        # Decoder: 输出整个 extended 语音 vocab（无 bias）
        self.llm_decoder = ParallelLMHead(
            self.extended_speech_vocab_size,
            self.llm_output_size,
            bias=False,  # CosyVoice3 使用无 bias
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "llm_decoder")
        )

        self.logits_processor = LogitsProcessor(self.extended_speech_vocab_size)

        # 所有 token（包括特殊 token）都使用同一个 speech_embedding
        self.speech_embedding = torch.nn.Embedding(self.extended_speech_vocab_size, self.llm_input_size)


        self.sampler = get_sampler()
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

        self.mix_ratio: List[int] = [5, 15]

        # 缓冲区（用于 zero token，但 CosyVoice3 似乎未使用？保留兼容）
        self.zero_embed_buffer = torch.zeros(
            (vllm_config.scheduler_config.max_num_seqs, self.llm_input_size),
            dtype=self.speech_embedding.weight.dtype,
            device=self.speech_embedding.weight.device
        )
        self.inputs_embed_buffer = torch.zeros(
            (vllm_config.scheduler_config.max_num_batched_tokens, self.llm_input_size),
            dtype=self.speech_embedding.weight.dtype,
            device=self.speech_embedding.weight.device,
        )

        # 用于区分文本 token 和语音 token 的偏移量
        # 文本 token ID 将被映射为: original_id + self.text_token_offset
        self.text_token_offset = self.extended_speech_vocab_size

        # Stop token IDs for sampling (all special tokens in the 200 range)
        self.stop_token_ids = set(range(self.speech_token_size, self.extended_speech_vocab_size))


        # === RAS GPU 缓存机制（用于 vLLM 0.7.3 兼容）===
        self._ras_cache = {
            "seq_output_tokens": None,  # [B, max_len], cuda
            "seq_lens_tensor": None,   # [B], cuda
            "last_output_token_ids_hash": None,  # hash for invalidation
        }



    def get_sos_emb(self):
        return self.speech_embedding.weight[self.sos_token_id].reshape(1, 1, -1)

    def get_task_emb(self):
        return self.speech_embedding.weight[self.task_token_id].reshape(1, 1, -1)

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[T] = None,
        attn_metadata: Optional["AttentionMetadata"] = None,
    ) -> torch.Tensor:
        """
        input_ids 设计：
          - [0, extended_speech_vocab_size): 语音 token（含特殊 token）
          - [extended_speech_vocab_size, ...): 文本 token（原始 Qwen2 ID + offset）
        """
        input_shape = input_ids.shape
        flat_input_ids = input_ids.view(-1)

        # 判断是否为语音 token（包括特殊 token）
        is_speech = flat_input_ids < self.extended_speech_vocab_size

        inputs_embeds = self.inputs_embed_buffer[:flat_input_ids.shape[0]]
        inputs_embeds.zero_()

        # 处理语音 token（含 sos/eos/task/fill）
        if is_speech.any():
            speech_ids = flat_input_ids[is_speech]
            inputs_embeds[is_speech] = self.speech_embedding(speech_ids)

        # 处理文本 token
        if (~is_speech).any():
            text_ids = flat_input_ids[~is_speech] - self.text_token_offset
            # 确保 text_ids 在合法范围内
            assert (text_ids >= 0).all() and (text_ids < self.llm_token_size).all(), \
                f"Text token out of range: min={text_ids.min()}, max={text_ids.max()}, vocab={self.llm_token_size}"
            inputs_embeds[~is_speech] = self.model.get_input_embeddings(text_ids)

        inputs_embeds = inputs_embeds.view(*input_shape, self.llm_input_size)

        if multimodal_embeddings is not None:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, multimodal_embeddings, self.config.audio_token_index
            )

        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings(input_ids, attn_metadata=attn_metadata)
        return self.model(input_ids, positions, kv_caches, attn_metadata, intermediate_tensors, inputs_embeds)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.llm_decoder, hidden_states, sampling_metadata)
        return logits
		


    # def sample(
    #     self,
    #     logits: torch.Tensor,
    #     sampling_metadata: SamplingMetadata,
    # ) -> Optional[SamplerOutput]:
    #     next_tokens = self.sampler(logits, sampling_metadata)
    #     return next_tokens

    # def sample(
    #     self,
    #     logits: torch.Tensor,
    #     sampling_metadata: SamplingMetadata,
    # ) -> Optional[SamplerOutput]:
    #     next_tokens = self.sampler(logits, sampling_metadata)
    #     if next_tokens in sampling_metadata.output_token_ids[:-10]:
    #         sampling_metadata.top_p = torch.tensor([1.0], device=sampling_metadata.temperature.device)
    #         sampling_metadata.top_k = torch.tensor([3000], dtype=torch.int32, device=sampling_metadata.temperature.device)
    #         next_tokens = self.sampler(logits, sampling_metadata)
    #     return next_tokens


    # def _apply_ras_strategy(
    #     self,
    #     outputs: "SamplerOutput",
    #     logits: torch.Tensor,
    #     sampling_metadata: SamplingMetadata
    # ):
    #     """
    #     Apply RAS strategy to post-process the sampling output.
    #     If repetition is detected, resample using random sampling.
    #     """
    #     # sampling_metadata.output_token_ids is List[List[int]]
    #     # outputs.sampled_token_ids is [num_reqs, 1] tensor
    #     num_requests = len(sampling_metadata.output_token_ids)
        
    #     for req_idx in range(num_requests):
    #         # 1. Get the newly generated token
    #         new_token_id = outputs.sampled_token_ids[req_idx, 0].item()
            
    #         # 2. Get history tokens
    #         history_tokens = sampling_metadata.output_token_ids[req_idx]
            
    #         # 3. RAS Repetition Detection Logic
    #         # Check the last win_size tokens
    #         win_history = history_tokens[-self.ras_win_size:]
    #         if len(win_history) > 0:
    #             # Count repetitions
    #             rep_num = win_history.count(new_token_id)
                
    #             # 4. If threshold is triggered, perform Random Sampling
    #             if rep_num >= self.ras_win_size * self.ras_tau_r:
    #                 # Get the corresponding logits row
    #                 logit_row = logits[req_idx]
                    
    #                 # Apply Temperature
    #                 temp = sampling_metadata.temperature[req_idx].item()
    #                 if temp < 1e-5:
    #                     temp = 1.0
                    
    #                 # Resample
    #                 # Use softmax + multinomial to implement random sampling
    #                 probs = torch.softmax(logit_row / temp, dim=-1)
    #                 new_token_tensor = torch.multinomial(probs, num_samples=1)
    #                 replacement_token_id = new_token_tensor.item()
                    
    #                 # 5. Replace the result
    #                 outputs.sampled_token_ids[req_idx, 0] = replacement_token_id
    
    # def sample(
    #     self,
    #     logits: torch.Tensor,
    #     sampling_metadata: SamplingMetadata,
    # ) -> Optional[SamplerOutput]:
    #     next_tokens = self.sampler(logits, sampling_metadata)
    
    #     # Apply RAS strategy
    #     if self.use_ras and next_tokens is not None:
    #         # logger.debug(f"sampling_metadata after: {repr(sampling_metadata)}")
    #         self._apply_ras_strategy(next_tokens, logits, sampling_metadata)
        
    #     return next_tokens





    # ***************use fast ras*****************
    @torch.no_grad()
    def _build_ras_tensors(
        self,
        output_token_ids: List[List[int]],
        device: torch.device,
        dtype: torch.dtype = torch.long,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build padded GPU tensors from output_token_ids.
        
        Returns:
            seq_output_tokens: [B, win_size], padded with ras_pad_token_id
            seq_lens_tensor: [B], actual lengths (0 for empty sequences)
        """
        B = len(output_token_ids)
        if B == 0:
            # 修改：空序列时显式重置缓存哈希
            self._ras_cache["last_output_token_ids_hash"] = None
            return (
                torch.empty((0, self.ras_win_size), dtype=dtype, device=device),
                torch.empty(0, dtype=torch.long, device=device),
            )

        # Get actual lengths and max length
        seq_lens = [len(seq) for seq in output_token_ids]
        max_len = max(seq_lens) if seq_lens else 0

        # Always pad to ras_win_size (not max_len!) for consistent RAS window ops
        # This avoids shape mismatch in _fast_ras_apply when max_len == 0 or < win_size
        win_size = self.ras_win_size

        # Pad each sequence to at least win_size, but only keep last win_size tokens
        # (for memory efficiency & semantic correctness: RAS only cares about recent window)
        padded = []
        for seq in output_token_ids:
            if len(seq) >= win_size:
                # Keep only the most recent `win_size` tokens (LIFO)
                padded.append(seq[-win_size:])
            else:
                # Pad left with zeros: [0,0,...,0, seq...]
                # So index 0 always refers to oldest in window, index -1 to newest
                padded_seq = [self.ras_pad_token_id] * (win_size - len(seq)) + seq
                padded.append(padded_seq)

        # Convert to tensor: [B, win_size]
        seq_output_tokens = torch.tensor(padded, dtype=dtype, device=device)
        seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.long, device=device)

        return seq_output_tokens, seq_lens_tensor
        
    @torch.no_grad()
    def _fast_ras_apply(
        self,
        sampled_tokens: torch.LongTensor,
        logits: torch.FloatTensor,
        seq_output_tokens: torch.LongTensor,
        seq_lens: torch.LongTensor,
        temperature: Optional[torch.FloatTensor],
    ) -> torch.LongTensor:
        # === 防御性输入检查 ===
        if seq_output_tokens.ndim != 2 or seq_output_tokens.size(1) != self.ras_win_size:
            log.warning(
                f"RAS skipped: expected seq_output_tokens.shape=[B, {self.ras_win_size}], "
                f"got {seq_output_tokens.shape}. Returning original sampled tokens."
            )
            return sampled_tokens

        B = sampled_tokens.size(0)
        if B == 0:
            return sampled_tokens

        win_size = self.ras_win_size
        tau_threshold = int(math.ceil(win_size * self.ras_tau_r))

        # 构造有效位置掩码（左填充，有效 token 在右侧）
        arange = torch.arange(win_size, device=seq_output_tokens.device).unsqueeze(0)  # [1, win_size]
        valid_mask = arange >= (win_size - seq_lens).unsqueeze(1)  # [B, win_size]

        # 统计 sampled_token 在有效窗口内的出现次数
        sampled_expanded = sampled_tokens.unsqueeze(1).expand(-1, win_size)  # [B, win_size]
        match = (seq_output_tokens == sampled_expanded) & valid_mask
        rep_counts = match.sum(dim=1)  # [B]

        need_resample = rep_counts >= tau_threshold
        if not need_resample.any():
            return sampled_tokens

        # 执行重采样
        logits_rs = logits[need_resample]
        temp_rs = temperature[need_resample] if temperature is not None else None
        if temp_rs is not None:
            temp_rs = torch.clamp(temp_rs, min=1e-5)
            logits_rs = logits_rs / temp_rs.unsqueeze(-1)
        probs = torch.softmax(logits_rs, dim=-1)
        new_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
        new_tokens = new_tokens.to(sampled_tokens.dtype)

        result = sampled_tokens.clone()
        result[need_resample] = new_tokens
        return result

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        outputs = self.sampler(logits, sampling_metadata)
        if not self.use_ras or outputs is None:
            return outputs

        sampled_tokens = outputs.sampled_token_ids.flatten() # outputs.sampled_token_ids.shape=torch.Size([1, 1])
        B = sampled_tokens.size(0)
        if B == 0:
            return outputs

        # --- Lazy build GPU history tensors ---
        # 修改：使用加密级哈希替换内置hash，降低冲突风险
        hash_obj = hashlib.sha256()
        for seq in sampling_metadata.output_token_ids:
            hash_obj.update(str(seq).encode('utf-8'))
        current_hash = hash_obj.hexdigest()
        
        cache = self._ras_cache
        device = sampled_tokens.device

        if (cache["last_output_token_ids_hash"] != current_hash or
            cache["seq_output_tokens"] is None):

            seq_out, seq_len = self._build_ras_tensors(
                sampling_metadata.output_token_ids, device=device
            )
            cache.update({
                "seq_output_tokens": seq_out,
                "seq_lens_tensor": seq_len,
                "last_output_token_ids_hash": current_hash,
            })

        # --- Apply RAS ---
        new_tokens = self._fast_ras_apply(
            sampled_tokens=sampled_tokens,
            logits=logits,
            seq_output_tokens=cache["seq_output_tokens"],
            seq_lens=cache["seq_lens_tensor"],
            temperature=sampling_metadata.temperature,
        )

        outputs.sampled_token_ids.copy_(new_tokens.unsqueeze(-1))
        return outputs


    @staticmethod
    def convert_weights(weights: Iterable[Tuple[str, torch.Tensor]]) -> Iterable[Tuple[str, torch.Tensor]]:
        for name, param in weights:
            #print(name)
            if name.startswith("llm."):
                if name.startswith("llm.model.model."):
                    name = name.replace("llm.model.model.", "model.")
                else:
                    continue
            yield name, param

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):

        print(">>> Loading weights in CosyVoice3Model...")
        print(f">>> Weights type: {type(weights)}")

        weights = self.convert_weights(weights)
        loader = AutoWeightsLoader(self)
        loader.load_weights(weights)