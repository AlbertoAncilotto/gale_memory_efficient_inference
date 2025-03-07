
from typing import Optional
import torch
import time
import torch.nn.functional as F
from torch import nn
from diffusers.models import attention_processor
from diffusers.models.attention_processor import Attention
from .memory_efficient_attention import memory_efficient_attention

def replace_attn_processors_in_unet(unet, split_factor=1):
    if split_factor == 1:
        return
    for name, module in unet.named_modules():
        if hasattr(module, "attn1") and (isinstance(module.attn1.processor, attention_processor.AttnProcessor2_0) or isinstance(module.attn1.processor, AttnProcessor2_0)):
            module.attn1.processor = AttnProcessor2_0(split_factor=split_factor)
        if hasattr(module, "attn2") and (isinstance(module.attn2.processor, attention_processor.AttnProcessor2_0) or isinstance(module.attn1.processor, AttnProcessor2_0)):
            module.attn2.processor = AttnProcessor2_0(split_factor=split_factor)

printed=False
def scaled_dot_product_attention_full(query, key, value, scale=1.0, attention_mask=None):
    global printed
    query = query * scale
    attention_map = query @ key.transpose(-1, -2)
    if attention_mask is not None:
        # NOTE: assumes mask is float and in correct shape
        attention_map = attention_map + attention_mask
    attention_map = attention_map.softmax(dim=-1)
    hidden_states = attention_map @ value
    if not printed:
        print("Key shape", key.shape)
        print("Query shape", query.shape)
        print("Attn shape:", attention_map.shape)
        print("Value shape", value.shape)
        printed=True
    # breakpoint()

    return hidden_states

def scaled_dot_product_attention_GaLe_no_global(query, key, value, scale=1.0, attention_mask=None, split_factor=4):
    hit_dim = -2
    if key.size(hit_dim) > 100:
        hidden_states = torch.empty_like(query)
        for i in range(split_factor):
            query_part = query[..., i::split_factor, :]
            key_part = key[..., i::split_factor, :]
            value_part = value[..., i::split_factor, :]
            hidden_states[..., i::split_factor, :] = F.scaled_dot_product_attention(query_part, key_part, value_part, attn_mask=attention_mask, scale=scale)
    else:

        hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, scale=scale)

    return hidden_states

def scaled_dot_product_attention_GaLe_no_global_fast(query, key, value, scale=1.0, attention_mask=None, split_factor=4):
    hit_dim = -2
    if key.size(hit_dim) > 100:
        batch_size, num_heads, seq_len, dim = query.shape  # Assuming shape is (B, H, L, D)
        new_seq_len = seq_len // split_factor  # Assuming evenly divisible
        
        query_reshaped = query.view(batch_size, num_heads, split_factor, new_seq_len, dim)
        key_reshaped = key.view(batch_size, num_heads, split_factor, new_seq_len, dim)
        value_reshaped = value.view(batch_size, num_heads, split_factor, new_seq_len, dim)

        hidden_states = scaled_dot_product_attention_full(
            query_reshaped, key_reshaped, value_reshaped, scale=scale, attention_mask=attention_mask
        )

        hidden_states = hidden_states.view(batch_size, num_heads, seq_len, dim)
    else:
        hidden_states = scaled_dot_product_attention_full(
            query, key, value, scale=scale, attention_mask=attention_mask
        )

    return hidden_states



# Modified version of the original AttnProcessor2_0 to allow coparing the different attention mechanisms
class AttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self, split_factor=2, processor='gale', benchmark=False):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        # print('Usign custom AttnProcessor2_0')
        self.split_factor = split_factor
        self.processor = processor
        self.benchmark = benchmark
        # print('Using processor:', processor)

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
    ) -> torch.FloatTensor:
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        args = ()#(scale,)
        query = attn.to_q(hidden_states, *args)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, *args)
        value = attn.to_v(encoder_hidden_states, *args)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if self.benchmark:
            runtimes = []
            for _ in range(10):
                start_time = time.time()
                if self.processor == 'gale':
                    hidden_states = scaled_dot_product_attention_GaLe_no_global(query, key, value, attention_mask=attention_mask, scale=attn.scale, split_factor=self.split_factor)
                elif self.processor == 'full':
                    hidden_states = scaled_dot_product_attention_full(query, key, value, attention_mask=attention_mask, scale=attn.scale)
                elif self.processor == 'xformers':
                    hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, scale=attn.scale)
                elif self.processor == 'efficient':
                    if query.shape == key.shape:
                        hidden_states = memory_efficient_attention(query, key, value, mask=attention_mask, q_bucket_size=query.shape[-2]//self.split_factor, k_bucket_size=key.shape[-2]//self.split_factor)
                    else:
                        hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask)
                runtimes.append(time.time() - start_time)
            avg_runtime = sum(runtimes) / len(runtimes)
            print(f"Average runtime for processor '{self.processor}': {avg_runtime:.6f} seconds")
        else:
            if self.processor == 'gale':
                hidden_states = scaled_dot_product_attention_GaLe_no_global(query, key, value, attention_mask=attention_mask, scale=attn.scale, split_factor=self.split_factor)
            elif self.processor == 'full':
                hidden_states = scaled_dot_product_attention_full(query, key, value, attention_mask=attention_mask, scale=attn.scale)
            elif self.processor == 'xformers':
                hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, scale=attn.scale)
            elif self.processor == 'efficient':
                if query.shape == key.shape:
                    hidden_states = memory_efficient_attention(query, key, value, mask=attention_mask, q_bucket_size=query.shape[-2]//self.split_factor, k_bucket_size=key.shape[-2]//self.split_factor)
                else:
                    hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask)




        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states, *args)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
