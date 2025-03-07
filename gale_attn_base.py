import torch
import torch.nn.functional as F
from torch import nn

def forward_mit_v2(self, x, stride_h=2, stride_w=2):
    B, _, H, W = x.shape
    
    subtensors = [
        x[:, :, h::stride_h, w::stride_w] 
        for h in range(stride_h) 
        for w in range(stride_w)
    ]
    subtensors.append(F.interpolate(x, scale_factor=1 / max(stride_h, stride_w), mode='bilinear', align_corners=False))
    
    attention_maps = []
    for sub_x in subtensors:
        qkv = self.qkv(sub_x)
        multi_scale_qkv = [qkv]
        for op in self.aggreg:
            multi_scale_qkv.append(op(qkv))
        multi_scale_qkv = torch.cat(multi_scale_qkv, dim=1)
        B, _, h, w = sub_x.shape
        multi_scale_qkv = multi_scale_qkv.reshape(B, -1, 3 * self.dim, h * w).transpose(-1, -2)
        q, k, v = multi_scale_qkv.chunk(3, dim=-1)

        q = self.kernel_func(q)
        k = self.kernel_func(k)
        v = F.pad(v, (0, 1), mode="constant", value=1.0)

        if not torch.jit.is_scripting():
            with torch.autocast(device_type=v.device.type, enabled=False):
                out = self._attn(q, k, v)
        else:
            out = self._attn(q, k, v)

        out = out.transpose(-1, -2).reshape(B, -1, h, w)
        
        out = self.proj(out)
        attention_maps.append(out)
    
    output = torch.zeros((B, attention_maps[0].shape[1], H, W), device=x.device, dtype=x.dtype)
    for i, sub_x in enumerate(attention_maps[:-1]):
        h, w = divmod(i, stride_w)
        output[:, :, h::stride_h, w::stride_w] = sub_x
    
    output = output * 0.8 + 0.2 * F.interpolate(attention_maps[-1], size=(H, W), mode='bilinear', align_corners=False)
    
    return output

def forward_mb4(self, x, attn_mask=None, stride_h=2, stride_w=2):

    def forward_mb4_original(self, x, attn_mask = None):
        """Run layer computation."""
        B, C, H, W = s = x.shape

        q = self.query(x)
        q = self._reshape_projected_query(q, self.num_heads, self.key_dim)
        k = self.key(x)
        k = self._reshape_input(k)
        v = self.value(x)
        v = self._reshape_input(v)
        # breakpoint()
        if self.einsum:
            attn = torch.einsum('blhk,bpk->blhp', q, k) * self.scale
            if attn_mask is not None:
                # NOTE: assumes mask is float and in correct shape
                attn = attn + attn_mask
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            o = torch.einsum('blhp,bpk->blhk', attn, v)
        else:
            if False:
                o = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attn_mask,
                    dropout_p=self.attn_drop.p if self.training else 0.
                )
            else:
                q = q * self.scale
                attn = q @ k.transpose(-1, -2)
                if attn_mask is not None:
                    # NOTE: assumes mask is float and in correct shape
                    attn = attn + attn_mask
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)
                o = attn @ v
        o = self._reshape_output(o, self.num_heads, H // self.query_strides[0], W // self.query_strides[1])
        x = self.output(o)
        return x
    
    B, _, H, W = x.shape
    subtensors = [
        x[:, :, h::stride_h, w::stride_w] 
        for h in range(stride_h) 
        for w in range(stride_w)
    ]
    subtensors.append(F.interpolate(x, scale_factor=1 / max(stride_h, stride_w), mode='bilinear', align_corners=False))
    
    attention_maps = [forward_mb4_original(self, sub_x, attn_mask) for sub_x in subtensors]
    
    output = torch.zeros((B, attention_maps[0].shape[1], H, W), device=x.device, dtype=x.dtype)
    for i, sub_x in enumerate(attention_maps[:-1]):
        h, w = divmod(i, stride_w)
        output[:, :, h::stride_h, w::stride_w] = sub_x
    
    output = output * 0.8 + 0.2 * F.interpolate(attention_maps[-1], size=(H, W), mode='bilinear', align_corners=False)
    # output = F.interpolate(attention_maps[-1], size=(H, W), mode='bilinear', align_corners=False) #for ablation study
    
    return output
    

def forward_fastvit_sliced(self, x, stride_h=2, stride_w=2):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W
        x = x.flatten(2).transpose(-2, -1)  # (B, N, C)
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)

        if self.fused_attn:
            x = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.transpose(-2, -1).reshape(B, C, H, W)

        return x
    
    B, _, H, W = x.shape
    
    subtensors = [
        x[:, :, h::stride_h, w::stride_w] 
        for h in range(stride_h) 
        for w in range(stride_w)
    ]
    subtensors.append(F.interpolate(x, scale_factor=1 / max(stride_h, stride_w), mode='bilinear', align_corners=False))
    attention_maps = [forward(self, sub_x) for sub_x in subtensors]
    
    output = torch.zeros((B, attention_maps[0].shape[1], H, W), device=x.device, dtype=x.dtype)
    for i, sub_x in enumerate(attention_maps[:-1]):
        h, w = divmod(i, stride_w)
        output[:, :, h::stride_h, w::stride_w] = sub_x
    
    output = output * 0.8 + 0.2 * F.interpolate(attention_maps[-1], size=(H, W), mode='bilinear', align_corners=False)
    
    return output
    

def scaled_dot_product_attention4(query, key, value, scale=None, attention_mask=None, split_factor=4):

    hidden_states = torch.empty_like(query)

    for i in range(split_factor):
        query_part = query[..., i::split_factor, :]
        key_part = key[..., i::split_factor, :]
        value_part = value[..., i::split_factor, :]

        hidden_states_part = F.scaled_dot_product_attention(
            query_part, key_part, value_part, scale=scale, attn_mask=attention_mask
        )

        hidden_states[..., i::split_factor, :] = hidden_states_part

    def downsample(x, factor=split_factor):
        b, h, s, d = x.shape  # batch, heads, seq_len, dim
        x = x.permute(0, 1, 3, 2)  # (b, h, d, s) to make seq_len the last dim
        x = F.avg_pool1d(x.reshape(b * h * d, 1, s), kernel_size=factor, stride=factor).reshape(b, h, d, -1)
        return x.permute(0, 1, 3, 2)  # Back to (b, h, seq_len, d)

    down_q = downsample(query)
    down_k = downsample(key)
    down_v = downsample(value)


    refined_output = F.scaled_dot_product_attention(down_q, down_k, down_v, scale=scale, attn_mask=attention_mask)
    refined_output = F.interpolate(refined_output, scale_factor=(split_factor,1), mode='bilinear', align_corners=False)

    if hidden_states.shape != refined_output.shape:
        print("[WARN] Tokenization was made by Satan himself")
    else:
        hidden_states = hidden_states * .5 + refined_output * 0.5

    return hidden_states

def forward_vit_sliced(self, x: torch.Tensor, stride_h=2, stride_w=1) -> torch.Tensor:
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = self.q_norm(q), self.k_norm(k)
    x = scaled_dot_product_attention4(
        q, k, v, split_factor=stride_h * stride_w, scale=self.scale)

    x = x.transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x
