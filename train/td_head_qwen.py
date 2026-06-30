"""
Qwen3-Omni-Style Classifier Head

Implements GQA (Grouped Query Attention) + MoE FFN + DecoderLayer,
matching Qwen3-Omni-30B-A3B's attention architecture.

Architecture params (matching Qwen3-Omni):
  H = 2048 (hidden_size)
  num_heads = 32, num_kv_heads = 4, head_dim = 128
  num_experts = 128, top_k = 8, intermediate_size = 768
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (x * self.weight).to(dtype)


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------

def precompute_freqs_cis(dim: int, max_seq_len: int, theta: float = 1000000.0):
    """Precompute rotary position embedding frequencies."""
    freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    pos = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(pos, freq)  # [max_seq_len, dim/2]
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis  # [max_seq_len, dim/2]


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
):
    """Apply rotary embeddings to query and key tensors.

    Args:
        xq: [B, nH, T, head_dim]
        xk: [B, nKV, T, head_dim]
        freqs_cis: [T, head_dim/2]
    Returns:
        xq_out, xk_out with RoPE applied
    """
    T = xq.shape[2]
    freqs_cis = freqs_cis.to(xq.device)

    # Reshape xq/xk as complex numbers
    xq_r, xq_i = xq.float().reshape(*xq.shape[:-1], -1, 2).unbind(-1)
    xk_r, xk_i = xk.float().reshape(*xk.shape[:-1], -1, 2).unbind(-1)

    freqs_real = freqs_cis.real  # [T, dim/2]
    freqs_imag = freqs_cis.imag  # [T, dim/2]

    # reshape for broadcasting: [T, dim/2] -> [1, 1, T, dim/2]
    freqs_real = freqs_real[None, None, :, :]
    freqs_imag = freqs_imag[None, None, :, :]

    xq_out_r = xq_r * freqs_real - xq_i * freqs_imag
    xq_out_i = xq_r * freqs_imag + xq_i * freqs_real
    xk_out_r = xk_r * freqs_real - xk_i * freqs_imag
    xk_out_i = xk_r * freqs_imag + xk_i * freqs_real

    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(-2)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(-2)

    return xq_out.type_as(xq), xk_out.type_as(xk)


# ---------------------------------------------------------------------------
# Grouped Query Attention
# ---------------------------------------------------------------------------

class GroupedQueryAttention(nn.Module):
    """GQA with RoPE, matching Qwen3-Omni architecture.

    Params:
        hidden_size = 2048
        num_heads = 32, num_kv_heads = 4, head_dim = 128
        q_proj: 2048 -> 4096 (32*128)
        k_proj: 2048 -> 512 (4*128)
        v_proj: 2048 -> 512 (4*128)
        o_proj: 4096 -> 2048
    """
    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 32,
        num_kv_heads: int = 4,
        head_dim: int = 128,
        rope_theta: float = 1000000.0,
        max_seq_len: int = 8192,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_key_value_groups = num_heads // num_kv_heads

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        self.q_norm = RMSNorm(head_dim, eps=1e-6)
        self.k_norm = RMSNorm(head_dim, eps=1e-6)

        # Precompute RoPE frequencies
        freqs_cis = precompute_freqs_cis(head_dim, max_seq_len, rope_theta)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """Forward pass for GQA.

        Args:
            hidden_states: [B, T, H]
            past_key_value: optional (key, value) from previous steps
            use_cache: whether to return updated KV cache
            attention_mask: [B, S] 1=valid, 0=padding (applied to key positions)
        Returns:
            output: [B, T, H]
            new_kv: (key, value) if use_cache else None
        """
        B, T, _ = hidden_states.shape

        # Project
        q = self.q_proj(hidden_states)  # [B, T, nH * hd]
        k = self.k_proj(hidden_states)  # [B, T, nKV * hd]
        v = self.v_proj(hidden_states)  # [B, T, nKV * hd]

        # Reshape
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)     # [B, nH, T, hd]
        k = k.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)  # [B, nKV, T, hd]
        v = v.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)  # [B, nKV, T, hd]

        # Apply QK norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # RoPE with past_len offset for KV cache streaming
        past_len = past_key_value[0].size(2) if past_key_value is not None else 0

        if past_len + T > self.freqs_cis.size(0):
            raise ValueError(
                f"RoPE max_seq_len exceeded: need {past_len + T}, "
                f"but max_seq_len={self.freqs_cis.size(0)}"
            )

        freqs_cis = self.freqs_cis[past_len:past_len + T]
        q, k = apply_rotary_emb(q, k, freqs_cis)

        # KV cache
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)

        new_kv = (k, v) if use_cache else None

        # GQA: expand KV heads to match Q heads
        k = k.repeat_interleave(self.num_key_value_groups, dim=1)
        v = v.repeat_interleave(self.num_key_value_groups, dim=1)

        # Causal mask (position-based, safe with KV cache)
        S = k.size(2)
        past_len = S - T
        q_pos = torch.arange(past_len, past_len + T, device=q.device).unsqueeze(1)
        k_pos = torch.arange(0, S, device=q.device).unsqueeze(0)
        causal_mask = torch.zeros((T, S), device=q.device, dtype=q.dtype)
        causal_mask = causal_mask.masked_fill(k_pos > q_pos, torch.finfo(q.dtype).min)

        # Attention
        attn_weights = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        attn_weights = attn_weights + causal_mask[None, None, :, :]

        # Padding mask: mask out padding key positions
        if attention_mask is not None:
            # attention_mask only covers current input: [B, T]
            # past positions are assumed valid
            if past_key_value is not None:
                past_valid = torch.ones(
                    B,
                    past_len,
                    device=attention_mask.device,
                    dtype=attention_mask.dtype,
                )
                key_mask = torch.cat([past_valid, attention_mask], dim=1)  # [B, S]
            else:
                key_mask = attention_mask  # [B, T]

            key_mask = key_mask[:, None, None, :]  # [B, 1, 1, S]
            attn_weights = attn_weights.masked_fill(
                key_mask == 0,
                torch.finfo(q.dtype).min,
            )

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)

        attn_output = torch.matmul(attn_weights, v)  # [B, nH, T, hd]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, -1)
        output = self.o_proj(attn_output)

        return output, new_kv


# ---------------------------------------------------------------------------
# MoE FFN
# ---------------------------------------------------------------------------

class MoEFFN(nn.Module):
    """MoE FFN with 128 experts, top-8 routing.

    Architecture:
        router: 2048 -> 128
        Each expert: gate_proj(2048->768), up_proj(2048->768), down_proj(768->2048)
        Top-8 experts selected per token.

    Uses per-expert dispatch to avoid OOM from gather-all approach.
    """
    def __init__(
        self,
        hidden_size: int = 2048,
        intermediate_size: int = 768,
        num_experts: int = 128,
        top_k: int = 8,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k

        self.router = nn.Linear(hidden_size, num_experts, bias=False)

        # Expert weights stored as nn.ParameterList for per-expert dispatch
        self.experts = nn.ModuleList([
            nn.ModuleDict({
                'gate_proj': nn.Linear(hidden_size, intermediate_size, bias=False),
                'up_proj': nn.Linear(hidden_size, intermediate_size, bias=False),
                'down_proj': nn.Linear(intermediate_size, hidden_size, bias=False),
            })
            for _ in range(num_experts)
        ])

        with torch.no_grad():
            for expert in self.experts:
                nn.init.normal_(expert['gate_proj'].weight, mean=0.0, std=0.02)
                nn.init.normal_(expert['up_proj'].weight, mean=0.0, std=0.02)
                nn.init.normal_(expert['down_proj'].weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """MoE FFN forward with per-expert dispatch.

        Args:
            x: [B, T, H]
        Returns:
            output: [B, T, H]
        """
        B, T, H = x.shape
        dtype = x.dtype

        # Routing
        router_logits = self.router(x)  # [B, T, num_experts]
        router_probs = F.softmax(router_logits.float(), dim=-1)  # [B, T, num_experts]
        topk_weights, topk_indices = torch.topk(router_probs, self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)  # renormalize
        topk_weights = topk_weights.to(dtype)
        topk_indices = topk_indices.to(torch.long)

        # Flatten batch and sequence dims
        flat_x = x.view(-1, H)         # [N, H] where N = B*T
        flat_weights = topk_weights.view(-1, self.top_k)  # [N, top_k]
        flat_indices = topk_indices.view(-1, self.top_k)  # [N, top_k]

        N = flat_x.size(0)

        # Output accumulator
        output = torch.zeros(N, H, device=x.device, dtype=dtype)

        # Per-expert dispatch: for each expert, find which (token, slot) pairs
        # route to it, compute only those, and accumulate with weights
        for expert_idx in range(self.num_experts):
            # Find all (token_idx, slot_idx) where this expert is selected
            # flat_indices[t, s] == expert_idx
            mask = (flat_indices == expert_idx)  # [N, top_k]
            if not mask.any():
                continue

            # Get (token_idx, slot_idx) pairs
            token_ids, slot_ids = mask.nonzero(as_tuple=True)
            weights = flat_weights[token_ids, slot_ids]  # [K]

            # Gather tokens assigned to this expert
            expert_input = flat_x[token_ids]  # [K, H]

            # Expert FFN: silu(gate(x)) * up(x) -> down
            expert = self.experts[expert_idx]
            h1 = expert['gate_proj'](expert_input)  # [K, I]
            h2 = expert['up_proj'](expert_input)    # [K, I]
            h = F.silu(h1) * h2                     # [K, I]
            expert_out = expert['down_proj'](h)     # [K, H]

            # Accumulate weighted output
            output.index_add_(0, token_ids, expert_out * weights[:, None])

        output = output.view(B, T, H)
        return output


# ---------------------------------------------------------------------------
# Dense SwiGLU FFN (non-MoE)
# ---------------------------------------------------------------------------

class DenseSwiGLUFFN(nn.Module):
    """Dense FFN version, non-MoE.

    Qwen3-style SwiGLU:
        gate_proj: 2048 -> 6144
        up_proj:   2048 -> 6144
        down_proj: 6144 -> 2048

    FFN(x) = down_proj(silu(gate_proj(x)) * up_proj(x))
    """
    def __init__(
        self,
        hidden_size: int = 2048,
        intermediate_size: int = 6144,
    ):
        super().__init__()

        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.gate_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.up_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.down_proj.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            F.silu(self.gate_proj(x)) * self.up_proj(x)
        )


# ---------------------------------------------------------------------------
# Decoder Layer
# ---------------------------------------------------------------------------

class DecoderLayer(nn.Module):
    """Single decoder layer: GQA + MoE FFN with residual connections.

    Architecture:
        x -> RMSNorm -> GQA -> residual
        -> RMSNorm -> MoE FFN -> residual
    """
    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 32,
        num_kv_heads: int = 4,
        head_dim: int = 128,
        rope_theta: float = 1000000.0,
        num_experts: int = 128,
        top_k: int = 8,
        intermediate_size: int = 768,
    ):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size, eps=1e-6)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=1e-6)

        self.self_attn = GroupedQueryAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            rope_theta=rope_theta,
        )

        self.moe = MoEFFN(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            top_k=top_k,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """Forward pass with optional KV cache.

        Args:
            hidden_states: [B, T, H]
            past_key_value: optional KV from previous steps
            use_cache: whether to output updated KV
            attention_mask: [B, S] 1=valid, 0=padding
        Returns:
            hidden_states: [B, T, H]
            new_kv: (key, value) if use_cache else None
        """
        residual = hidden_states
        h = self.input_layernorm(hidden_states)
        attn_out, new_kv = self.self_attn(h, past_key_value, use_cache, attention_mask)
        h = residual + attn_out

        residual = h
        h = self.post_attention_layernorm(h)
        ffn_out = self.moe(h)
        h = residual + ffn_out

        return h, new_kv


class DecoderLayerDense(nn.Module):
    """Single decoder layer: GQA + Dense SwiGLU FFN with residual connections.

    Architecture:
        x -> RMSNorm -> GQA -> residual
        -> RMSNorm -> Dense SwiGLU FFN -> residual
    """
    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 32,
        num_kv_heads: int = 4,
        head_dim: int = 128,
        rope_theta: float = 1000000.0,
        intermediate_size: int = 6144,
    ):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size, eps=1e-6)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=1e-6)

        self.self_attn = GroupedQueryAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            rope_theta=rope_theta,
        )

        self.mlp = DenseSwiGLUFFN(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """Forward pass with optional KV cache.

        Args:
            hidden_states: [B, T, H]
            past_key_value: optional KV from previous steps
            use_cache: whether to output updated KV
            attention_mask: [B, S] 1=valid, 0=padding
        Returns:
            hidden_states: [B, T, H]
            new_kv: (key, value) if use_cache else None
        """
        residual = hidden_states
        h = self.input_layernorm(hidden_states)
        attn_out, new_kv = self.self_attn(h, past_key_value, use_cache, attention_mask)
        h = residual + attn_out

        residual = h
        h = self.post_attention_layernorm(h)
        ffn_out = self.mlp(h)
        h = residual + ffn_out

        return h, new_kv


# ---------------------------------------------------------------------------
# Qwen Classifier Head
# ---------------------------------------------------------------------------


def init_qwen_style(module: nn.Module):
    """Unified initialization matching Qwen3 style."""
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class QwenClassifierHead(nn.Module):
    """Qwen3-Omni-style classifier head with GQA, FFN, and learned label token.

    Architecture:
        1. Learned label token [1, 1, 2048] appended to input
        2. N decoder layers (GQA + FFN)
        3. RMSNorm + Linear classifier on last token

    FFN types:
        - "moe":   MoE FFN (128 experts, top-8 routing)
        - "dense": Dense SwiGLU FFN (intermediate_size=6144)

    For streaming, supports KV cache across chunks.
    For training, processes full sequence at once.

    Params (matching Qwen3-Omni-30B-A3B):
        hidden_size = 2048
        num_labels = 4
        num_layers = 1 (configurable)
        num_heads = 32, num_kv_heads = 4, head_dim = 128
        ffn_type = "moe" or "dense"
    """
    def __init__(
        self,
        hidden_size: int = 2048,
        num_labels: int = 4,
        num_layers: int = 1,
        num_heads: int = 32,
        num_kv_heads: int = 4,
        head_dim: int = 128,
        rope_theta: float = 1000000.0,
        ffn_type: str = "moe",
        num_experts: int = 128,
        top_k: int = 8,
        intermediate_size: int = 768,
        dense_intermediate_size: int = 6144,
    ):
        super().__init__()
        assert ffn_type in ["moe", "dense"], f"Unsupported ffn_type: {ffn_type}"

        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.num_layers = num_layers
        self.ffn_type = ffn_type

        # Learned label token [1, 1, hidden_size]
        self.label_token = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)

        # Decoder layers
        if ffn_type == "dense":
            self.layers = nn.ModuleList([
                DecoderLayerDense(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    rope_theta=rope_theta,
                    intermediate_size=dense_intermediate_size,
                )
                for _ in range(num_layers)
            ])
        else:
            self.layers = nn.ModuleList([
                DecoderLayer(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    rope_theta=rope_theta,
                    num_experts=num_experts,
                    top_k=top_k,
                    intermediate_size=intermediate_size,
                )
                for _ in range(num_layers)
            ])

        # Final norm and classifier
        self.norm = RMSNorm(hidden_size, eps=1e-6)
        self.classifier = nn.Linear(hidden_size, num_labels)

        # Unified initialization
        self.apply(init_qwen_style)
        nn.init.normal_(self.label_token, mean=0.0, std=0.02)

    def forward(
        self,
        audio_hidden: torch.Tensor,
        audio_mask: torch.Tensor | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
        drop_label_from_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list]:
        """Forward pass.

        Args:
            audio_hidden: [B, A, H] audio encoder hidden states
            audio_mask: [B, A] 1=valid, 0=padding
            past_key_values: list of (key, value) per layer for streaming
            use_cache: whether to return updated KV caches
            drop_label_from_cache: if True, strip the label token from returned
                KV caches so that past_kv only contains audio positions.
                Useful for streaming so the label token doesn't accumulate.
        Returns:
            If use_cache:
                turn_logits: [B, num_labels]
                new_kvs: list of (key, value) per layer
            Else:
                turn_logits: [B, num_labels]
        """
        B, A, H = audio_hidden.shape

        # Expand label token to batch size and append to sequence
        # [audio_hidden; label_token] -> [B, A+1, H]
        label = self.label_token.expand(B, 1, H)  # [B, 1, H]
        x = torch.cat([audio_hidden, label], dim=1)  # [B, A+1, H]

        # Build attention_mask: [B, A+1] — label token is always valid
        if audio_mask is not None:
            label_mask = torch.ones(B, 1, device=audio_mask.device, dtype=audio_mask.dtype)
            attention_mask = torch.cat([audio_mask, label_mask], dim=1)  # [B, A+1]
        else:
            attention_mask = None

        # Process through decoder layers
        new_kvs = []
        for i, layer in enumerate(self.layers):
            pkv = past_key_values[i] if past_key_values is not None else None
            x, kv = layer(x, past_key_value=pkv, use_cache=use_cache, attention_mask=attention_mask)
            new_kvs.append(kv)

        # Take the last token (label token position) for classification
        h = x[:, -1, :]  # [B, H]

        # Final norm and classifier
        h = self.norm(h)
        logits = self.classifier(h)  # [B, num_labels]

        if use_cache:
            # Optionally strip label token position from KV caches
            if drop_label_from_cache and new_kvs and new_kvs[0] is not None:
                new_kvs = [
                    (kv[0][:, :, :-1, :], kv[1][:, :, :-1, :])
                    for kv in new_kvs
                ]
            return logits, new_kvs
        return logits