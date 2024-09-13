import torch
from flash_attn.flash_attn_interface import (
    _flash_attn_forward,
    _flash_attn_varlen_forward,
    _flash_attn_backward,
    _flash_attn_varlen_backward,
)
from modules.profiler import pytorch_profiler as profiler
from itertools import accumulate


def run_flash_attention_fwd(params, is_varlen=True, verbose=False):
    head_dim = params["head_dim"]
    num_heads_q = params["num_heads_q"]
    num_heads_kv = params["num_heads_kv"]
    batch_size = params["batch_size"]
    seqlens_q = params["seqlens_q"]
    seqlens_kv = params["seqlens_kv"]
    print(seqlens_kv)
    assert len(seqlens_q) == batch_size
    assert len(seqlens_kv) == batch_size
    max_seqlen_q = max(seqlens_q)
    max_seqlen_kv = max(seqlens_kv)
    dropout_p = 0.17 if params["dropout"] else 0.0
    causal = params["causal"]
    is_training = params["is_training"]
    dtype = getattr(torch, params["dtype"])

    if is_varlen:
        q = torch.randn(
            sum(seqlens_q),
            num_heads_q,
            head_dim,
            device="cuda",
            dtype=dtype,
            requires_grad=is_training,
        )
        k = torch.randn(
            sum(seqlens_kv),
            num_heads_kv,
            head_dim,
            device="cuda",
            dtype=dtype,
            requires_grad=is_training,
        )
        v = torch.randn(
            sum(seqlens_kv),
            num_heads_kv,
            head_dim,
            device="cuda",
            dtype=dtype,
            requires_grad=is_training,
        )

        cu_seqlens_q = torch.tensor(
            [0] + list(accumulate(seqlens_q)), dtype=torch.int32, device="cuda"
        )
        cu_seqlens_kv = torch.tensor(
            [0] + list(accumulate(seqlens_kv)), dtype=torch.int32, device="cuda"
        )
    else:
        q = torch.randn(
            batch_size,
            seqlens_q[0],
            num_heads_q,
            head_dim,
            device="cuda",
            dtype=dtype,
            requires_grad=is_training,
        )
        k = torch.randn(
            batch_size,
            seqlens_kv[0],
            num_heads_kv,
            head_dim,
            device="cuda",
            dtype=dtype,
            requires_grad=is_training,
        )
        v = torch.randn(
            batch_size,
            seqlens_kv[0],
            num_heads_kv,
            head_dim,
            device="cuda",
            dtype=dtype,
            requires_grad=is_training,
        )

    softmax_scale = head_dim ** (-0.5)
    window_size = (-1, -1)  # TODO: not support
    alibi_slopes = None  # TODO: not support
    attn_mask = None  # TODO: not support
    return_softmax = False

    if is_varlen:
        return profiler(
            _flash_attn_varlen_forward,
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_kv,
            max_seqlen_q,
            max_seqlen_kv,
            dropout_p,
            softmax_scale,
            causal,
            window_size,
            alibi_slopes,
            return_softmax,
            verbose=verbose,
        )
    else:
        return profiler(
            _flash_attn_forward,
            q,
            k,
            v,
            dropout_p,
            softmax_scale,
            causal,
            window_size,
            alibi_slopes,
            attn_mask,
            return_softmax,
            verbose=verbose,
        )


def run_flash_attention_bwd(params, is_varlen=True, verbose=False):
    head_dim = params["head_dim"]
    num_heads_q = params["num_heads_q"]
    num_heads_kv = params["num_heads_kv"]
    batch_size = params["batch_size"]
    seqlens_q = params["seqlens_q"]
    seqlens_kv = params["seqlens_kv"]
    assert len(seqlens_q) == batch_size
    assert len(seqlens_kv) == batch_size
    max_seqlen_q = max(seqlens_q)
    max_seqlen_kv = max(seqlens_kv)
    dropout_p = 0.17 if params["dropout"] else 0.0
    causal = params["causal"]
    deterministic = params["deterministic"]
    is_training = params["is_training"]
    dtype = getattr(torch, params["dtype"])

    if is_varlen:
        q = torch.randn(
            sum(seqlens_q),
            num_heads_q,
            head_dim,
            device="cuda",
            dtype=dtype,
            requires_grad=is_training,
        )
        k = torch.randn(
            sum(seqlens_kv),
            num_heads_kv,
            head_dim,
            device="cuda",
            dtype=dtype,
            requires_grad=is_training,
        )
        v = torch.randn(
            sum(seqlens_kv),
            num_heads_kv,
            head_dim,
            device="cuda",
            dtype=dtype,
            requires_grad=is_training,
        )
        cu_seqlens_q = torch.tensor(
            [0] + list(accumulate(seqlens_q)), dtype=torch.int32, device="cuda"
        )
        cu_seqlens_kv = torch.tensor(
            [0] + list(accumulate(seqlens_kv)), dtype=torch.int32, device="cuda"
        )
    else:
        q = torch.randn(
            batch_size,
            seqlens_q[0],
            num_heads_q,
            head_dim,
            device="cuda",
            dtype=dtype,
            requires_grad=is_training,
        )
        k = torch.randn(
            batch_size,
            seqlens_kv[0],
            num_heads_kv,
            head_dim,
            device="cuda",
            dtype=dtype,
            requires_grad=is_training,
        )
        v = torch.randn(
            batch_size,
            seqlens_kv[0],
            num_heads_kv,
            head_dim,
            device="cuda",
            dtype=dtype,
            requires_grad=is_training,
        )

    out = torch.empty_like(q)
    dout = torch.empty_like(out)
    softmax_lse = torch.empty(
        (batch_size, num_heads_q, max_seqlen_q), device="cuda", dtype=dtype
    )
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    softmax_scale = head_dim ** (-0.5)
    window_size = (-1, -1)  # TODO: not support
    alibi_slopes = None  # TODO: not support
    attn_mask = None  # TODO: not support
    rng_state = torch.empty((2), dtype=torch.int64)

    if is_varlen:
        return profiler(
            _flash_attn_varlen_backward,
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            dq,
            dk,
            dv,
            cu_seqlens_q,
            cu_seqlens_kv,
            max_seqlen_q,
            max_seqlen_kv,
            dropout_p,
            softmax_scale,
            causal,
            window_size,
            alibi_slopes,
            deterministic,
            rng_state,
            verbose=verbose,
        )
    else:
        return profiler(
            _flash_attn_backward,
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            dq,
            dk,
            dv,
            dropout_p,
            softmax_scale,
            causal,
            window_size,
            alibi_slopes,
            attn_mask,
            deterministic,
            rng_state,
            verbose=verbose,
        )


# TODO: support qkv packed and fixlen
def dispatch_flash_attention_api(params, op_type, verbose=False):
    if op_type == "fwd":
        return run_flash_attention_fwd(params, is_varlen=True, verbose=verbose)
    elif op_type == "bwd":
        return run_flash_attention_bwd(params, is_varlen=True, verbose=verbose)

# def run_flash_attention(params):
#     head_dim = params["head_dim"]
#     num_heads_q = params["num_heads_q"]
#     num_heads_kv = params["num_heads_kv"]
#     batch_size = params["batch_size"]
#     seqlen_q = params["seqlen_q"]
#     seqlen_kv = params["seqlen_kv"]
#     dropout_p = 0.17 if params["dropout"] else 0.0
#     causal = params["causal"]
#     deterministic = params["deterministic"]
#     is_training = params["is_training"]
#     dtype = getattr(torch, params["dtype"])

#     q = torch.randn(
#         batch_size,
#         seqlen_q,
#         num_heads_q,
#         head_dim,
#         device="cuda",
#         dtype=dtype,
#         requires_grad=is_training,
#     )
#     k = torch.randn(
#         batch_size,
#         seqlen_kv,
#         num_heads_kv,
#         head_dim,
#         device="cuda",
#         dtype=dtype,
#         requires_grad=is_training,
#     )
#     v = torch.randn(
#         batch_size,
#         seqlen_kv,
#         num_heads_kv,
#         head_dim,
#         device="cuda",
#         dtype=dtype,
#         requires_grad=is_training,
#     )

#     return profiler(
#         flash_attn_func, q, k, v, dropout_p, None, causal, backward=is_training
#     )


# def run_varlen_flash_attention(params):
#     head_dim = params["head_dim"]
#     num_heads_q = params["num_heads_q"]
#     num_heads_kv = params["num_heads_kv"]
#     seqlens_q = params["seqlens_q"]
#     seqlens_kv = params["seqlens_kv"]
#     max_seqlen_q = max(seqlens_q)
#     max_seqlen_kv = max(seqlens_kv)
#     batch_size = len(seqlens_q)
#     dropout_p = 0.17 if params["dropout"] else 0.0
#     causal = params["causal"]
#     deterministic = params["deterministic"]
#     is_training = params["is_training"]
#     dtype = getattr(torch, params["dtype"])

#     q = torch.randn(
#         sum(seqlens_q),
#         num_heads_q,
#         head_dim,
#         device="cuda",
#         dtype=dtype,
#         requires_grad=is_training,
#     )
#     k = torch.randn(
#         sum(seqlens_kv),
#         num_heads_kv,
#         head_dim,
#         device="cuda",
#         dtype=dtype,
#         requires_grad=is_training,
#     )
#     v = torch.randn(
#         sum(seqlens_kv),
#         num_heads_kv,
#         head_dim,
#         device="cuda",
#         dtype=dtype,
#         requires_grad=is_training,
#     )

#     cu_seqlens_q = torch.tensor(
#         [0] + list(accumulate(seqlens_q)), dtype=torch.int32, device="cuda"
#     )
#     cu_seqlens_kv = torch.tensor(
#         [0] + list(accumulate(seqlens_kv)), dtype=torch.int32, device="cuda"
#     )

#     return profiler(
#         flash_attn_varlen_func,
#         q,
#         k,
#         v,
#         cu_seqlens_q,
#         cu_seqlens_kv,
#         max_seqlen_q,
#         max_seqlen_kv,
#         dropout_p,
#         None,
#         causal,
#         backward=is_training,
#     )
