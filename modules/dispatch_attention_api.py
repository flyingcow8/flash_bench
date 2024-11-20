from FlashBenchData.KernelType import KernelType
import torch
from flash_attn.flash_attn_interface import (
    _flash_attn_forward,
    _flash_attn_varlen_forward,
    _flash_attn_backward,
    _flash_attn_varlen_backward,
    flash_attn_with_kvcache,
)
from modules.profiler import pytorch_profiler as profiler
from itertools import accumulate
import flash_attn_2_cuda as flash_attn_cuda
from flash_attn_2_cuda import (
    FLASH_FWD,
    FLASH_VARLEN_FWD,
    FLASH_BWD,
    FLASH_VARLEN_BWD,
    FLASH_FWD_KVCACHE,
    FLASH_FWD_INFER,
    FLASH_VARLEN_INFER,
)


def set_solution_params(kernel_id, kernel_type, grid_type, num_splits=1):
    is_balance = False
    flash_attn_cuda.ks_set_solution(
        kernel_id, kernel_type, num_splits, grid_type, is_balance
    )


def run_forward_interface(params, verbose=False):
    head_dim_qk = params["head_dim"]
    head_dim_v = params["head_dim_v"]
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
    dtype = torch.float16 if params["is_fp16"] else torch.bfloat16

    softmax_scale = head_dim_qk ** (-0.5)
    window_size = (-1, -1)  # TODO: not support
    alibi_slopes = None  # TODO: not support
    attn_mask = None  # TODO: not support
    return_softmax = False

    q = torch.randn(
        batch_size,
        seqlens_q[0],
        num_heads_q,
        head_dim_qk,
        device="cuda",
        dtype=dtype,
    )
    k = torch.randn(
        batch_size,
        seqlens_kv[0],
        num_heads_kv,
        head_dim_qk,
        device="cuda",
        dtype=dtype,
    )
    v = torch.randn(
        batch_size,
        seqlens_kv[0],
        num_heads_kv,
        head_dim_v,
        device="cuda",
        dtype=dtype,
    )

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


def run_varlen_forward_interface(params, verbose=False):
    head_dim_qk = params["head_dim"]
    head_dim_v = params["head_dim_v"]
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
    dtype = torch.float16 if params["is_fp16"] else torch.bfloat16

    softmax_scale = head_dim_qk ** (-0.5)
    window_size = (-1, -1)  # TODO: not support
    alibi_slopes = None  # TODO: not support
    attn_mask = None  # TODO: not support
    return_softmax = False

    q = torch.randn(
        sum(seqlens_q),
        num_heads_q,
        head_dim_qk,
        device="cuda",
        dtype=dtype,
    )
    k = torch.randn(
        sum(seqlens_kv),
        num_heads_kv,
        head_dim_qk,
        device="cuda",
        dtype=dtype,
    )
    v = torch.randn(
        sum(seqlens_kv),
        num_heads_kv,
        head_dim_v,
        device="cuda",
        dtype=dtype,
    )

    cu_seqlens_q = torch.tensor(
        [0] + list(accumulate(seqlens_q)), dtype=torch.int32, device="cuda"
    )
    cu_seqlens_kv = torch.tensor(
        [0] + list(accumulate(seqlens_kv)), dtype=torch.int32, device="cuda"
    )

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


def run_backward_interface(params, verbose=False):
    head_dim_qk = params["head_dim"]
    head_dim_v = params["head_dim_v"]
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
    dtype = torch.float16 if params["is_fp16"] else torch.bfloat16

    q = torch.randn(
        batch_size,
        seqlens_q[0],
        num_heads_q,
        head_dim_qk,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )
    k = torch.randn(
        batch_size,
        seqlens_kv[0],
        num_heads_kv,
        head_dim_qk,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )
    v = torch.randn(
        batch_size,
        seqlens_kv[0],
        num_heads_kv,
        head_dim_v,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )

    out = torch.empty_like(q)
    dout = torch.empty_like(out)
    softmax_lse = torch.empty(
        (batch_size, num_heads_q, max_seqlen_q), device="cuda", dtype=dtype
    )
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    softmax_scale = head_dim_qk ** (-0.5)
    window_size = (-1, -1)  # TODO: not support
    alibi_slopes = None  # TODO: not support
    attn_mask = None  # TODO: not support
    rng_state = torch.empty((2), dtype=torch.int64)

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


def run_varlen_backward_interface(params, is_varlen=True, verbose=False):
    head_dim_qk = params["head_dim"]
    head_dim_v = params["head_dim_v"]
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
    dtype = torch.float16 if params["is_fp16"] else torch.bfloat16

    q = torch.randn(
        sum(seqlens_q),
        num_heads_q,
        head_dim_qk,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )
    k = torch.randn(
        sum(seqlens_kv),
        num_heads_kv,
        head_dim_qk,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )
    v = torch.randn(
        sum(seqlens_kv),
        num_heads_kv,
        head_dim_v,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )
    cu_seqlens_q = torch.tensor(
        [0] + list(accumulate(seqlens_q)), dtype=torch.int32, device="cuda"
    )
    cu_seqlens_kv = torch.tensor(
        [0] + list(accumulate(seqlens_kv)), dtype=torch.int32, device="cuda"
    )

    out = torch.empty_like(q)
    dout = torch.empty_like(out)
    softmax_lse = torch.empty(
        (batch_size, num_heads_q, max_seqlen_q), device="cuda", dtype=dtype
    )
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    softmax_scale = head_dim_qk ** (-0.5)
    window_size = (-1, -1)  # TODO: not support
    alibi_slopes = None  # TODO: not support
    attn_mask = None  # TODO: not support
    rng_state = torch.empty((2), dtype=torch.int64)

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


# TODO: support paged kv cache and rope
def run_kvcache_interface(params, verbose=False):
    head_dim_qk = params["head_dim"]
    head_dim_v = params["head_dim_v"]
    num_heads_q = params["num_heads_q"]
    num_heads_kv = params["num_heads_kv"]
    batch_size = params["batch_size"]
    seqlens_q = params["seqlens_q"]
    seqlens_kv = params["seqlens_kv"]
    assert len(seqlens_q) == batch_size
    assert len(seqlens_kv) == batch_size
    causal = params["causal"]
    dtype = torch.float16 if params["is_fp16"] else torch.bfloat16

    q = torch.randn(
        batch_size,
        seqlens_q[0],
        num_heads_q,
        head_dim_qk,
        device="cuda",
        dtype=dtype,
    )
    k_cache = torch.randn(
        batch_size,
        seqlens_kv[0],
        num_heads_kv,
        head_dim_qk,
        device="cuda",
        dtype=dtype,
    )
    v_cache = torch.randn(
        batch_size,
        seqlens_kv[0],
        num_heads_kv,
        head_dim_v,
        device="cuda",
        dtype=dtype,
    )

    softmax_scale = head_dim_qk ** (-0.5)
    window_size = (-1, -1)  # TODO: not support
    alibi_slopes = None  # TODO: not support
    return_softmax = False

    return profiler(
        flash_attn_with_kvcache,
        q,
        k_cache,
        v_cache,
        k=None,
        v=None,
        rotary_cos=None,
        rotary_sin=None,
        cache_seqlens=None,
        cache_batch_idx=None,
        cache_leftpad=None,
        block_table=None,
        causal=causal,
        window_size=window_size,
        rotary_interleaved=None,
        alibi_slopes=alibi_slopes,
        num_splits=None,
        verbose=verbose,
    )


# TODO: support qkv packed
def dispatch_flash_attention_api(params, verbose=False):
    api = params["api"]

    if api == FLASH_FWD or api == FLASH_FWD_INFER:
        return run_forward_interface(params, verbose=verbose)
    elif api == FLASH_VARLEN_FWD or api == FLASH_VARLEN_INFER:
        return run_varlen_forward_interface(params, verbose=verbose)
    elif api == FLASH_BWD:
        return run_backward_interface(params, verbose=verbose)
    elif api == FLASH_VARLEN_BWD:
        return run_varlen_backward_interface(params, verbose=verbose)
    elif api == FLASH_FWD_KVCACHE:
        return run_kvcache_interface(params, verbose=verbose)
    else:
        raise ValueError(f"Unsupported API: {api}")
