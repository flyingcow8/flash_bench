from modules.dispatch_attention_api import (
    dispatch_flash_attention_api,
    set_solution_params,
)
from modules.kernel_registry import (
    forward_kernels_fp16,
    forward_kernels_bf16,
    forward_splitkv_kernels_fp16,
    forward_splitkv_kernels_bf16,
    backward_kernels_fp16,
    backward_kernels_bf16,
)
from modules.profiler import (
    find_flash_fwd_cuda_time,
    find_flash_bwd_cuda_time,
    find_flash_fwd_splitkv_cuda_time,
)
from modules.splits_heuristic import num_splits_heuristic
from FlashBenchData.GridType import GridType
from FlashBenchData.BalanceType import BalanceType
from FlashBenchData.KernelType import KernelType
from modules.logger import logger

from flash_attn_2_cuda import (
    FLASH_FWD,
    FLASH_VARLEN_FWD,
    FLASH_BWD,
    FLASH_VARLEN_BWD,
    FLASH_FWD_KVCACHE,
    FLASH_FWD_INFER,
    FLASH_VARLEN_INFER,
)


# for one mha problem
def bench_process(config, params):
    api = params.get("api")

    # Get the head dimension from params and round up to the nearest multiple of 32
    head_dim = params.get("head_dim")
    rounded_head_dim = ((head_dim + 31) // 32) * 32
    is_fp16 = params.get("is_fp16")

    if (
        api == FLASH_FWD
        or api == FLASH_VARLEN_FWD
        or api == FLASH_FWD_INFER
        or api == FLASH_VARLEN_INFER
        or api == FLASH_FWD_KVCACHE
    ):
        op_type = "fwd"
    elif api == FLASH_BWD or api == FLASH_VARLEN_BWD:
        op_type = "bwd"

    # Read forward kernels for the rounded head dimension
    if is_fp16:
        if op_type == "fwd":
            fwd_kernels = forward_kernels_fp16.get(f"hdim{rounded_head_dim}", [])
            fwd_splitkv_kernels = forward_splitkv_kernels_fp16.get(
                f"hdim{rounded_head_dim}", []
            )
        else:
            bwd_kernels = backward_kernels_fp16.get(f"hdim{rounded_head_dim}", [])
    else:
        if op_type == "fwd":
            fwd_kernels = forward_kernels_bf16.get(f"hdim{rounded_head_dim}", [])
            fwd_splitkv_kernels = forward_splitkv_kernels_bf16.get(
                f"hdim{rounded_head_dim}", []
            )
        else:
            bwd_kernels = backward_kernels_bf16.get(f"hdim{rounded_head_dim}", [])

    logger.info("Current parameters:")
    for key, value in params.items():
        logger.info(f" - {key}: {value}")
    logger.info("=" * 50)

    if op_type == "fwd":
        logger.debug(f"Forward kernels for rounded head_dim {rounded_head_dim}:")
        for kernel in fwd_kernels:
            logger.debug(f" - {kernel['kernel_id']}")
        logger.debug(
            f"Forward-splitkv kernels for rounded head_dim {rounded_head_dim}:"
        )
        for kernel in fwd_splitkv_kernels:
            logger.debug(f" - {kernel['kernel_id']}")

    if op_type == "bwd":
        logger.debug(f"Backward kernels for rounded head_dim {rounded_head_dim}:")
        for kernel in bwd_kernels:
            logger.debug(f" - {kernel['kernel_id']}")
    logger.debug("=" * 50)

    # use flash attention's policy to determine whether to force use split kernels
    force_split_kernels = (
        api == FLASH_FWD_KVCACHE
        or api == FLASH_VARLEN_FWD
        or api == FLASH_VARLEN_INFER
    ) and params.get("paged_kv", True)
    logger.debug(f"Force split kernels: {force_split_kernels}")
    logger.debug("=" * 50)

    profile_verbose = config["profile_verbose"]
    platform = config["platform"]
    grid_types = [
        GridType().NHB,
        GridType().NBH,
        GridType().HNB,
        GridType().BNH,
        GridType().BHN,
        GridType().HBN,
    ]
    balance_types = [BalanceType().None_]

    def process_fwd_kernel(kernel, grid_type, balance_type, params):
        logger.debug(f"Processing fwd kernel: {kernel['kernel_id']}, {grid_type}, {balance_type}")
        set_solution_params(kernel["kernel_id"], KernelType().Fwd, grid_type)
        prof = dispatch_flash_attention_api(params, verbose=profile_verbose)
        cuda_time = find_flash_fwd_cuda_time(prof)
        if cuda_time is None:
            logger.warning(f"[Fwd] CUDA time is None!")
            return None

        return {
            "head_dim": rounded_head_dim,
            "kernel_id": kernel["kernel_id"],
            "grid_type": grid_type,
            "balance_type": balance_type,
            "num_splits": 1,
            "kernel_type": KernelType().Fwd,
            "time_us": cuda_time,
        }

    def process_bwd_kernel(kernel, grid_type, balance_type, params):
        logger.debug(f"Processing bwd kernel: {kernel['kernel_id']}, {grid_type}, {balance_type}")
        set_solution_params(kernel["kernel_id"], KernelType().Bwd, grid_type)
        prof = dispatch_flash_attention_api(params, verbose=profile_verbose)
        cuda_time = find_flash_bwd_cuda_time(prof)
        if cuda_time is None:
            logger.warning(f"[Bwd] CUDA time is None!")
            return None

        return {
            "head_dim": rounded_head_dim,
            "kernel_id": kernel["kernel_id"],
            "grid_type": grid_type,
            "balance_type": balance_type,
            "num_splits": 1,
            "kernel_type": KernelType().Bwd,
            "time_us": cuda_time,
        }

    def process_fwd_splitkv_kernel(kernel, grid_type, params):
        batch_size = params.get("batch_size")
        num_heads = params.get("num_heads_q")
        max_seqlen_q = max(params.get("seqlens_q"))
        max_seqlen_kv = max(params.get("seqlens_kv"))
        block_m = kernel["block_m"]
        block_n = kernel["block_n"]
        num_m_blocks = (max_seqlen_q + block_m - 1) // block_m
        num_n_blocks = (max_seqlen_kv + block_n - 1) // block_n
        assert (
            platform == "MXC500" or platform == "MXC550"
        ), f"Unsupported platform: {platform}. Only MXC500 or MXC550 is supported."
        num_SMs = 104
        num_splits = num_splits_heuristic(
            batch_size * num_heads * num_m_blocks, num_SMs * 2, num_n_blocks
        )
        results = []
        kernel_id = kernel["kernel_id"]
        for ns in num_splits:
            logger.debug(f"Processing fwd-splitkv kernel: {kernel_id}, {grid_type}, {ns}")
            if (ns == 1 and kernel["is_splits"]) or (
                ns > 1 and not kernel["is_splits"]
            ):
                continue

            set_solution_params(
                kernel["kernel_id"], KernelType().FwdSplitkv, grid_type, ns
            )
            prof = dispatch_flash_attention_api(params, verbose=profile_verbose)
            cuda_time = find_flash_fwd_splitkv_cuda_time(prof)
            if cuda_time is None:
                logger.warning(f"[FwdSplitkv] CUDA time is None!")
            else:
                results.append(
                    {
                        "head_dim": rounded_head_dim,
                        "kernel_id": kernel_id,
                        "grid_type": grid_type,
                        "num_splits": ns,
                        "balance_type": 0,
                        "kernel_type": KernelType().FwdSplitkv,
                        "time_us": cuda_time,
                    }
                )
        return results

    results = []
    if op_type == "fwd":
        if not force_split_kernels:
            results = [
                result
                for result in (
                    process_fwd_kernel(kernel, grid_type, balance_type, params)
                    for kernel in fwd_kernels
                    for grid_type in grid_types
                    for balance_type in balance_types
                )
                if result is not None
            ]

        fwd_splitkv_results = [
            process_fwd_splitkv_kernel(kernel, grid_type, params)
            for kernel in fwd_splitkv_kernels
            for grid_type in grid_types
        ]
        for result in fwd_splitkv_results:
            results.extend(result)
    else:
        results = [
            result
            for result in (
                process_bwd_kernel(kernel, grid_type, balance_type, params)
                for kernel in bwd_kernels
                for grid_type in grid_types
                for balance_type in balance_types
            )
            if result is not None
        ]

    return results
