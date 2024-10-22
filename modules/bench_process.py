from modules.dispatch_attention_api import dispatch_flash_attention_api
from modules.kernel_registry import (
    forward_kernels,
    forward_splitkv_kernels,
    backward_kernels,
)
from modules.testing_solution import create_solution_test_binary
from modules.config_constants import FLASH_MHA_APIS
from modules.profiler import (
    find_flash_fwd_cuda_time,
    find_flash_bwd_cuda_time,
    find_flash_fwd_splitkv_cuda_time,
)
from modules.splits_heuristic import num_splits_heuristic
from FlashBenchData.GridType import GridType
from FlashBenchData.BalanceType import BalanceType
from FlashBenchData.KernelType import KernelType


def get_all_grid_types():
    return [
        getattr(GridType, attr)
        for attr in dir(GridType)
        if not callable(getattr(GridType, attr)) and not attr.startswith("__")
    ]


def get_all_balance_types():
    return [
        getattr(BalanceType, attr)
        for attr in dir(BalanceType)
        if not callable(getattr(BalanceType, attr)) and not attr.startswith("__")
    ]


# for one mha problem
def bench_process(config, api, params):
    # Get the head dimension from params and round up to the nearest multiple of 32
    head_dim = params.get("head_dim")
    rounded_head_dim = ((head_dim + 31) // 32) * 32

    # Read forward kernels for the rounded head dimension
    fwd_kernels = forward_kernels.get(f"hdim{rounded_head_dim}", [])
    fwd_splitkv_kernels = forward_splitkv_kernels.get(f"hdim{rounded_head_dim}", [])

    if params.get("is_training", True):
        # Read backward kernels for the rounded head dimension
        bwd_kernels = backward_kernels.get(f"hdim{rounded_head_dim}", [])

    print(f"Original head_dim: {head_dim}")
    print(f"Rounded head_dim: {rounded_head_dim}")
    print(f"Forward kernels for rounded head_dim {rounded_head_dim}: {fwd_kernels}")
    print(
        f"Forward-splitkv kernels for rounded head_dim {rounded_head_dim}: {fwd_splitkv_kernels}"
    )
    print(f"Backward kernels for rounded head_dim {rounded_head_dim}: {bwd_kernels}")

    # Read grid_types from config, use get_all_grid_types() if not specified
    config_grid_types = config.get("grid_types")
    if config_grid_types:
        grid_types = [
            getattr(GridType, grid_type)
            for grid_type in config_grid_types
            if hasattr(GridType, grid_type)
        ]
        if not grid_types:
            print("Warning: No valid grid types found in config. Using all grid types.")
            grid_types = get_all_grid_types()
    else:
        grid_types = get_all_grid_types()

    # Read balance_types from config, use get_all_balance_types() if not specified
    config_balance_types = config.get("balance_types")
    if config_balance_types:
        balance_types = [
            getattr(BalanceType, balance_type)
            for balance_type in config_balance_types
            if hasattr(BalanceType, balance_type)
        ]
        if not balance_types:
            print(
                "Warning: No valid balance types found in config. Using all balance types."
            )
            balance_types = get_all_balance_types()
    else:
        balance_types = get_all_balance_types()

    print("Grid types for testing:")
    for grid_type in grid_types:
        print(f"  - {grid_type}")

    print("\nBalance types for testing:")
    for balance_type in balance_types:
        print(f"  - {balance_type}")

    print("=" * 50)

    profile_verbose = config.get("profile_verbose", False)
    output_dir = config.get("output_dir")

    fwd_results = []
    bwd_results = []

    print("Starting benchmark process...")
    print(f"Number of forward kernels: {len(fwd_kernels)}")
    print(f"Number of forward splitkv kernels: {len(fwd_splitkv_kernels)}")
    if params.get("is_training", True):
        print(f"Number of backward kernels: {len(bwd_kernels)}")
    print(f"Number of grid types: {len(grid_types)}")
    print(f"Number of balance types: {len(balance_types)}")
    print("=" * 50)

    flash_attn_varlen_apis = [
        api for api in FLASH_MHA_APIS if api.startswith("flash_attn_varlen")
    ]

    # use flash attention's policy to determine whether to force use split kernels
    force_split_kernels = (
        api in flash_attn_varlen_apis and params.get("paged_kv", True)
    ) or (api == "flash_attn_with_kvcache" and params.get("paged_kv", True))

    def process_fwd_kernel(kernel, grid_type, balance_type, params, config, output_dir):
        create_solution_test_binary(
            rounded_head_dim,
            kernel,
            KernelType().Fwd,
            grid_type,
            balance_type=balance_type,
            config=config,
            output_dir=output_dir,
        )

        prof = dispatch_flash_attention_api(
            api, params, KernelType().Fwd, verbose=profile_verbose
        )

        cuda_time = find_flash_fwd_cuda_time(prof)

        return {
            "head_dim": rounded_head_dim,
            "kernel_id": kernel["kernel_id"],
            "grid_type": grid_type,
            "balance_type": balance_type,
            "kernel_type": KernelType().Fwd,
            "time_us": cuda_time,
        }

    def process_bwd_kernel(kernel, grid_type, balance_type, params, config, output_dir):
        create_solution_test_binary(
            rounded_head_dim,
            kernel,
            KernelType().Bwd,
            grid_type,
            balance_type,
            config=config,
            output_dir=output_dir,
        )

        prof = dispatch_flash_attention_api(
            api, params, KernelType().Bwd, verbose=profile_verbose
        )

        cuda_time = find_flash_bwd_cuda_time(prof)

        return {
            "head_dim": rounded_head_dim,
            "kernel_id": kernel["kernel_id"],
            "grid_type": grid_type,
            "balance_type": balance_type,
            "kernel_type": KernelType().Bwd,
            "time_us": cuda_time,
        }

    def process_fwd_splitkv_kernel(kernel, grid_type, params, config, output_dir):
        batch_size = params.get("batch_size")
        num_heads = params.get("num_heads")
        max_seqlen_q = max(params.get("seqlens_q"))
        max_seqlen_kv = max(params.get("seqlens_kv"))
        block_m = kernel["block_m"]
        block_n = kernel["block_n"]
        num_m_blocks = (max_seqlen_q + block_m - 1) // block_m
        num_n_blocks = (max_seqlen_kv + block_n - 1) // block_n
        platform = config.get("platform")
        num_SMs = {
            "MXC500": 104,
        }.get(platform)
        if num_SMs is None:
            raise ValueError(f"Unsupported platform: {platform}")
        num_splits = num_splits_heuristic(
            batch_size * num_heads * num_m_blocks, num_SMs * 2, num_n_blocks
        )
        results = []
        for ns in num_splits:
            create_solution_test_binary(
                rounded_head_dim,
                kernel,
                KernelType().FwdSplitkv,
                grid_type,
                num_splits=ns,
                config=config,
                output_dir=output_dir,
            )

            prof = dispatch_flash_attention_api(
                api, params, KernelType().FwdSplitkv, verbose=profile_verbose
            )

            cuda_time = find_flash_fwd_splitkv_cuda_time(prof)

            results.append({
                "head_dim": rounded_head_dim,
                "kernel_id": kernel["kernel_id"],
                "grid_type": grid_type,
                "num_splits": num_splits,
                "kernel_type": KernelType().FwdSplitkv,
                "time_us": cuda_time,
            })
        return results


    if not force_split_kernels:
        fwd_results = [
            process_fwd_kernel(
                kernel, grid_type, balance_type, params, config, output_dir
            )
            for kernel in fwd_kernels
            for grid_type in grid_types
            for balance_type in balance_types
        ]

    fwd_splitkv_results = [
        process_fwd_splitkv_kernel(kernel, grid_type, params, config, output_dir)
        for kernel in fwd_splitkv_kernels
        for grid_type in grid_types
    ]

    if params.get("is_training", True):
        bwd_results = [
            process_bwd_kernel(
                kernel, grid_type, balance_type, params, config, output_dir
            )
            for kernel in bwd_kernels
            for grid_type in grid_types
            for balance_type in balance_types
        ]

    print("End of benchmark process")

    return fwd_results + fwd_splitkv_results, bwd_results
