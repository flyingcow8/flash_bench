from modules.dispatch_attention_api import dispatch_flash_attention_api
from modules.kernel_registry import (
    forward_kernels_fp16,
    forward_kernels_bf16,
    forward_splitkv_kernels_fp16,
    forward_splitkv_kernels_bf16,
    backward_kernels_fp16,
    backward_kernels_bf16,
)
from modules.export_fb import create_solution_test_binary
from modules.profiler import (
    find_flash_fwd_cuda_time,
    find_flash_bwd_cuda_time,
    find_flash_fwd_splitkv_cuda_time,
)
from modules.splits_heuristic import num_splits_heuristic
from FlashBenchData.GridType import GridType
from FlashBenchData.BalanceType import BalanceType
from FlashBenchData.KernelType import KernelType


FLASH_MHA_APIS = [
    "flash_attn_func",
    "flash_attn_qkvpacked_func",
    "flash_attn_kvpacked_func",
    "flash_attn_varlen_func",
    "flash_attn_varlen_qkvpacked_func",
    "flash_attn_varlen_kvpacked_func",
    "flash_attn_with_kvcache"
]

# for one mha problem
def bench_process(config, api, params):
    # Get the head dimension from params and round up to the nearest multiple of 32
    head_dim = params.get("head_dim_qk")
    rounded_head_dim = ((head_dim + 31) // 32) * 32

    # Read forward kernels for the rounded head dimension
    if params.get("dtype") == "float16":
        fwd_kernels = forward_kernels_fp16.get(f"hdim{rounded_head_dim}", [])
    elif params.get("dtype") == "bfloat16":
        fwd_kernels = forward_kernels_bf16.get(f"hdim{rounded_head_dim}", [])

    if params.get("dtype") == "float16":
        fwd_splitkv_kernels = forward_splitkv_kernels_fp16.get(f"hdim{rounded_head_dim}", [])
    elif params.get("dtype") == "bfloat16":
        fwd_splitkv_kernels = forward_splitkv_kernels_bf16.get(f"hdim{rounded_head_dim}", [])

    if params.get("is_training", True):
        # Read backward kernels for the rounded head dimension
        if params.get("dtype") == "float16":
            bwd_kernels = backward_kernels_fp16.get(f"hdim{rounded_head_dim}", [])
        elif params.get("dtype") == "bfloat16":
            bwd_kernels = backward_kernels_bf16.get(f"hdim{rounded_head_dim}", [])

    print("\nCurrent parameters:")
    for key, value in params.items():
        print(f"{key}: {value}")
    print("=" * 50)

    print(f"Forward kernels for rounded head_dim {rounded_head_dim}:")
    for kernel in fwd_kernels:
        print(f"  {kernel['kernel_id']}")
    
    print(f"\nForward-splitkv kernels for rounded head_dim {rounded_head_dim}:")
    for kernel in fwd_splitkv_kernels:
        print(f"  {kernel['kernel_id']}")
    
    print(f"\nBackward kernels for rounded head_dim {rounded_head_dim}:")
    for kernel in bwd_kernels:
        print(f"  {kernel['kernel_id']}")
    print("=" * 50)

    fwd_results = []
    bwd_results = []

    print("Starting benchmark process...")

    flash_attn_varlen_apis = [
        api for api in FLASH_MHA_APIS if api.startswith("flash_attn_varlen")
    ]

    # use flash attention's policy to determine whether to force use split kernels
    force_split_kernels = (
        api in flash_attn_varlen_apis and params.get("paged_kv", True)
    ) or (api == "flash_attn_with_kvcache" and params.get("paged_kv", True))
    print(f"Force split kernels: {force_split_kernels}")
    print("=" * 50)

    output_dir = config["output_dir"]
    version = config["project"]["version"]
    profile_verbose = config["profile_verbose"]
    platform = config["platform"]
    grid_types = config["grid_types"]
    balance_types = config["balance_types"]

    def process_fwd_kernel(kernel, grid_type, balance_type, params):
        create_solution_test_binary(
            rounded_head_dim,
            grid_type,
            KernelType().Fwd,
            kernel["kernel_id"],
            version,
            balance_type=balance_type,
            output_dir=output_dir,
        )

        prof = dispatch_flash_attention_api(
            api, params, KernelType().Fwd, verbose=profile_verbose
        )

        cuda_time = find_flash_fwd_cuda_time(prof)
        if cuda_time is None:
            print(f"[Warning][Fwd] CUDA time is None!")
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
        create_solution_test_binary(
            rounded_head_dim,
            grid_type,
            KernelType().Bwd,
            kernel["kernel_id"],
            version,
            balance_type=balance_type,
            output_dir=output_dir,
        )

        prof = dispatch_flash_attention_api(
            api, params, KernelType().Bwd, verbose=profile_verbose
        )

        cuda_time = find_flash_bwd_cuda_time(prof)
        if cuda_time is None:
            print(f"[Warning][Bwd] CUDA time is None!")
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
            platform == "MXC500"
        ), f"Unsupported platform: {platform}. Only MXC500 is supported."
        num_SMs = 104
        num_splits = num_splits_heuristic(
            batch_size * num_heads * num_m_blocks, num_SMs * 2, num_n_blocks
        )
        results = []
        kernel_id = kernel["kernel_id"]
        for ns in num_splits:
            if (ns == 1 and kernel["is_splits"]) or (ns > 1 and not kernel["is_splits"]):
                continue
            create_solution_test_binary(
                rounded_head_dim,
                grid_type,
                KernelType().FwdSplitkv,
                kernel_id,
                version,
                num_splits=ns,
                output_dir=output_dir,
            )

            prof = dispatch_flash_attention_api(
                api, params, KernelType().FwdSplitkv, verbose=profile_verbose
            )

            cuda_time = find_flash_fwd_splitkv_cuda_time(prof)
            if cuda_time is None:
                print(f"[Warning][FwdSplitkv] CUDA time is None!")
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

    if not force_split_kernels:
        fwd_results = [
            result for result in (
                process_fwd_kernel(kernel, grid_type, balance_type, params)
                for kernel in fwd_kernels
                for grid_type in grid_types
                for balance_type in balance_types
            ) if result is not None
        ]

    fwd_splitkv_results = [
        process_fwd_splitkv_kernel(kernel, grid_type, params)
        for kernel in fwd_splitkv_kernels
        for grid_type in grid_types
    ]
    for result in fwd_splitkv_results:
        fwd_results.extend(result)

    if params.get("is_training", True):
        bwd_results = [
            result for result in (
                process_bwd_kernel(kernel, grid_type, balance_type, params)
                for kernel in bwd_kernels
                for grid_type in grid_types
                for balance_type in balance_types
            ) if result is not None
        ]

    print("End of benchmark process")

    return fwd_results, bwd_results
