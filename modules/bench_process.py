from modules.dispatch_attention_api import dispatch_flash_attention_api
from modules.kernel_registry import forward_kernels, forward_kernels_splitkv, backward_kernels
from modules.testing_solution import create_test_solution
from FlashBenchData.GridType import GridType
from FlashBenchData.BalanceType import BalanceType
from FlashBenchData.OpType import OpType
from modules.profiler import find_flash_fwd_cuda_time, find_flash_bwd_cuda_time

def create_solution_dict(
    head_dim, tile_m, tile_n, num_waves, grid_type, balance_type, op_type, time_us
):
    return {
        "head_dim": head_dim,
        "tile_m": tile_m,
        "tile_n": tile_n,
        "num_waves": num_waves,
        "grid_type": grid_type,
        "balance_type": balance_type,
        "op_type": op_type,
        "time_us": time_us,
    }


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


def bench_process(config, params):
    # Get the head dimension from params and round up to the nearest multiple of 32
    head_dim = params.get("head_dim")
    rounded_head_dim = (
        (head_dim + 31) // 32
    ) * 32  # TODO: add testing for rounded_head_dim + 32

    # Read forward kernels for the rounded head dimension
    fwd_kernels = forward_kernels.get(rounded_head_dim, [])
    fwd_kernels_splitkv = forward_kernels_splitkv.get(rounded_head_dim, [])

    if params.get("is_training", True):
        # Read backward kernels for the rounded head dimension
        bwd_kernels = backward_kernels.get(rounded_head_dim, [])

    print(f"Original head_dim: {head_dim}")
    print(f"Rounded head_dim: {rounded_head_dim}")
    print(f"Forward kernels for rounded head_dim {rounded_head_dim}: {fwd_kernels}")
    print(f"Forward-splitkv kernels for rounded head_dim {rounded_head_dim}: {fwd_kernels_splitkv}")
    print(f"Backward kernels for rounded head_dim {rounded_head_dim}: {bwd_kernels}")

    # Read GridType from FlashBenchData and iterate through all enum values
    # grid_types = get_all_grid_types()
    grid_types = [GridType.NHB, GridType.NBH, GridType.HNB]

    # Read BalanceType from FlashBenchData and iterate through all enum values
    # balance_types = get_all_balance_types()
    balance_types = [BalanceType.None_]

    profile_verbose = config.get("profile_verbose", False)
    output_dir = config.get("output_dir")
    enable_splitkv = config.get("enable_splitkv", False)

    fwd_results = []
    bwd_results = []

    print("Starting benchmark process...")
    print(f"Number of forward kernels: {len(fwd_kernels)}")
    if params.get("is_training", True):
        print(f"Number of backward kernels: {len(bwd_kernels)}")
    print(f"Number of grid types: {len(grid_types)}")
    print(f"Number of balance types: {len(balance_types)}")
    print("=" * 50)

    # iterate through all grid types and balance types
    for grid_type in grid_types:
        for balance_type in balance_types:
            for kernel in fwd_kernels:
                print(f"Fwd kernel: {kernel}, Grid type: {grid_type}, Balance type: {balance_type}")
                tile_m, tile_n, num_waves = kernel
                create_test_solution(
                    rounded_head_dim,
                    tile_m,
                    tile_n,
                    num_waves,
                    grid_type,
                    balance_type,
                    OpType().Fwd,
                    config,
                    output_dir,
                )
                prof = dispatch_flash_attention_api(
                    params, "fwd", verbose=profile_verbose
                )
                flash_fwd_cuda_time = find_flash_fwd_cuda_time(prof)
                fwd_results.append(
                    create_solution_dict(
                        rounded_head_dim,
                        tile_m,
                        tile_n,
                        num_waves,
                        grid_type,
                        balance_type,
                        OpType().Fwd,
                        flash_fwd_cuda_time,
                    )
                )


            if enable_splitkv:
                for kernel in fwd_kernels_splitkv:
                    print(f"Fwd-splitkv kernel: {kernel}, Grid type: {grid_type}, Balance type: {balance_type}")
                    tile_m, tile_n, num_waves = kernel
                    create_test_solution(
                        rounded_head_dim,
                        tile_m,
                        tile_n,
                        num_waves,
                        grid_type,

            if params.get("is_training", True):
                for kernel in bwd_kernels:
                    print(f"Bwd kernel: {kernel}, Grid type: {grid_type}, Balance type: {balance_type}")
                    tile_m, tile_n, num_waves = kernel
                    create_test_solution(
                        rounded_head_dim,
                        tile_m,
                        tile_n,
                        num_waves,
                        grid_type,
                        balance_type,
                        OpType().Bwd,
                        config,
                        output_dir,
                    )
                    prof = dispatch_flash_attention_api(
                        params, "bwd", verbose=profile_verbose
                    )
                    flash_bwd_cuda_time = find_flash_bwd_cuda_time(prof)
                    bwd_results.append(
                        create_solution_dict(
                            rounded_head_dim,
                            tile_m,
                            tile_n,
                            num_waves,
                            grid_type,
                            balance_type,
                            OpType().Bwd,
                            flash_bwd_cuda_time,
                        )
                    )

    print("End of benchmark process")

    return fwd_results, bwd_results
