import yaml
import os
import datetime
import flatbuffers
from FlashBenchData.BalanceType import BalanceType
from FlashBenchData.GridType import GridType
from modules.bench_process import bench_process, FLASH_MHA_APIS
from modules.export_fb import (
    create_attention_problem,
    create_bench_table_binary,
)
from modules.export_yaml import append_results_to_yaml
from FlashBenchData import AttentionBenchTable
from modules.kernel_registry import register_kernels


def load_config(config_file):
    with open(config_file, "r") as file:
        return yaml.safe_load(file)


def validate_config(config):
    keys = ["input_path", "output_dir", "kernels_file"]
    for key in keys:
        if key not in config:
            print(f"Error: '{key}' not found in config.yaml")
            return False

    grid_types = config.get("grid_types")
    if not grid_types or len(grid_types) == 0:
        config["grid_types"] = [
            getattr(GridType, attr)
            for attr in dir(GridType)
            if not callable(getattr(GridType, attr)) and not attr.startswith("__")
        ]

    balance_types = config.get("balance_types")
    if not balance_types or len(balance_types) == 0:
        config["balance_types"] = [
            getattr(BalanceType, attr)
            for attr in dir(BalanceType)
            if not callable(getattr(BalanceType, attr)) and not attr.startswith("__")
        ]

    return True


def find_best_solution(solutions):
    if not solutions:
        return {}  # Return an empty dict if results is empty

    return min(solutions, key=lambda x: x.get("time_us", float("inf")))


def process_input_file(config, input_file, output_dir, builder):
    print(f"Processing file: {input_file}")

    with open(input_file, "r") as file:
        params_file = yaml.safe_load(file)

    if "mha_api" not in params_file or "mha_params" not in params_file:
        print(f"Error: 'mha_api' or 'mha_params' not found in {input_file}")
        return []

    # Check if the mha_api is in the allowed list
    mha_api = params_file["mha_api"]
    if mha_api not in FLASH_MHA_APIS:
        print(f"Error: Invalid 'mha_api' value '{mha_api}' in {input_file}")
        print(f"Allowed values are: {', '.join(FLASH_MHA_APIS)}")
        return []

    output_yaml = os.path.join(
        output_dir, f"{os.path.basename(input_file)}_results.yaml"
    )
    attention_problems = []

    print(f"Testing Api: {mha_api}")
    for idx, params in enumerate(params_file["mha_params"]):
        print(f"Running case {idx}:")
        fwd_results, bwd_results = bench_process(config, mha_api, params)
        best_fwd_solution = find_best_solution(fwd_results)
        best_bwd_solution = (
            find_best_solution(bwd_results) if params.get("is_training", True) else {}
        )

        append_results_to_yaml(
            params,
            fwd_results,
            best_fwd_solution,
            bwd_results,
            best_bwd_solution,
            output_yaml,
        )
        problem = create_attention_problem(
            builder, params, best_fwd_solution, best_bwd_solution
        )
        attention_problems.append(problem)

    print(f"Finished processing {input_file}\n")
    print("=" * 50 + "\n")
    return attention_problems


def main():
    config = load_config("config.yaml")
    ret = validate_config(config)
    if not ret:
        print("Error: Invalid config.yaml")
        return

    input_files = config.get("input_path")
    output_dir = config.get("output_dir")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Project Name: {config['project']['name']}")
    print(f"Project Version: {config['project']['version']}")
    print(f"Platform: {config['platform']}")
    print(f"Input files to process: {', '.join(input_files)}")
    print(
        f"Grid types: {', '.join(str(grid_type) for grid_type in config['grid_types'])}"
    )
    print(
        f"Balance types: {', '.join(str(balance_type) for balance_type in config['balance_types'])}"
    )
    print("=" * 50 + "\n")

    # Register kernels from the YAML file
    kernels_file = config.get("kernels_file")
    if kernels_file and os.path.exists(kernels_file):
        print(f"Registering kernels from: {kernels_file}")
        register_kernels(kernels_file)
    else:
        print("Warning: kernel_traits file not found or not specified in config.")
    print("=" * 50 + "\n")

    builder = flatbuffers.Builder(1024)
    all_attention_problems = []

    for input_file in input_files:
        if not os.path.exists(input_file):
            print(f"Warning: File {input_file} not found. Skipping.")
            continue
        all_attention_problems.extend(
            process_input_file(config, input_file, output_dir, builder)
        )

    current_date = datetime.datetime.now().strftime("%Y%m%d")
    bench_file = f"{output_dir}/bench_table_{current_date}.bin"
    create_bench_table_binary(builder, all_attention_problems, config, bench_file)
    print(f"Benchmark table written to: {bench_file}")


if __name__ == "__main__":
    main()
