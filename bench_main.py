import yaml
import os
import datetime
import flatbuffers
from FlashBenchData.BalanceType import BalanceType
from FlashBenchData.GridType import GridType
from modules.bench_process import bench_process
from modules.export_fb import (
    create_attention_problem,
    create_bench_table_binary,
)
from modules.export_yaml import append_results_to_yaml
from modules.kernel_registry import register_kernels
from tqdm import tqdm
import sys
from modules.logger import logger
from flash_attn_2_cuda import FlashAPI


# Define custom constructor
def flash_api_constructor(loader, node):
    val = int(node.value[0][1].value)
    return FlashAPI(val)


# Register the constructor
yaml.SafeLoader.add_constructor(
    "tag:yaml.org,2002:python/object/new:flash_attn_2_cuda.FlashAPI",
    flash_api_constructor,
)


def load_config(config_file):
    with open(config_file, "r") as f:
        return yaml.safe_load(f)


def validate_config(config):
    keys = ["input_path", "output_dir", "kernels_file"]
    for key in keys:
        if key not in config:
            logger.error(f"Error: '{key}' not found in config.yaml")
            return False

    return True


def find_best_solution(solutions):
    if not solutions:
        return {}  # Return an empty dict if results is empty

    return min(solutions, key=lambda x: x.get("time_us", float("inf")))


def process_input_file(config, input_file, output_file, builder):
    logger.info(f"Processing file: {input_file}")
    with open(input_file, "r") as f:
        params_file = yaml.safe_load(f)

    attention_problems = []

    logger.info(f"Starting benchmark for input_file: {input_file}")
    with tqdm(
        enumerate(params_file),
        desc="Processing parameters",
        total=len(params_file),
    ) as pbar:
        for idx, params in pbar:
            pbar.set_postfix({"Case": idx})
            results = bench_process(config, params.get("prob"))
            if not results:
                logger.warning(f"No valid solution found for case {idx}. Skipping.")
                continue
            best_solution = find_best_solution(results)

            append_results_to_yaml(
                params.get("prob"),
                results,
                best_solution,
                output_file,
            )
            problem = create_attention_problem(
                builder, params.get("prob"), best_solution
            )
            attention_problems.append(problem)

    logger.info(f"Finished processing {input_file}\n")
    logger.info("=" * 50 + "\n")
    return attention_problems


def main():
    config = load_config("config.yaml")
    ret = validate_config(config)
    if not ret:
        logger.error("Error: Invalid config.yaml")
        return

    input_files = config.get("input_path")
    output_dir = config.get("output_dir")
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Project Name: {config['project']['name']}")
    logger.info(f"Project Version: {config['project']['version']}")
    logger.info(f"Platform: {config['platform']}")
    logger.info(f"Input files to process: {', '.join(input_files)}")
    logger.info("=" * 50 + "\n")

    # set env variables
    os.environ["KINETO_LOG_LEVEL"] = "5"  # disable kineto logging
    os.environ["MHA_LOG_ENABLE"] = str(config["logging"]["flash_attn_log_enable"])
    os.environ["MHA_LOG_LEVEL"] = str(config["logging"]["flash_attn_log_level"])
    if config["logging"]["flash_attn_log_file"]:
        os.environ["MHA_LOG_OUTPUT"] = "file"

    # Register kernels from the YAML file
    kernels_file = config.get("kernels_file")
    if kernels_file and os.path.exists(kernels_file):
        logger.info(f"Registering kernels from: {kernels_file}")
        register_kernels(kernels_file)
    else:
        logger.error("Error: kernel_traits file not found or not specified in config.")
        sys.exit(1)
    logger.info("=" * 50 + "\n")

    builder = flatbuffers.Builder(1024)
    all_attention_problems = []

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    for input_file in input_files:
        if not os.path.exists(input_file):
            logger.warning(f"File {input_file} not found. Skipping.")
            continue

        output_file = os.path.join(
            output_dir,
            f"{os.path.splitext(os.path.basename(input_file))[0]}_results_{timestamp}.yaml",
        )
        all_attention_problems.extend(
            process_input_file(config, input_file, output_file, builder)
        )

    bench_file = f"{output_dir}/bench_table_{timestamp}.bin"
    create_bench_table_binary(builder, all_attention_problems, config, bench_file)
    logger.info(f"Benchmark table written to: {bench_file}")


if __name__ == "__main__":
    main()
