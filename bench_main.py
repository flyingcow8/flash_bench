import yaml
import os
from modules.bench_process import bench_process
from modules.postprocess import (
    append_results_to_yaml,
    find_best_solution,
    create_attention_problem,
    create_bench_table_binary,
)

import yaml
import flatbuffers
from FlashBenchData import AttentionBenchTable
import datetime

# Read the config.yaml file
with open("config.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

# Extract the input files from the config
input_files = config.get("input_path", [])

# Generate output filename
output_dir = config.get("output_dir", "output")
os.makedirs(output_dir, exist_ok=True)

# Print some information from the config
print(f"Project Name: {config['project']['name']}")
print(f"Project Version: {config['project']['version']}")
print(f"Input files to process: {', '.join(input_files)}")
print("=" * 50 + "\n")

# Create a FlatBuffers builder
builder = flatbuffers.Builder(1024)
attention_problems = []

for input_file in input_files:
    if not os.path.exists(input_file):
        print(f"Warning: File {input_file} not found. Skipping.")
        continue

    print(f"Processing file: {input_file}")

    # Read and parse the YAML file of mha params
    with open(input_file, "r") as file:
        params_file = yaml.safe_load(file)

    output_yaml = os.path.join(
        output_dir, f"{os.path.basename(input_file)}_results.yaml"
    )

    # Run Flash Attention for each case in the current YAML file
    for idx, params in enumerate(params_file["mha_params"]):
        print(f"Running case {idx}:")
        best_fwd_solution = {}
        best_bwd_solution = {}
        fwd_results, bwd_results = bench_process(config, params)
        best_fwd_solution = find_best_solution(fwd_results)
        if params.get("is_training", True):
            best_bwd_solution = find_best_solution(bwd_results)
        append_results_to_yaml(
            params,
            fwd_results,
            best_fwd_solution,
            bwd_results,
            best_bwd_solution,
            output_yaml,
        )
        problem = create_attention_problem(builder, params, best_fwd_solution, best_bwd_solution)
        attention_problems.append(problem)

    print(f"Finished processing {input_file}\n")
    print("=" * 50 + "\n")

# Write the binary buffer to a file
current_date = datetime.datetime.now().strftime("%Y%m%d")
bench_file = f"{output_dir}/bench_table_{current_date}.bin"
create_bench_table_binary(builder, attention_problems, config, bench_file)
print(f"Benchmark table written to: {bench_file}")


