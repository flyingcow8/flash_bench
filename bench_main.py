import yaml
import os
import datetime
import flatbuffers
from modules.bench_process import bench_process
from modules.postprocess import (
    append_results_to_yaml,
    find_best_solution,
    create_attention_problem,
    create_bench_table_binary,
)
from FlashBenchData import AttentionBenchTable
from modules.config_constants import ALLOWED_MHA_APIS

def load_config(config_file):
    with open(config_file, "r") as file:
        return yaml.safe_load(file)

def process_input_file(config, input_file, output_dir, builder):
    print(f"Processing file: {input_file}")
    
    with open(input_file, "r") as file:
        params_file = yaml.safe_load(file)

    if 'mha_api' not in params_file or 'mha_params' not in params_file:
        print(f"Error: 'mha_api' or 'mha_params' not found in {input_file}")
        return []

    # Check if the mha_api is in the allowed list
    mha_api = params_file['mha_api']
    if mha_api not in ALLOWED_MHA_APIS:
        print(f"Error: Invalid 'mha_api' value '{mha_api}' in {input_file}")
        print(f"Allowed values are: {', '.join(ALLOWED_MHA_APIS)}")
        return []

    output_yaml = os.path.join(output_dir, f"{os.path.basename(input_file)}_results.yaml")
    attention_problems = []

    for idx, params in enumerate(params_file["mha_params"]):
        print(f"Running case {idx}:")
        fwd_results, bwd_results = bench_process(config, params)
        best_fwd_solution = find_best_solution(fwd_results)
        best_bwd_solution = find_best_solution(bwd_results) if params.get("is_training", True) else {}
        
        append_results_to_yaml(params, fwd_results, best_fwd_solution, bwd_results, best_bwd_solution, output_yaml)
        problem = create_attention_problem(builder, params, best_fwd_solution, best_bwd_solution)
        attention_problems.append(problem)

    print(f"Finished processing {input_file}\n")
    print("=" * 50 + "\n")
    return attention_problems

def main():
    config = load_config("config.yaml")
    input_files = config.get("input_path", [])
    output_dir = config.get("output_dir", "output")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Project Name: {config['project']['name']}")
    print(f"Project Version: {config['project']['version']}")
    print(f"Input files to process: {', '.join(input_files)}")
    print("=" * 50 + "\n")

    builder = flatbuffers.Builder(1024)
    all_attention_problems = []

    for input_file in input_files:
        if not os.path.exists(input_file):
            print(f"Warning: File {input_file} not found. Skipping.")
            continue
        all_attention_problems.extend(process_input_file(config, input_file, output_dir, builder))

    current_date = datetime.datetime.now().strftime("%Y%m%d")
    bench_file = f"{output_dir}/bench_table_{current_date}.bin"
    create_bench_table_binary(builder, all_attention_problems, config, bench_file)
    print(f"Benchmark table written to: {bench_file}")

if __name__ == "__main__":
    main()



