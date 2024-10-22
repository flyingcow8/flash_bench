import yaml
from FlashBenchData import (
    AttentionBenchTable,
    AttentionProblem,
    AttentionSolution,
)
from FlashBenchData.DataType import DataType
from collections import OrderedDict
import yaml


def ordered_dict_representer(dumper, data):
    return dumper.represent_mapping("tag:yaml.org,2002:map", data.items())


yaml.add_representer(OrderedDict, ordered_dict_representer)


def convert_dtype_to_fb(dtype_str):
    """
    Convert data type string from YAML to FlashBenchData DataType enum.
    """

    dtype_map = {
        "float16": DataType.float16,
        "bfloat16": DataType.bfloat16,
    }

    return dtype_map.get(
        dtype_str.lower(), DataType.bfloat16
    )  # Default to bfloat16 if not found


def find_best_solution(solutions):
    if not solutions:
        return {}  # Return an empty dict if results is empty

    return min(solutions, key=lambda x: x.get("time_us", float("inf")))


def append_results_to_yaml(
    params, fwd_results, best_fwd_solution, bwd_results, best_bwd_solution, output_file
):
    # Create a new dictionary to maintain the desired order
    ordered_params = OrderedDict()

    # Copy existing keys from params
    for key in params:
        ordered_params[key] = params[key]

    ordered_params["fwd_results"] = [
        str(list(result.values())) for result in fwd_results
    ]
    ordered_params["best_fwd_solution"] = str(list(best_fwd_solution.values()))

    if params.get("is_training", True):
        ordered_params["bwd_results"] = [
            str(list(result.values())) for result in bwd_results
        ]
        ordered_params["best_bwd_solution"] = str(list(best_bwd_solution.values()))

    with open(output_file, "a") as f:
        yaml.dump([ordered_params], f, default_flow_style=False)
        f.write("\n")  # Add a newline for better readability between entries


def create_attention_int_vector(builder, int_list):
    AttentionProblem.AttentionProblemStartSeqlensQVector(builder, len(int_list))
    for it in reversed(int_list):
        builder.PrependInt32(it)
    return builder.EndVector()


def create_attention_solution(builder, solution):
    AttentionSolution.AttentionSolutionStart(builder)
    AttentionSolution.AttentionSolutionAddHeadDim(builder, solution["head_dim"])
    AttentionSolution.AttentionSolutionAddBlockM(builder, solution["block_m"])
    AttentionSolution.AttentionSolutionAddBlockN(builder, solution["block_n"])
    AttentionSolution.AttentionSolutionAddNumWarps(builder, solution["num_warps"])
    AttentionSolution.AttentionSolutionAddGridType(builder, solution["grid_type"])
    AttentionSolution.AttentionSolutionAddBlanceType(builder, solution["balance_type"])
    AttentionSolution.AttentionSolutionAddOpType(builder, solution["op_type"])
    return AttentionSolution.AttentionSolutionEnd(builder)


def create_attention_problem(builder, params, best_fwd_solution, best_bwd_solution):
    # Create all nested first
    seqlens_q_vector = create_attention_int_vector(builder, params["seqlens_q"])
    seqlens_kv_vector = create_attention_int_vector(builder, params["seqlens_kv"])
    solution_fwd = create_attention_solution(builder, best_fwd_solution)
    solution_bwd = create_attention_solution(builder, best_bwd_solution)

    # Now start the AttentionProblem
    AttentionProblem.AttentionProblemStart(builder)
    AttentionProblem.AttentionProblemAddDtype(
        builder, convert_dtype_to_fb(params["dtype"])
    )
    AttentionProblem.AttentionProblemAddHeadDim(builder, params["head_dim"])
    AttentionProblem.AttentionProblemAddNumHeadsQ(builder, params["num_heads_q"])
    AttentionProblem.AttentionProblemAddNumHeadsKv(builder, params["num_heads_kv"])
    AttentionProblem.AttentionProblemAddBatchSize(builder, params["batch_size"])
    AttentionProblem.AttentionProblemAddSeqlensQ(builder, seqlens_q_vector)
    AttentionProblem.AttentionProblemAddSeqlensKv(builder, seqlens_kv_vector)
    AttentionProblem.AttentionProblemAddCausal(builder, params["causal"])
    AttentionProblem.AttentionProblemAddDropout(builder, params["dropout"])
    AttentionProblem.AttentionProblemAddAlibi(builder, params["alibi"])
    AttentionProblem.AttentionProblemAddLocal(builder, params["local"])
    AttentionProblem.AttentionProblemAddAttnMask(builder, params["attn_mask"])
    AttentionProblem.AttentionProblemAddDeterministic(builder, params["deterministic"])
    AttentionProblem.AttentionProblemAddIsTraining(builder, params["is_training"])
    AttentionProblem.AttentionProblemAddSolutionFwd(builder, solution_fwd)
    AttentionProblem.AttentionProblemAddSolutionBwd(builder, solution_bwd)
    return AttentionProblem.AttentionProblemEnd(builder)


def create_bench_table_binary(
    builder, problems, config, output_file="attention_bench_table.bin"
):
    # Create a vector of problem offsets
    AttentionBenchTable.AttentionBenchTableStartProblemsVector(builder, len(problems))
    for problem in reversed(problems):
        builder.PrependUOffsetTRelative(problem)
    problems_vector = builder.EndVector()

    # Create an AttentionBenchTable
    AttentionBenchTable.AttentionBenchTableStart(builder)
    AttentionBenchTable.AttentionBenchTableAddProblems(builder, problems_vector)
    version = config["project"]["version"]
    AttentionBenchTable.AttentionBenchTableAddVersion(builder, version)
    bench_table = AttentionBenchTable.AttentionBenchTableEnd(builder)

    # Finish the FlatBuffer
    builder.Finish(bench_table)

    # Get the binary buffer
    buf = builder.Output()

    # Now you can write this buffer to a file or send it over the network
    with open(output_file, "wb") as f:
        f.write(buf)
