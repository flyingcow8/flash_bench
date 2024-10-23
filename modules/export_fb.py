from FlashBenchData import (
    AttentionBenchTable,
    AttentionProblem,
    AttentionSolution,
)
from FlashBenchData.DataType import DataType
import flatbuffers

from FlashBenchData.KernelType import KernelType
from FlashBenchData.TAG import TAG


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


def create_attention_int_vector(builder, int_list):
    AttentionProblem.AttentionProblemStartSeqlensQVector(builder, len(int_list))
    for it in reversed(int_list):
        builder.PrependInt32(it)
    return builder.EndVector()


def create_attention_solution(builder, solution):
    kernel_id_offset = builder.CreateString(solution["kernel_id"])
    AttentionSolution.AttentionSolutionStart(builder)
    AttentionSolution.AttentionSolutionAddHeadDim(builder, solution["head_dim"])
    AttentionSolution.AttentionSolutionAddGridType(builder, solution["grid_type"])
    AttentionSolution.AttentionSolutionAddBalanceType(builder, solution["balance_type"])
    AttentionSolution.AttentionSolutionAddKernelType(builder, solution["kernel_type"])
    AttentionSolution.AttentionSolutionAddNumSplits(builder, solution["num_splits"])
    AttentionSolution.AttentionSolutionAddKernelId(builder, kernel_id_offset)
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
    AttentionProblem.AttentionProblemAddWindowLeft(builder, params["window_left"])
    AttentionProblem.AttentionProblemAddWindowRight(builder, params["window_right"])
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

    # Create a string of version
    version_offset = builder.CreateString(config["project"]["version"])

    # Create an AttentionBenchTable
    AttentionBenchTable.AttentionBenchTableStart(builder)
    AttentionBenchTable.AttentionBenchTableAddProblems(builder, problems_vector)
    AttentionBenchTable.AttentionBenchTableAddTag(builder, TAG.Deploy)
    AttentionBenchTable.AttentionBenchTableAddVersion(builder, version_offset)
    bench_table = AttentionBenchTable.AttentionBenchTableEnd(builder)

    # Finish the FlatBuffer
    builder.Finish(bench_table)

    # Get the binary buffer
    buf = builder.Output()

    # Now you can write this buffer to a file or send it over the network
    with open(output_file, "wb") as f:
        f.write(buf)


def create_solution_test_binary(
    head_dim,
    grid_type,
    kernel_type,
    kernel_id,
    version,
    balance_type=0,
    num_splits=1,
    output_dir="./",
):
    builder = flatbuffers.Builder(1024)

    # Create string offset for kernel_id first
    kernel_id_offset = builder.CreateString(kernel_id)

    # Create AttentionSolution
    AttentionSolution.AttentionSolutionStart(builder)
    AttentionSolution.AttentionSolutionAddHeadDim(builder, head_dim)
    AttentionSolution.AttentionSolutionAddGridType(builder, grid_type)
    AttentionSolution.AttentionSolutionAddKernelType(builder, kernel_type)
    AttentionSolution.AttentionSolutionAddNumSplits(builder, num_splits)
    AttentionSolution.AttentionSolutionAddBalanceType(builder, balance_type)
    AttentionSolution.AttentionSolutionAddKernelId(builder, kernel_id_offset)

    solution = AttentionSolution.AttentionSolutionEnd(builder)

    # Create AttentionProblem
    from FlashBenchData import AttentionProblem

    AttentionProblem.AttentionProblemStart(builder)
    if kernel_type == KernelType.Fwd:
        AttentionProblem.AttentionProblemAddSolutionFwd(builder, solution)
    elif kernel_type == KernelType.Bwd:
        AttentionProblem.AttentionProblemAddSolutionBwd(builder, solution)
    problem = AttentionProblem.AttentionProblemEnd(builder)

    # Create vector of problems
    from FlashBenchData import AttentionBenchTable

    AttentionBenchTable.AttentionBenchTableStartProblemsVector(builder, 1)
    builder.PrependUOffsetTRelative(problem)
    problems = builder.EndVector()

    # Create AttentionBenchTable
    version_offset = builder.CreateString(version)
    AttentionBenchTable.AttentionBenchTableStart(builder)
    AttentionBenchTable.AttentionBenchTableAddProblems(builder, problems)
    AttentionBenchTable.AttentionBenchTableAddVersion(builder, version_offset)

    # Set tag based on kernel_type
    tag = TAG.TestFwd if kernel_type == KernelType.Fwd else TAG.TestBwd
    AttentionBenchTable.AttentionBenchTableAddTag(builder, tag)

    bench_table = AttentionBenchTable.AttentionBenchTableEnd(builder)

    builder.Finish(bench_table)
    buf = builder.Output()

    with open(f"{output_dir}/test_solution.bin", "wb") as f:
        f.write(buf)
