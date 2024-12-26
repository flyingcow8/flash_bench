from FlashBenchData import (
    AttentionBenchTable,
    AttentionProblem,
    AttentionSolution,
)
from FlashBenchData.DataType import DataType
from FlashBenchData.CppApiType import CppApiType
from FlashBenchData.Platform import Platform
from flash_attn_2_cuda import (
    FLASH_FWD,
    FLASH_VARLEN_FWD,
    FLASH_BWD,
    FLASH_VARLEN_BWD,
    FLASH_FWD_KVCACHE,
    FLASH_FWD_INFER,
    FLASH_VARLEN_INFER,
)

PLATFORM_MAP = {
    Platform.MXC500: "MXC500",
    Platform.MXC550: "MXC550",
}


def convert_api_to_fb(api):
    """
    Convert API enum from YAML to FlashBenchData CppApiType enum.
    """
    api_map = {
        FLASH_FWD: CppApiType.FlashFwd,
        FLASH_VARLEN_FWD: CppApiType.FlashVarlenFwd,
        FLASH_BWD: CppApiType.FlashBwd,
        FLASH_VARLEN_BWD: CppApiType.FlashVarlenBwd,
        FLASH_FWD_KVCACHE: CppApiType.FlashFwdKvcache,
        FLASH_FWD_INFER: CppApiType.FlashFwdInfer,
        FLASH_VARLEN_INFER: CppApiType.FlashVarlenFwdInfer,
    }

    return api_map.get(api, CppApiType.FlashFwd)


def convert_platform_to_fb(platform):
    """
    Convert platform type string from config to FlashBenchData Platform enum.
    """
    platform_map = {
        "MXC500": Platform.MXC500,
        "MXC550": Platform.MXC550,
    }

    return platform_map.get(platform, Platform.MXC500)


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


def create_attention_problem(builder, params, best_solution):

    # Create all nested first
    seqlens_q_vector = create_attention_int_vector(builder, params["seqlens_q"])
    seqlens_kv_vector = create_attention_int_vector(builder, params["seqlens_kv"])
    solution = create_attention_solution(builder, best_solution)

    # Now start the AttentionProblem
    AttentionProblem.AttentionProblemStart(builder)
    AttentionProblem.AttentionProblemAddDtype(
        builder, DataType.float16 if params["is_fp16"] else DataType.bfloat16
    )
    AttentionProblem.AttentionProblemAddHeadDim(builder, params["head_dim"])
    AttentionProblem.AttentionProblemAddHeadDim(builder, params["head_dim_v"])
    AttentionProblem.AttentionProblemAddNumHeadsQ(builder, params["num_heads_q"])
    AttentionProblem.AttentionProblemAddNumHeadsKv(builder, params["num_heads_kv"])
    AttentionProblem.AttentionProblemAddBatchSize(builder, params["batch_size"])
    AttentionProblem.AttentionProblemAddSeqlensQ(builder, seqlens_q_vector)
    AttentionProblem.AttentionProblemAddSeqlensKv(builder, seqlens_kv_vector)
    AttentionProblem.AttentionProblemAddTotalSeqlensQ(
        builder, params["total_seqlens_q"]
    )
    AttentionProblem.AttentionProblemAddTotalSeqlensKv(
        builder, params["total_seqlens_kv"]
    )
    AttentionProblem.AttentionProblemAddMaxSeqlenQ(builder, params["max_seqlen_q"])
    AttentionProblem.AttentionProblemAddMaxSeqlenKv(builder, params["max_seqlen_kv"])
    AttentionProblem.AttentionProblemAddCausal(builder, params["causal"])
    AttentionProblem.AttentionProblemAddDropout(builder, params["dropout"])
    AttentionProblem.AttentionProblemAddAlibi(builder, params["alibi"])
    AttentionProblem.AttentionProblemAddWindowLeft(builder, params["window_left"])
    AttentionProblem.AttentionProblemAddWindowRight(builder, params["window_right"])
    AttentionProblem.AttentionProblemAddAttnMask(builder, params["attn_mask"])
    AttentionProblem.AttentionProblemAddDeterministic(builder, params["deterministic"])
    AttentionProblem.AttentionProblemAddPagedKv(builder, params["paged_kv"])
    AttentionProblem.AttentionProblemAddPagedBlockSize(
        builder, params["paged_block_size"]
    )
    AttentionProblem.AttentionProblemAddPagedNumBlocks(
        builder, params["paged_num_blocks"]
    )
    AttentionProblem.AttentionProblemAddAppendKv(builder, params["append_kv"])
    AttentionProblem.AttentionProblemAddRope(builder, params["rope"])
    AttentionProblem.AttentionProblemAddHashCode(builder, params["hash_code"])
    AttentionProblem.AttentionProblemAddCppapi(
        builder, convert_api_to_fb(params["api"])
    )
    AttentionProblem.AttentionProblemAddSolution(builder, solution)
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
    AttentionBenchTable.AttentionBenchTableAddPlatform(
        builder, convert_platform_to_fb(config["platform"])
    )
    AttentionBenchTable.AttentionBenchTableAddVersion(builder, version_offset)
    bench_table = AttentionBenchTable.AttentionBenchTableEnd(builder)

    # Finish the FlatBuffer
    builder.Finish(bench_table)

    # Get the binary buffer
    buf = builder.Output()

    # Now you can write this buffer to a file or send it over the network
    with open(output_file, "wb") as f:
        f.write(buf)

def merge_tables(base_file, new_file, merged_file):
    with open(base_file, "rb") as f:
        base_buf = f.read()
        
    with open(new_file, "rb") as f:
        new_buf = f.read()

    base_table = AttentionBenchTable.GetRootAsAttentionBenchTable(base_buf, 0)
    new_table = AttentionBenchTable.GetRootAsAttentionBenchTable(new_buf, 0)

    # Check if versions match
    base_version = base_table.Version().decode('utf-8') if base_table.Version() else None
    new_version = new_table.Version().decode('utf-8') if new_table.Version() else None
    
    if base_version != new_version:
        raise ValueError(f"Version mismatch: base={base_version}, new={new_version}")

    # Check if platforms match
    base_platform = base_table.Platform()
    new_platform = new_table.Platform()

    if base_platform != new_platform:
        raise ValueError(f"Platform mismatch: base={PLATFORM_MAP[base_platform]}, new={PLATFORM_MAP[new_platform]}")

    # Create a new builder for the merged table
    builder = flatbuffers.Builder(1024)

    # Create hash map of base problems for quick lookup
    base_problems = {}
    for i in range(base_table.ProblemsLength()):
        base_problem = base_table.Problems(i)
        base_problems[base_problem.HashCode()] = base_problem

    # Collect problems, updating solutions for matching hash codes
    problems = []
    seen_hashes = set()
    
    # Process new problems first to get updated solutions
    for i in range(new_table.ProblemsLength()):
        new_problem = new_table.Problems(i)
        hash_code = new_problem.HashCode()
        seen_hashes.add(hash_code)
        
        if hash_code in base_problems:
            # Use base problem but update its solution from new problem
            base_prob = base_problems[hash_code]
            problems.append((base_prob, new_problem.Solution()))
        else:
            # New unique problem
            problems.append((new_problem, new_problem.Solution()))

    # Add remaining base problems that weren't updated
    for i in range(base_table.ProblemsLength()):
        base_problem = base_table.Problems(i)
        if base_problem.HashCode() not in seen_hashes:
            problems.append((base_problem, base_problem.Solution()))

    # Create problem vector
    problem_offsets = []
    for problem, solution in problems:
        # Create seqlens vectors
        seqlens_q = []
        seqlens_kv = []
        for i in range(problem.SeqlensQLength()):
            seqlens_q.append(problem.SeqlensQ(i))
        for i in range(problem.SeqlensKvLength()):
            seqlens_kv.append(problem.SeqlensKv(i))
            
        seqlens_q_vec = create_attention_int_vector(builder, seqlens_q)
        seqlens_kv_vec = create_attention_int_vector(builder, seqlens_kv)

        # Create solution
        solution_dict = {
            "head_dim": solution.HeadDim(),
            "grid_type": solution.GridType(),
            "balance_type": solution.BalanceType(),
            "kernel_type": solution.KernelType(),
            "num_splits": solution.NumSplits(),
            "kernel_id": solution.KernelId().decode('utf-8') if solution.KernelId() else ""
        }
        solution_offset = create_attention_solution(builder, solution_dict)

        # Create problem
        AttentionProblem.AttentionProblemStart(builder)
        AttentionProblem.AttentionProblemAddDtype(builder, problem.Dtype())
        AttentionProblem.AttentionProblemAddHeadDim(builder, problem.HeadDim())
        AttentionProblem.AttentionProblemAddHeadDimV(builder, problem.HeadDimV())
        AttentionProblem.AttentionProblemAddNumHeadsQ(builder, problem.NumHeadsQ())
        AttentionProblem.AttentionProblemAddNumHeadsKv(builder, problem.NumHeadsKv())
        AttentionProblem.AttentionProblemAddBatchSize(builder, problem.BatchSize())
        AttentionProblem.AttentionProblemAddSeqlensQ(builder, seqlens_q_vec)
        AttentionProblem.AttentionProblemAddSeqlensKv(builder, seqlens_kv_vec)
        AttentionProblem.AttentionProblemAddTotalSeqlensQ(builder, problem.TotalSeqlensQ())
        AttentionProblem.AttentionProblemAddTotalSeqlensKv(builder, problem.TotalSeqlensKv())
        AttentionProblem.AttentionProblemAddMaxSeqlenQ(builder, problem.MaxSeqlenQ())
        AttentionProblem.AttentionProblemAddMaxSeqlenKv(builder, problem.MaxSeqlenKv())
        AttentionProblem.AttentionProblemAddCausal(builder, problem.Causal())
        AttentionProblem.AttentionProblemAddDropout(builder, problem.Dropout())
        AttentionProblem.AttentionProblemAddAlibi(builder, problem.Alibi())
        AttentionProblem.AttentionProblemAddWindowLeft(builder, problem.WindowLeft())
        AttentionProblem.AttentionProblemAddWindowRight(builder, problem.WindowRight())
        AttentionProblem.AttentionProblemAddAttnMask(builder, problem.AttnMask())
        AttentionProblem.AttentionProblemAddDeterministic(builder, problem.Deterministic())
        AttentionProblem.AttentionProblemAddPagedKv(builder, problem.PagedKv())
        AttentionProblem.AttentionProblemAddPagedBlockSize(builder, problem.PagedBlockSize())
        AttentionProblem.AttentionProblemAddPagedNumBlocks(builder, problem.PagedNumBlocks())
        AttentionProblem.AttentionProblemAddAppendKv(builder, problem.AppendKv())
        AttentionProblem.AttentionProblemAddRope(builder, problem.Rope())
        AttentionProblem.AttentionProblemAddHashCode(builder, problem.HashCode())
        AttentionProblem.AttentionProblemAddPyapi(builder, problem.Pyapi())
        AttentionProblem.AttentionProblemAddCppapi(builder, problem.Cppapi())
        AttentionProblem.AttentionProblemAddSolution(builder, solution_offset)
        problem_offsets.append(AttentionProblem.AttentionProblemEnd(builder))

    # Create problems vector
    AttentionBenchTable.AttentionBenchTableStartProblemsVector(builder, len(problem_offsets))
    for problem in reversed(problem_offsets):
        builder.PrependUOffsetTRelative(problem)
    problems_vector = builder.EndVector()

    # Create version string
    version_offset = builder.CreateString(base_version)

    # Create merged table
    AttentionBenchTable.AttentionBenchTableStart(builder)
    AttentionBenchTable.AttentionBenchTableAddProblems(builder, problems_vector)
    AttentionBenchTable.AttentionBenchTableAddPlatform(builder, base_platform)
    AttentionBenchTable.AttentionBenchTableAddVersion(builder, version_offset)
    bench_table = AttentionBenchTable.AttentionBenchTableEnd(builder)

    builder.Finish(bench_table)
    buf = builder.Output()

    # Write merged table to file
    with open(merged_file, "wb") as f:
        f.write(buf)