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
    hash_code = builder.CreateString(params["hash_code"])

    # Now start the AttentionProblem
    AttentionProblem.AttentionProblemStart(builder)
    AttentionProblem.AttentionProblemAddDtype(
        builder, DataType.float16 if params["is_fp16"] else DataType.bfloat16
    )
    AttentionProblem.AttentionProblemAddHeadDim(builder, params["head_dim"])
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
    AttentionProblem.AttentionProblemAddHashcode(builder, hash_code)
    AttentionProblem.AttentionProblemAddCppApi(
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
