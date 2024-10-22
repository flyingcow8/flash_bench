import flatbuffers
from FlashBenchData import AttentionSolution
from FlashBenchData.KernelType import KernelType
from FlashBenchData.TAG import TAG
import os


def create_solution_test_binary(
    head_dim,
    kernel,
    kernel_type,
    grid_type,
    balance_type=0,
    num_splits=1,
    config=None,
    output_dir="./",
):
    builder = flatbuffers.Builder(1024)

    # the keys of kernel are defined in kernel_registry.py
    block_m = kernel["block_m"]
    block_n = kernel["block_n"]
    num_warps = kernel["num_warps"]
    grid_type = kernel["grid_type"]

    # Create AttentionSolution
    AttentionSolution.AttentionSolutionStart(builder)
    AttentionSolution.AttentionSolutionAddHeadDim(builder, head_dim)
    AttentionSolution.AttentionSolutionAddBlockM(builder, block_m)
    AttentionSolution.AttentionSolutionAddBlockN(builder, block_n)
    AttentionSolution.AttentionSolutionAddNumWarps(builder, num_warps)
    AttentionSolution.AttentionSolutionAddGridType(builder, grid_type)
    AttentionSolution.AttentionSolutionAddKernelType(builder, kernel_type)

    if kernel_type == KernelType.Fwd or kernel_type == KernelType.Bwd:
        balance_type = kernel["balance_type"]
        AttentionSolution.AttentionSolutionAddBalanceType(builder, balance_type)

    if kernel_type == KernelType.Fwd or kernel_type == KernelType.FwdSplitkv:
        is_q_in_regs = kernel["is_q_in_regs"]
        AttentionSolution.AttentionSolutionAddIsQInRegs(builder, is_q_in_regs)
        share_q_k_smem = kernel["share_q_k_smem"]
        AttentionSolution.AttentionSolutionAddShareQKSmem(builder, share_q_k_smem)
        num_splits = kernel["num_splits"]
        AttentionSolution.AttentionSolutionAddNumSplits(builder, num_splits)

    if kernel_type == KernelType.Bwd:
        is_k_in_regs = kernel["is_k_in_regs"]
        AttentionSolution.AttentionSolutionAddIsKInRegs(builder, is_k_in_regs)
        is_v_in_regs = kernel["is_v_in_regs"]
        AttentionSolution.AttentionSolutionAddIsVInRegs(builder, is_v_in_regs)
        atom_layout_mdq = kernel["atom_layout_mdq"]
        AttentionSolution.AttentionSolutionAddAtomLayoutMdq(builder, atom_layout_mdq)
        atom_layout_msdp = kernel["atom_layout_msdp"]
        AttentionSolution.AttentionSolutionAddAtomLayoutMsdp(builder, atom_layout_msdp)
        no_double_buffer = kernel["no_double_buffer"]
        AttentionSolution.AttentionSolutionAddNoDoubleBuffer(builder, no_double_buffer)

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
    AttentionBenchTable.AttentionBenchTableStart(builder)
    AttentionBenchTable.AttentionBenchTableAddProblems(builder, problems)
    AttentionBenchTable.AttentionBenchTableAddVersion(
        builder,
        builder.CreateString(
            "1.0" if config is None else str(config["project"]["version"])
        ),
    )

    # Set tag based on kernel_type
    tag = TAG.TestFwd if kernel_type == KernelType.Fwd else TAG.TestBwd
    AttentionBenchTable.AttentionBenchTableAddTag(builder, tag)

    bench_table = AttentionBenchTable.AttentionBenchTableEnd(builder)

    builder.Finish(bench_table)
    buf = builder.Output()

    with open(f"{output_dir}/test_solution.bin", "wb") as f:
        f.write(buf)
