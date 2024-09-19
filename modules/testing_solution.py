import flatbuffers
from FlashBenchData import AttentionSolution
from FlashBenchData.OpType import OpType
from FlashBenchData.TAG import TAG
import os


def create_test_solution(
    head_dim,
    tile_m,
    tile_n,
    num_waves,
    grid_type,
    balance_type,
    op_type,
    config=None,
    output_dir="./",
):
    builder = flatbuffers.Builder(1024)

    # Create AttentionSolution
    AttentionSolution.AttentionSolutionStart(builder)
    AttentionSolution.AttentionSolutionAddHeadDim(builder, head_dim)
    AttentionSolution.AttentionSolutionAddTileM(builder, tile_m)
    AttentionSolution.AttentionSolutionAddTileN(builder, tile_n)
    AttentionSolution.AttentionSolutionAddNumWaves(builder, num_waves)
    AttentionSolution.AttentionSolutionAddGridType(builder, grid_type)
    AttentionSolution.AttentionSolutionAddBlanceType(builder, balance_type)
    AttentionSolution.AttentionSolutionAddOpType(builder, op_type)
    solution = AttentionSolution.AttentionSolutionEnd(builder)

    # Create AttentionProblem
    from FlashBenchData import AttentionProblem

    AttentionProblem.AttentionProblemStart(builder)
    if op_type == OpType.Fwd:
        AttentionProblem.AttentionProblemAddSolutionFwd(builder, solution)
    elif op_type == OpType.Bwd:
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
        builder, 1 if config is None else config["project"]["version"]
    )

    # Set tag based on op_type
    tag = TAG.TestFwd if op_type == OpType.Fwd else TAG.TestBwd
    AttentionBenchTable.AttentionBenchTableAddTag(builder, tag)

    bench_table = AttentionBenchTable.AttentionBenchTableEnd(builder)

    builder.Finish(bench_table)
    buf = builder.Output()

    with open(f"{output_dir}/test_solution.bin", "wb") as f:
        f.write(buf)
