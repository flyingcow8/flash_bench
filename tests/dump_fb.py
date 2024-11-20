#!/usr/bin/env python3
import sys
import flatbuffers
from typing import List, Optional

# Import the generated Python classes (assuming they're generated in the same directory)
import FlashBenchData.AttentionBenchTable as AttentionBenchTable
import FlashBenchData.AttentionProblem as AttentionProblem
import FlashBenchData.AttentionSolution as AttentionSolution
from FlashBenchData.Platform import Platform
from FlashBenchData.GridType import GridType
from FlashBenchData.BalanceType import BalanceType
from FlashBenchData.PyApiType import PyApiType
from FlashBenchData.CppApiType import CppApiType
from FlashBenchData.KernelType import KernelType
from FlashBenchData.DataType import DataType

GRID_TYPE_MAP = {
    GridType.NHB: "NHB",
    GridType.NBH: "NBH", 
    GridType.HNB: "HNB",
    GridType.BNH: "BNH",
    GridType.BHN: "BHN",
    GridType.HBN: "HBN",
}

BALANCE_TYPE_MAP = {
    BalanceType.None_: "None",
    BalanceType.Mode1: "Mode1",
}

KERNEL_TYPE_MAP = {
    KernelType.Fwd: "Fwd",
    KernelType.FwdSplitkv: "FwdSplitkv", 
    KernelType.Bwd: "Bwd",
}

DATA_TYPE_MAP = {
    DataType.float16: "float16",
    DataType.bfloat16: "bfloat16",
}

PLATFORM_MAP = {
    Platform.MXC500: "MXC500",
    Platform.MXC550: "MXC550",
}

PYAPI_TYPE_MAP = {
    PyApiType.FlashAttnFunc: "FlashAttnFunc",
    PyApiType.FlashAttnQKVPackedFunc: "FlashAttnQKVPackedFunc",
    PyApiType.FlashAttnKVPackedFunc: "FlashAttnKVPackedFunc", 
    PyApiType.FlashAttnVarlenQKVPackedFunc: "FlashAttnVarlenQKVPackedFunc",
    PyApiType.FlashAttnVarlenKVPackedFunc: "FlashAttnVarlenKVPackedFunc",
    PyApiType.FlashAttnVarlenFunc: "FlashAttnVarlenFunc",
    PyApiType.FlashAttnWithKVCache: "FlashAttnWithKVCache",
}

CPPAPI_TYPE_MAP = {
    CppApiType.FlashFwd: "FlashFwd",
    CppApiType.FlashVarlenFwd: "FlashVarlenFwd",
    CppApiType.FlashBwd: "FlashBwd",
    CppApiType.FlashVarlenBwd: "FlashVarlenBwd",
    CppApiType.FlashFwdKvcache: "FlashFwdKvcache",
    CppApiType.FlashFwdInfer: "FlashFwdInfer",
    CppApiType.FlashVarlenFwdInfer: "FlashVarlenFwdInfer",
}

def dump_solution(solution: AttentionSolution.AttentionSolution) -> dict:
    return {
        "head_dim": solution.HeadDim(),
        "grid_type": GRID_TYPE_MAP[solution.GridType()],
        "balance_type": BALANCE_TYPE_MAP[solution.BalanceType()],
        "num_splits": solution.NumSplits(),
        "kernel_type": KERNEL_TYPE_MAP[solution.KernelType()],
        "kernel_id": solution.KernelId().decode('utf-8') if solution.KernelId() else None,
    }

def dump_problem(problem: AttentionProblem.AttentionProblem) -> dict:
    seqlens_q = [problem.SeqlensQ(i) for i in range(problem.SeqlensQLength())]
    seqlens_kv = [problem.SeqlensKv(i) for i in range(problem.SeqlensKvLength())]
    
    return {
        "dtype": DATA_TYPE_MAP[problem.Dtype()],
        "head_dim": problem.HeadDim(),
        "head_dim_v": problem.HeadDimV(),
        "num_heads_q": problem.NumHeadsQ(),
        "num_heads_kv": problem.NumHeadsKv(),
        "batch_size": problem.BatchSize(),
        "seqlens_q": seqlens_q,
        "seqlens_kv": seqlens_kv,
        "total_seqlens_q": problem.TotalSeqlensQ(),
        "total_seqlens_kv": problem.TotalSeqlensKv(),
        "max_seqlen_q": problem.MaxSeqlenQ(),
        "max_seqlen_kv": problem.MaxSeqlenKv(),
        "causal": problem.Causal(),
        "dropout": problem.Dropout(),
        "alibi": problem.Alibi(),
        "window_left": problem.WindowLeft(),
        "window_right": problem.WindowRight(),
        "attn_mask": problem.AttnMask(),
        "deterministic": problem.Deterministic(),
        "paged_kv": problem.PagedKv(),
        "paged_block_size": problem.PagedBlockSize(),
        "paged_num_blocks": problem.PagedNumBlocks(),
        "append_kv": problem.AppendKv(),
        "rope": problem.Rope(),
        "hash_code": problem.HashCode(),
        "pyapi": PYAPI_TYPE_MAP[problem.Pyapi()],
        "cppapi": CPPAPI_TYPE_MAP[problem.Cppapi()],
        "solution": dump_solution(problem.Solution()) if problem.Solution() else None,
    }

def dump_bench_table(table: AttentionBenchTable.AttentionBenchTable) -> dict:
    problems = [dump_problem(table.Problems(i)) for i in range(table.ProblemsLength())]
    
    return {
        "platform": PLATFORM_MAP[table.Platform()],
        "version": table.Version().decode('utf-8') if table.Version() else None,
        "problems": problems,
    }

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <flatbuffer_file>")
        sys.exit(1)
        
    # Read the binary file
    with open(sys.argv[1], 'rb') as f:
        buf = f.read()
        
    # Get the root table
    table = AttentionBenchTable.AttentionBenchTable.GetRootAsAttentionBenchTable(buf, 0)
    
    # Dump to dictionary and print
    import json
    result = dump_bench_table(table)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()