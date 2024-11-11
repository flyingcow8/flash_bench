import argparse
import os
from FlashBenchData import AttentionBenchTable
from FlashBenchData.GridType import GridType
from FlashBenchData.KernelType import KernelType
from FlashBenchData.BalanceType import BalanceType
from FlashBenchData.TAG import TAG

def parse_solution_binary(binary_file):
    """Parse a solution test binary file and print its contents."""
    if not os.path.exists(binary_file):
        print(f"Error: File {binary_file} does not exist")
        return

    with open(binary_file, "rb") as f:
        buf = f.read()
        bench_table = AttentionBenchTable.AttentionBenchTable.GetRootAsAttentionBenchTable(buf, 0)

    tag = bench_table.Tag()
    if tag not in [TAG.TestFwd, TAG.TestBwd]:
        print(f"Error: Invalid tag {tag}, must be {TAG.TestFwd} or {TAG.TestBwd}")
        return

    problems_length = bench_table.ProblemsLength()
    if problems_length != 1:
        print(f"Error: Number of problems must be 1, got {problems_length}")
        return

    print(f"Version: {bench_table.Version().decode('utf-8')}")

    problem = bench_table.Problems(0)
    solution = problem.SolutionFwd() if tag == TAG.TestFwd else problem.SolutionBwd()
    
    print(f"\nSolution {tag}:")
    print(f"  Head Dimension: {solution.HeadDim()}")
    print(f"  Grid Type: {solution.GridType()}")
    print(f"  Kernel Type: {solution.KernelType()}")
    print(f"  Kernel ID: {solution.KernelId().decode('utf-8')}")
    print(f"  Balance Type: {solution.BalanceType()}")
    print(f"  Number of Splits: {solution.NumSplits()}")

def main():
    parser = argparse.ArgumentParser(description='Parse a solution test binary file')
    parser.add_argument('binary_file', help='Path to the solution test binary file')
    args = parser.parse_args()

    parse_solution_binary(args.binary_file)

if __name__ == "__main__":
    main()

