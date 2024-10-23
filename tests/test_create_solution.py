import unittest
import os
import flatbuffers
from FlashBenchData.KernelType import KernelType
from modules.export_fb import create_solution_test_binary
from FlashBenchData import AttentionBenchTable, AttentionProblem, AttentionSolution
from FlashBenchData.GridType import GridType
from FlashBenchData.BalanceType import BalanceType
from FlashBenchData.TAG import TAG


class TestGenerateSolution(unittest.TestCase):
    def setUp(self):
        self.test_file = "./test_solution.bin"
        self.output_dir = "./"

    def tearDown(self):
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

    
    def test_create_solution_test_binary(self):
        # Test parameters
        head_dim = 64
        grid_type = GridType.NHB
        kernel_type = KernelType.Fwd
        kernel_id = "test_kernel"
        balance_type = BalanceType.Mode1
        num_splits = 2
        config = {"project": {"version": "1.0"}}

        # Create the binary file
        create_solution_test_binary(
            head_dim,
            grid_type,
            kernel_type,
            kernel_id,
            balance_type,
            num_splits,
            config,
            self.output_dir
        )

        # Check if the file was created
        self.assertTrue(os.path.exists(self.test_file))

        # Read and verify the contents of the binary file
        with open(self.test_file, "rb") as f:
            buf = f.read()
            bench_table = AttentionBenchTable.AttentionBenchTable.GetRootAsAttentionBenchTable(buf, 0)

        # Verify the contents
        self.assertEqual(bench_table.ProblemsLength(), 1)
        self.assertEqual(bench_table.Version().decode('utf-8'), "1.0")
        self.assertEqual(bench_table.Tag(), TAG.TestFwd)

        problem = bench_table.Problems(0)
        solution = problem.SolutionFwd()

        self.assertEqual(solution.HeadDim(), head_dim)
        self.assertEqual(solution.GridType(), grid_type)
        self.assertEqual(solution.KernelType(), kernel_type)
        self.assertEqual(solution.KernelId().decode('utf-8'), kernel_id)
        self.assertEqual(solution.BalanceType(), balance_type)
        self.assertEqual(solution.NumSplits(), num_splits)

if __name__ == "__main__":
    unittest.main()
