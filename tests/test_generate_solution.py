import unittest
import os
import flatbuffers
from modules.testing_solution import create_solution_test_binary
from FlashBenchData import AttentionBenchTable, AttentionProblem, AttentionSolution
from FlashBenchData.GridType import GridType
from FlashBenchData.BalanceType import BalanceType
from FlashBenchData.OpType import OpType
from FlashBenchData.TAG import TAG


class TestGenerateSolution(unittest.TestCase):
    def setUp(self):
        self.test_file = "./test_solution.bin"
        self.output_dir = "./"

    def tearDown(self):
        pass
        # if os.path.exists(self.test_file):
        #     os.remove(self.test_file)

    def test_generate_attention_solution(self):
        # Test parameters
        head_dim = 96
        block_m = 64
        block_n = 64
        num_warps = 4
        grid_type = GridType.NHB
        balance_type = BalanceType.Mode1
        op_type = OpType.Fwd

        # Generate the solution
        create_solution_test_binary(
            head_dim,
            block_m,
            block_n,
            num_warps,
            grid_type,
            balance_type,
            op_type,
            None,
            self.output_dir
        )

        # Check if the file was created
        self.assertTrue(os.path.exists(self.test_file))

        # Read and verify the contents of the file
        with open(self.test_file, "rb") as f:
            buf = f.read()
            bench_table = AttentionBenchTable.AttentionBenchTable.GetRootAs(buf, 0)

        # Verify the bench table
        self.assertEqual(bench_table.Version(), 1)
        self.assertEqual(bench_table.Tag(), TAG.TestFwd)
        self.assertEqual(bench_table.ProblemsLength(), 1)

        # Verify the problem
        problem = bench_table.Problems(0)
        self.assertIsInstance(problem, AttentionProblem.AttentionProblem)

        # Verify the solution
        solution = problem.SolutionFwd()
        self.assertIsInstance(solution, AttentionSolution.AttentionSolution)

        # Verify the values
        self.assertEqual(solution.HeadDim(), head_dim)
        self.assertEqual(solution.TileM(), block_m)
        self.assertEqual(solution.TileN(), block_n)
        self.assertEqual(solution.NumWarps(), num_warps)
        self.assertEqual(solution.GridType(), grid_type)
        self.assertEqual(solution.BlanceType(), balance_type)
        self.assertEqual(solution.OpType(), op_type)


if __name__ == "__main__":
    unittest.main()
