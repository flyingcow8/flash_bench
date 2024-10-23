import unittest
import os
import flatbuffers
from modules.export_fb import create_bench_table_binary, create_attention_problem
from FlashBenchData import AttentionBenchTable, AttentionProblem, AttentionSolution
from FlashBenchData.TAG import TAG
from FlashBenchData.DataType import DataType
from FlashBenchData.KernelType import KernelType

class TestExportFb(unittest.TestCase):
    def setUp(self):
        self.output_file = "test_attention_bench_table.bin"

    def tearDown(self):
        if os.path.exists(self.output_file):
            os.remove(self.output_file)

    def test_create_bench_table_binary(self):
        # Prepare test data
        builder = flatbuffers.Builder(1024)
        problems = []
        
        # Create a sample problem
        params = {
            "dtype": "float16",
            "head_dim": 64,
            "num_heads_q": 8,
            "num_heads_kv": 8,
            "batch_size": 1,
            "seqlens_q": [128],
            "seqlens_kv": [128],
            "causal": True,
            "dropout": False,
            "alibi": False,
            "window_left": 0,
            "window_right": 0,
            "attn_mask": False,
            "deterministic": True,
            "is_training": True
        }
        
        best_fwd_solution = {
            "head_dim": 64,
            "grid_type": 0,
            "balance_type": 0,
            "kernel_type": KernelType.Fwd,
            "num_splits": 1,
            "kernel_id": "test_kernel_fwd"
        }
        
        best_bwd_solution = {
            "head_dim": 64,
            "grid_type": 0,
            "balance_type": 0,
            "kernel_type": KernelType.Bwd,
            "num_splits": 1,
            "kernel_id": "test_kernel_bwd"
        }

        problem = create_attention_problem(builder, params, best_fwd_solution, best_bwd_solution)
        problems.append(problem)

        config = {"project": {"version": "1.0.0"}}

        # Call the function
        create_bench_table_binary(builder, problems, config, self.output_file)

        # Verify the output
        self.assertTrue(os.path.exists(self.output_file))

        # Read and verify the contents
        with open(self.output_file, "rb") as f:
            buf = f.read()
            bench_table = AttentionBenchTable.AttentionBenchTable.GetRootAsAttentionBenchTable(buf, 0)

        # Verify bench table properties
        self.assertEqual(bench_table.ProblemsLength(), 1)
        self.assertEqual(bench_table.Tag(), TAG.Deploy)
        self.assertEqual(bench_table.Version().decode('utf-8'), "1.0.0")

        # Verify problem properties
        problem = bench_table.Problems(0)
        self.assertEqual(problem.Dtype(), DataType.float16)
        self.assertEqual(problem.HeadDim(), 64)
        self.assertEqual(problem.NumHeadsQ(), 8)
        self.assertEqual(problem.NumHeadsKv(), 8)
        self.assertEqual(problem.BatchSize(), 1)
        self.assertEqual(problem.SeqlensQAsNumpy().tolist(), [128])
        self.assertEqual(problem.SeqlensKvAsNumpy().tolist(), [128])
        self.assertTrue(problem.Causal())
        self.assertAlmostEqual(problem.Dropout(), False)
        self.assertFalse(problem.Alibi())
        self.assertEqual(problem.WindowLeft(), 0)
        self.assertEqual(problem.WindowRight(), 0)
        self.assertFalse(problem.AttnMask())
        self.assertTrue(problem.Deterministic())
        self.assertTrue(problem.IsTraining())

        # Verify solution properties
        fwd_solution = problem.SolutionFwd()
        self.assertEqual(fwd_solution.HeadDim(), 64)
        self.assertEqual(fwd_solution.GridType(), 0)
        self.assertEqual(fwd_solution.BalanceType(), 0)
        self.assertEqual(fwd_solution.KernelType(), KernelType.Fwd)
        self.assertEqual(fwd_solution.NumSplits(), 1)
        self.assertEqual(fwd_solution.KernelId().decode('utf-8'), "test_kernel_fwd")

        bwd_solution = problem.SolutionBwd()
        self.assertEqual(bwd_solution.HeadDim(), 64)
        self.assertEqual(bwd_solution.GridType(), 0)
        self.assertEqual(bwd_solution.BalanceType(), 0)
        self.assertEqual(bwd_solution.KernelType(), KernelType.Bwd)
        self.assertEqual(bwd_solution.NumSplits(), 1)
        self.assertEqual(bwd_solution.KernelId().decode('utf-8'), "test_kernel_bwd")

if __name__ == '__main__':
    unittest.main()