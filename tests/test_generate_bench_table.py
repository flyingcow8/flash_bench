import unittest
import os
import yaml
import flatbuffers
from modules.postprocess import create_attention_problem, create_bench_table_binary
from FlashBenchData import AttentionBenchTable, AttentionProblem, AttentionSolution
from FlashBenchData.GridType import GridType
from FlashBenchData.BalanceType import BalanceType
from FlashBenchData.OpType import OpType

class TestGenerateBenchTable(unittest.TestCase):
    def setUp(self):
        self.test_file = "test_bench_table.fb"
        self.yaml_file = "data/proj_0.yaml"

    def tearDown(self):
        pass
        # if os.path.exists(self.test_file):
        #     os.remove(self.test_file)

    def test_generate_bench_table(self):
        # Read proj_0.yaml
        with open(self.yaml_file, 'r') as f:
            yaml_data = yaml.safe_load(f)

        # Extract MHA parameters
        mha_params = yaml_data['mha_params'][0]

        # Create mock solutions
        mock_solution = {
            "head_dim": mha_params['head_dim'],
            "block_m": 64,
            "block_n": 64,
            "num_warps": 4,
            "grid_type": GridType.NHB,
            "balance_type": BalanceType.Mode1,
            "op_type": OpType.Fwd
        }

        # Create a FlatBuffers builder
        builder = flatbuffers.Builder(1024)

        # Create an attention problem
        problem = create_attention_problem(builder, mha_params, mock_solution, mock_solution)

        # Create a list of problems
        problems = [problem]

        # Create the bench table binary
        config = {"project": {"version": 1}}
        create_bench_table_binary(builder, problems, config, self.test_file)

        # Verify the file was created
        self.assertTrue(os.path.exists(self.test_file))

        # Read and verify the contents of the file
        with open(self.test_file, "rb") as f:
            buf = f.read()
            bench_table = AttentionBenchTable.AttentionBenchTable.GetRootAs(buf, 0)

        # Verify the version
        self.assertEqual(bench_table.Version(), 1)

        # Verify the number of problems
        self.assertEqual(bench_table.ProblemsLength(), 1)

        # Verify the contents of the problem
        problem = bench_table.Problems(0)
        self.assertEqual(problem.HeadDim(), mha_params['head_dim'])
        self.assertEqual(problem.NumHeadsQ(), mha_params['num_heads_q'])
        self.assertEqual(problem.NumHeadsKv(), mha_params['num_heads_kv'])
        self.assertEqual(problem.BatchSize(), mha_params['batch_size'])
        self.assertEqual(problem.SeqlensQ(0), mha_params['seqlen_q'])
        self.assertEqual(problem.SeqlensKv(0), mha_params['seqlen_kv'])
        self.assertEqual(problem.Causal(), mha_params['causal'])
        self.assertEqual(problem.Dropout(), mha_params['dropout'])

if __name__ == "__main__":
    unittest.main()
