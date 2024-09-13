
import unittest
import yaml
import os
from modules.postprocess import append_results_to_yaml

class TestAppendResultsToYaml(unittest.TestCase):
    def setUp(self):
        self.temp_yaml_file = "./test_results.yaml"

    def tearDown(self):
        os.remove(self.temp_yaml_file)

    def test_append_results_to_yaml(self):
        # Test data
        params = {
            "head_dim": 64,
            "num_heads_q": 8,
            "num_heads_kv": 8,
            "seqlens_q": [1024, 1024],
            "seqlens_kv": [1024, 1024],
            "dtype": "bfloat16",
            "dropout": False,
            "causal": True,
            "alibi": False,
            "local": False,
            "attn_mask": False,
            "deterministic": False,
            "is_training": True
        }
        
        fwd_results = [
            {"head_dim": 64, "tile_m": 64, "tile_n": 64, "num_waves": 4, "grid_type": 1, "balance_type": 0, "op_type": 0, "time_us": 100.5},
            {"head_dim": 64, "tile_m": 128, "tile_n": 64, "num_waves": 4, "grid_type": 1, "balance_type": 0, "op_type": 0, "time_us": 95.2}
        ]
        
        best_fwd_solution = {"head_dim": 64, "tile_m": 128, "tile_n": 64, "num_waves": 4, "grid_type": 1, "balance_type": 0, "op_type": 0, "time_us": 95.2}
        
        bwd_results = [
            {"head_dim": 64, "tile_m": 32, "tile_n": 64, "num_waves": 4, "grid_type": 1, "balance_type": 0, "op_type": 1, "time_us": 150.3},
            {"head_dim": 64, "tile_m": 64, "tile_n": 64, "num_waves": 4, "grid_type": 1, "balance_type": 0, "op_type": 1, "time_us": 140.1}
        ]
        
        best_bwd_solution = {"head_dim": 64, "tile_m": 64, "tile_n": 64, "num_waves": 4, "grid_type": 1, "balance_type": 0, "op_type": 1, "time_us": 140.1}

        # Call the function
        append_results_to_yaml(params, fwd_results, best_fwd_solution, bwd_results, best_bwd_solution, self.temp_yaml_file)

        # Read the YAML file and check its contents
        with open(self.temp_yaml_file, 'r') as f:
            data = yaml.safe_load(f)

        self.assertEqual(len(data), 1)
        result = data[0]

        self.assertEqual(result['head_dim'], 64)
        self.assertEqual(result['num_heads_q'], 8)
        self.assertEqual(result['num_heads_kv'], 8)
        self.assertEqual(result['seqlens_q'], [1024, 1024])
        self.assertEqual(result['seqlens_kv'], [1024, 1024])
        self.assertEqual(result['dtype'], 'bfloat16')
        self.assertFalse(result['dropout'])
        self.assertTrue(result['causal'])
        self.assertFalse(result['alibi'])
        self.assertFalse(result['local'])
        self.assertFalse(result['attn_mask'])
        self.assertFalse(result['deterministic'])
        self.assertTrue(result['is_training'])

        self.assertEqual(len(result['fwd_results']), 2)
        self.assertEqual(result['fwd_results'][0], str(list(fwd_results[0].values())))
        self.assertEqual(result['fwd_results'][1], str(list(fwd_results[1].values())))

        self.assertEqual(result['best_fwd_solution'], str(list(best_fwd_solution.values())))

        self.assertEqual(len(result['bwd_results']), 2)
        self.assertEqual(result['bwd_results'][0], str(list(bwd_results[0].values())))
        self.assertEqual(result['bwd_results'][1], str(list(bwd_results[1].values())))

        self.assertEqual(result['best_bwd_solution'], str(list(best_bwd_solution.values())))

if __name__ == '__main__':
    unittest.main()

