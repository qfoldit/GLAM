import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import server


def valid_payload():
    return {
        "contract": server.REQUEST_CONTRACT,
        "task": server.TASK_ID,
        "mission_id": "property-benchmark-001",
        "mission_version": "1.0.0",
        "dataset": "demo",
        "dataset_revision": "bundled-demo",
        "parameters": {
            "n_init_configs": 1,
            "n_low_fidelity_seed": 1,
            "n_top_blend": 1,
            "n_high_fidelity_seed": 1,
            "seed": 1234,
            "split_seed": 1234,
        },
    }


class AdapterContractTests(unittest.TestCase):
    def test_normalizes_valid_request_and_parameters(self):
        normalized = server.normalize_request(valid_payload())
        self.assertEqual(normalized["task"], "adaptive-property-search")
        self.assertEqual(normalized["dataset"], "demo")
        self.assertEqual(normalized["parameters"]["n_init_configs"], 1)

    def test_rejects_unknown_dataset(self):
        payload = valid_payload()
        payload["dataset"] = "unknown"
        with self.assertRaises(server.RequestError):
            server.normalize_request(payload)

    def test_rejects_unknown_top_level_fields(self):
        payload = valid_payload()
        payload["shell_command"] = "echo unsafe"
        with self.assertRaises(server.RequestError):
            server.normalize_request(payload)

    def test_rejects_unknown_parameters(self):
        payload = valid_payload()
        payload["parameters"]["learning_rate"] = 0.001
        with self.assertRaises(server.RequestError):
            server.normalize_request(payload)

    def test_rejects_unbounded_job_size(self):
        payload = valid_payload()
        payload["parameters"]["n_init_configs"] = 21
        with self.assertRaises(server.RequestError):
            server.normalize_request(payload)

    def test_rejects_boolean_where_integer_is_required(self):
        payload = valid_payload()
        payload["parameters"]["seed"] = True
        with self.assertRaises(server.RequestError):
            server.normalize_request(payload)

    def test_rejects_wrong_contract_and_task(self):
        payload = valid_payload()
        payload["contract"] = "untrusted/0.0"
        with self.assertRaises(server.RequestError):
            server.normalize_request(payload)
        payload = valid_payload()
        payload["task"] = "arbitrary-command"
        with self.assertRaises(server.RequestError):
            server.normalize_request(payload)

    def test_command_uses_argument_list_and_fixed_entrypoint(self):
        request = server.normalize_request(valid_payload())
        root = Path("/tmp/glam dataset root")
        command = server.build_command(request, root, python_executable="python-test")
        self.assertEqual(command[0], "python-test")
        self.assertEqual(command[1], "glam.py")
        self.assertIn("--dataset_root", command)
        self.assertEqual(command[command.index("--dataset_root") + 1], str(root))
        self.assertNotIsInstance(command, str)
        self.assertEqual(command[command.index("--n_init_configs") + 1], "1")


if __name__ == "__main__":
    unittest.main()
