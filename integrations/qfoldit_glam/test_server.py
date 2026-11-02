import sys
import tempfile
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
        "submission_id": "11111111-1111-4111-8111-111111111111",
        "action_envelope_hash": "sha256:" + "a" * 64,
        "dataset": "demo",
        "dataset_revision": "bundled-demo",
        "split": "scaffold",
        "parameters": {
            "n_init_configs": 1,
            "n_low_fidelity_seed": 1,
            "n_top_blend": 1,
            "n_high_fidelity_seed": 1,
            "seed": 1234,
            "split_seed": 1234,
            "search_seed": 4321,
        },
    }


class AdapterContractTests(unittest.TestCase):
    def test_normalizes_valid_request_and_parameters(self):
        normalized = server.normalize_request(valid_payload())
        self.assertEqual(normalized["task"], "adaptive-property-search")
        self.assertEqual(normalized["dataset"], "demo")
        self.assertEqual(normalized["submission_id"], "11111111-1111-4111-8111-111111111111")
        self.assertEqual(normalized["split"], "scaffold")
        self.assertEqual(normalized["parameters"]["search_seed"], 4321)

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

    def test_rejects_missing_or_invalid_provenance(self):
        payload = valid_payload()
        del payload["submission_id"]
        with self.assertRaises(server.RequestError):
            server.normalize_request(payload)
        payload = valid_payload()
        payload["action_envelope_hash"] = "not-a-hash"
        with self.assertRaises(server.RequestError):
            server.normalize_request(payload)

    def test_rejects_wrong_contract_task_and_split(self):
        payload = valid_payload()
        payload["contract"] = "untrusted/0.0"
        with self.assertRaises(server.RequestError):
            server.normalize_request(payload)
        payload = valid_payload()
        payload["task"] = "arbitrary-command"
        with self.assertRaises(server.RequestError):
            server.normalize_request(payload)
        payload = valid_payload()
        payload["split"] = "caller-controlled-path"
        with self.assertRaises(server.RequestError):
            server.normalize_request(payload)

    def test_command_uses_fixed_entrypoint_and_passes_reproducibility(self):
        request = server.normalize_request(valid_payload())
        root = Path("/tmp/glam dataset root")
        command = server.build_command(request, root, python_executable="python-test")
        self.assertEqual(command[0], "python-test")
        self.assertEqual(command[1], "glam.py")
        self.assertEqual(command[command.index("--dataset_root") + 1], str(root))
        self.assertEqual(command[command.index("--split") + 1], "scaffold")
        self.assertEqual(command[command.index("--search_seed") + 1], "4321")
        self.assertNotIsInstance(command, str)

    def test_manifest_hashes_dataset_and_actual_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            job_dir = Path(temporary)
            output = job_dir / "outputs" / "log_demo" / "seed_1"
            output.mkdir(parents=True)
            checkpoint = output / "best_save.ckpt"
            checkpoint.write_bytes(b"test-checkpoint")
            state = {
                "job_id": "11111111-1111-4111-8111-111111111111",
                "mission_id": "mission-1",
                "mission_version": "1.0.0",
                "submission_id": "11111111-1111-4111-8111-111111111111",
                "action_envelope_hash": "sha256:" + "a" * 64,
                "tenant_id": None,
                "task": server.TASK_ID,
                "dataset": "demo",
                "dataset_revision": "fixture-v1",
                "dataset_sha256": "b" * 64,
                "split": "scaffold",
                "parameters": {"seed": 1234},
            }
            manifest = server.build_run_manifest(state, job_dir)
            self.assertEqual(manifest["contract"], "qfoldit.glam.run-result/1.0")
            self.assertEqual(manifest["artifact_count"], 1)
            self.assertEqual(manifest["artifacts"][0]["sha256"], server.sha256_file(checkpoint))
            self.assertEqual(manifest["scientific_validation_status"], "NOT_PERFORMED")
            self.assertTrue((job_dir / "run_manifest.json").is_file())

    def test_manifest_fails_closed_without_best_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            job_dir = Path(temporary)
            (job_dir / "outputs").mkdir()
            state = {
                "job_id": "11111111-1111-4111-8111-111111111111",
                "mission_id": "mission-1",
                "mission_version": "1.0.0",
                "submission_id": "11111111-1111-4111-8111-111111111111",
                "action_envelope_hash": "sha256:" + "a" * 64,
                "tenant_id": None,
                "task": server.TASK_ID,
                "dataset": "demo",
                "dataset_revision": "fixture-v1",
                "dataset_sha256": "b" * 64,
                "split": "scaffold",
                "parameters": {"seed": 1234},
            }
            with self.assertRaises(RuntimeError):
                server.build_run_manifest(state, job_dir)


if __name__ == "__main__":
    unittest.main()
