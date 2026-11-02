from __future__ import annotations

import ast
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import server


ROOT = Path(__file__).resolve().parents[2]


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
        "parameters": {
            "n_init_configs": 1,
            "n_low_fidelity_seed": 1,
            "n_top_blend": 1,
            "n_high_fidelity_seed": 1,
            "seed": 1234,
            "split_seed": 1234,
            "split": "scaffold",
        },
    }


class AdapterContractTests(unittest.TestCase):
    def test_normalizes_valid_request_and_parameters(self):
        normalized = server.normalize_request(valid_payload())
        self.assertEqual(normalized["task"], server.TASK_ID)
        self.assertEqual(normalized["dataset"], "demo")
        self.assertEqual(normalized["parameters"]["n_init_configs"], 1)
        self.assertEqual(normalized["parameters"]["split"], "scaffold")
        self.assertEqual(normalized["submission_id"], "11111111-1111-4111-8111-111111111111")

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

    def test_rejects_invalid_action_provenance(self):
        payload = valid_payload()
        payload["action_envelope_hash"] = "bad"
        with self.assertRaises(server.RequestError):
            server.normalize_request(payload)
        payload = valid_payload()
        payload["submission_id"] = "not-a-uuid"
        with self.assertRaises(server.RequestError):
            server.normalize_request(payload)

    def test_rejects_invalid_split(self):
        payload = valid_payload()
        payload["parameters"]["split"] = "random-but-unregistered"
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
        self.assertEqual(command[command.index("--split") + 1], "scaffold")
        self.assertNotIsInstance(command, str)

    def test_artifact_manifest_is_hashed_and_requires_complete_pipeline(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "best_save.ckpt").write_bytes(b"checkpoint")
            (output / "inf_ckpt_selected.csv").write_text("id\nmodel-1\n", encoding="utf-8")
            (output / "inference_log.txt").write_text("inference done\n", encoding="utf-8")
            (output / "search_result.csv").write_text("id,metric\nmodel-1,0.9\n", encoding="utf-8")
            artifacts = server._artifact_records(output)
            self.assertEqual(len(artifacts), 4)
            checkpoint = next(item for item in artifacts if item["path"] == "best_save.ckpt")
            self.assertEqual(checkpoint["sha256"], hashlib.sha256(b"checkpoint").hexdigest())

    def test_artifact_manifest_fails_without_high_fidelity_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "inf_ckpt_selected.csv").write_text("id\nmodel-1\n", encoding="utf-8")
            (output / "inference_log.txt").write_text("inference done\n", encoding="utf-8")
            (output / "search_result.csv").write_text("id,metric\nmodel-1,0.9\n", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                server._artifact_records(output)

    def test_pipeline_process_invocations_disable_shell(self):
        for relative_path in ("src_1gp/glam.py", "src_1gp/trainer.py"):
            source = (ROOT / relative_path).read_text(encoding="utf-8")
            tree = ast.parse(source, filename=relative_path)
            calls = [
                node for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "Popen"
            ]
            self.assertTrue(calls, f"No subprocess.Popen calls found in {relative_path}")
            for call in calls:
                keywords = {item.arg: item.value for item in call.keywords}
                self.assertIn("shell", keywords, f"shell flag missing in {relative_path}")
                self.assertIsInstance(keywords["shell"], ast.Constant)
                self.assertIs(keywords["shell"].value, False)

    def test_pipeline_uses_seeded_split_and_high_fidelity_gate(self):
        utils = (ROOT / "src_1gp" / "utils.py").read_text(encoding="utf-8")
        dataset = (ROOT / "src_1gp" / "dataset.py").read_text(encoding="utf-8")
        trainer = (ROOT / "src_1gp" / "trainer.py").read_text(encoding="utf-8")
        self.assertIn("np.random.default_rng(seed)", utils)
        self.assertIn("np.random.default_rng(int(self.split_seed))", dataset)
        self.assertIn("more_epochs_run_", trainer)
        self.assertIn("high-fidelity checkpoints", trainer)

    def test_response_contract_never_promotes_science_or_evidence(self):
        state = {
            "status": "SUCCEEDED",
            "scientific_validation_status": "NOT_PERFORMED",
            "evidence_status": "NOT_CREATED",
        }
        self.assertEqual(state["scientific_validation_status"], "NOT_PERFORMED")
        self.assertEqual(state["evidence_status"], "NOT_CREATED")


if __name__ == "__main__":
    unittest.main()
