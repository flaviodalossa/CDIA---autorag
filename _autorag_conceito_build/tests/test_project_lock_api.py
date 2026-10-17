"""Concurrent API reads wait briefly without weakening mutation exclusion."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import api
from autorag.core import Project, ProjectBusyError


class ProjectLockApiTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.project = Project(Path(self.temporary.name) / "project")
        self.environment = patch.dict(os.environ, {"AUTORAG_API_TOKEN": "test-token", "AUTORAG_ENV": "test"})
        self.environment.start()
        self.project_patch = patch.object(api, "project", self.project)
        self.project_patch.start()
        self.client = TestClient(api.app, headers={"Authorization": "Bearer test-token"}, raise_server_exceptions=False)
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.project_patch.stop()
        self.environment.stop()
        self.temporary.cleanup()

    def test_parallel_status_and_models_wait_instead_of_returning_non_json_500(self):
        status_entered = threading.Event()
        models_waiting = threading.Event()
        release_status = threading.Event()
        original_lock = self.project.lock

        @contextmanager
        def observed_lock(wait_seconds=0.0):
            if status_entered.is_set() and not release_status.is_set():
                models_waiting.set()
            with original_lock(wait_seconds=wait_seconds):
                yield

        def held_status(project):
            status_entered.set()
            if not release_status.wait(5):
                raise AssertionError("The test did not release the status read.")
            return {"documents": 0, "topics": 0, "profiles": []}

        with patch.object(self.project, "lock", observed_lock), patch.object(api, "status", held_status), patch.object(api, "available_models", return_value=[]):
            with ThreadPoolExecutor(max_workers=2) as pool:
                status_future = pool.submit(self.client.get, "/api/status")
                try:
                    self.assertTrue(status_entered.wait(3))
                    models_future = pool.submit(self.client.get, "/api/models")
                    self.assertTrue(models_waiting.wait(3))
                    self.assertFalse(models_future.done())
                finally:
                    release_status.set()
                for response in (status_future.result(timeout=5), models_future.result(timeout=5)):
                    self.assertEqual(response.status_code, 200)
                    self.assertIn("application/json", response.headers["content-type"])
                    self.assertIsInstance(response.json(), dict)

    def test_read_timeout_is_json_409_and_does_not_remove_another_lock(self):
        with self.project.lock(), patch.object(api, "READ_LOCK_WAIT_SECONDS", 0.03), patch.object(api, "available_models", return_value=[]):
            for path in ("/api/status", "/api/models", "/api/base/export"):
                with self.subTest(path=path):
                    response = self.client.get(path)
                    self.assertEqual(response.status_code, 409)
                    self.assertIn("application/json", response.headers["content-type"])
                    self.assertEqual(response.headers["retry-after"], "1")
                    self.assertIn("Aguarde", response.json()["detail"])
                    self.assertNotIn(".escrita.lock", response.json()["detail"])
                    self.assertTrue((self.project.data / ".escrita.lock").is_file())
        self.assertFalse((self.project.data / ".escrita.lock").exists())

    def test_default_mutation_lock_remains_fail_fast_across_project_instances(self):
        other = Project(self.project.root)
        with self.project.lock(), patch("autorag.core.time.sleep") as sleep:
            with self.assertRaises(ProjectBusyError):
                with other.lock():
                    self.fail("A second mutation acquired an occupied project lock.")
            sleep.assert_not_called()
            self.assertTrue((self.project.data / ".escrita.lock").is_file())

    def test_catalog_discovery_does_not_hold_the_project_lock(self):
        def discover(*args, **kwargs):
            with self.project.lock():
                pass
            return []
        with patch.object(api, "available_models", side_effect=discover) as discovery:
            response = self.client.get("/api/models")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["models"], [])
        discovery.assert_called_once()

    def test_unexpected_runtime_error_is_not_reclassified_as_project_busy(self):
        with patch.object(api, "status", side_effect=RuntimeError("unrelated failure")):
            response = self.client.get("/api/status")
        self.assertEqual(response.status_code, 500)

    def test_wait_budget_rejects_nonfinite_negative_and_non_numeric_values(self):
        for wait_seconds in (float("nan"), float("inf"), -1, "2", True):
            with self.subTest(wait_seconds=wait_seconds), self.assertRaises(ValueError):
                with self.project.lock(wait_seconds=wait_seconds):
                    self.fail("An invalid lock wait budget was accepted.")
        self.assertFalse((self.project.data / ".escrita.lock").exists())


if __name__ == "__main__":
    unittest.main()
