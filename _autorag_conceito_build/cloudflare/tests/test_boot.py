import subprocess
import unittest
from unittest.mock import patch

from cloudflare.boot import start_ollama, stop_processes


class Process:
    def __init__(self, slow=False):
        self.returncode = None
        self.slow = slow
        self.terminated = False
        self.killed = False
    def poll(self):
        return self.returncode
    def terminate(self):
        self.terminated = True
    def wait(self, timeout=None):
        if self.slow and not self.killed:
            raise subprocess.TimeoutExpired("ollama", timeout)
        self.returncode = 0
        return 0
    def kill(self):
        self.killed = True


class BootTests(unittest.TestCase):
    def test_ollama_readiness_requires_audited_model_digests(self):
        process = Process()
        lock = {"models": [{"name": "bge-m3", "manifest_sha256": "abc"}]}
        with patch("cloudflare.boot.verify_installation", return_value=lock), patch("cloudflare.boot.subprocess.Popen", return_value=process) as popen, patch("cloudflare.boot.local_json", return_value={"models": [{"name": "bge-m3:latest", "digest": "abc"}]}):
            self.assertIs(start_ollama(), process)
            environment = popen.call_args.kwargs["env"]
            self.assertEqual(environment["OLLAMA_HOST"], "127.0.0.1:11434")
            self.assertEqual(environment["OLLAMA_NO_CLOUD"], "1")

    def test_wrong_model_digest_stops_startup(self):
        process = Process()
        lock = {"models": [{"name": "bge-m3", "manifest_sha256": "abc"}]}
        with patch("cloudflare.boot.verify_installation", return_value=lock), patch("cloudflare.boot.subprocess.Popen", return_value=process), patch("cloudflare.boot.local_json", return_value={"models": [{"name": "bge-m3:latest", "digest": "changed"}]}):
            with self.assertRaises(RuntimeError):
                start_ollama()
        self.assertTrue(process.terminated)

    def test_supervisor_terminates_and_reaps_unresponsive_children(self):
        fast, slow = Process(), Process(slow=True)
        stop_processes([fast, slow], timeout=0)
        self.assertTrue(fast.terminated)
        self.assertTrue(slow.terminated)
        self.assertTrue(slow.killed)
        self.assertEqual(slow.returncode, 0)


if __name__ == "__main__":
    unittest.main()
