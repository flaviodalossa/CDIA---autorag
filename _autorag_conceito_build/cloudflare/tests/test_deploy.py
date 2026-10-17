import hashlib
import io
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from cloudflare import deploy, docker_push


class DeployTests(unittest.TestCase):
    def test_bootstrap_installs_only_a_matching_pinned_binary(self):
        content = b"synthetic-regctl-binary"
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "regctl"
            with patch.object(deploy, "REGCTL_SHA256", hashlib.sha256(content).hexdigest()):
                result = deploy.install_regctl(target, lambda request, **kwargs: io.BytesIO(content))
            self.assertEqual(result.read_bytes(), content)
            self.assertFalse(target.with_name("regctl.download").exists())
            with self.assertRaises(deploy.DeployError):
                deploy.install_regctl(Path(directory) / "wrong", lambda request, **kwargs: io.BytesIO(content))
            self.assertFalse((Path(directory) / "wrong").exists())
            self.assertFalse((Path(directory) / "wrong.download").exists())

    def test_bootstrap_sets_wrapper_and_inherits_docker_credentials_without_shell(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wrapper = root / "docker_push.py"
            wrapper.write_text("# synthetic wrapper\n", encoding="utf-8")
            binary = root / "regctl"
            binary.write_bytes(b"synthetic uploader")
            with patch.object(deploy, "ROOT", root), patch.object(deploy.sys, "platform", "linux"), \
                    patch.object(deploy.platform, "machine", return_value="x86_64"), \
                    patch.object(deploy.shutil, "which", side_effect=lambda name: str(root / name)), \
                    patch.object(deploy, "install_regctl", return_value=binary), \
                    patch.dict(os.environ, {"DOCKER_CONFIG": "docker-config-private", "OPENAI_API_KEY": "test-only-secret"}), \
                    patch.object(deploy.subprocess, "run", return_value=subprocess.CompletedProcess([], 7)) as run:
                self.assertEqual(deploy.main(["--dry-run"]), 7)
            command = run.call_args.args[0]
            options = run.call_args.kwargs
            self.assertEqual(command[1:], ["pnpm", "exec", "wrangler", "deploy", "--dry-run"])
            self.assertEqual(options["env"]["DOCKER_CONFIG"], "docker-config-private")
            self.assertEqual(options["env"]["WRANGLER_DOCKER_BIN"], str(wrapper.resolve()))
            self.assertNotIn("test-only-secret", " ".join(command))
            self.assertNotIn("shell", options)


class DockerPushTests(unittest.TestCase):
    def test_non_push_preserves_docker_arguments_and_stdin_via_execv(self):
        with patch.object(docker_push, "real_docker", return_value="docker-real"), \
                patch.object(docker_push.os, "execv") as execute, \
                patch.object(docker_push, "push_image") as push:
            self.assertEqual(docker_push.main(["login", "registry.cloudflare.com", "--password-stdin"]), 0)
        execute.assert_called_once_with("docker-real", ["docker-real", "login", "registry.cloudflare.com", "--password-stdin"])
        push.assert_not_called()

    def test_chunked_push_exports_seekable_tar_and_reuses_docker_config(self):
        invocations = []
        def run(arguments, **kwargs):
            invocations.append((arguments, kwargs))
            if arguments[1:3] == ["image", "inspect"]:
                return subprocess.CompletedProcess(arguments, 0, stdout="1024\n")
            return subprocess.CompletedProcess(arguments, 0)
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(os.environ, {"DOCKER_CONFIG": "private-config"}), \
                patch.object(docker_push.subprocess, "run", side_effect=run), \
                patch.object(docker_push.shutil, "disk_usage", return_value=type("Disk", (), {"free": 10**10})()):
            docker_push.push_image("docker-real", "regctl-verified", "registry.cloudflare.com/account/image:tag", directory)
        commands = [args for args, _ in invocations]
        self.assertEqual(commands[1][1:4], ["image", "save", "--output"])
        settings = commands[3]
        self.assertIn("--blob-max", settings)
        self.assertEqual(settings[settings.index("--blob-max") + 1], "1")
        self.assertEqual(settings[settings.index("--blob-chunk") + 1], str(8 * 1024 * 1024))
        self.assertEqual(settings[settings.index("--req-concurrent") + 1], "1")
        self.assertEqual(commands[-1][1:4], ["image", "import", "registry.cloudflare.com/account/image:tag"])
        self.assertTrue(commands[-1][-1].endswith("image.tar"))
        for _, options in invocations:
            self.assertEqual(options["env"]["DOCKER_CONFIG"], "private-config")
            self.assertNotIn("shell", options)

    def test_insufficient_disk_prevents_export_or_upload(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(docker_push.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, stdout="100000\n")) as run, \
                patch.object(docker_push.shutil, "disk_usage", return_value=type("Disk", (), {"free": 0})()):
            with self.assertRaisesRegex(docker_push.PushError, "disk space") as failure:
                docker_push.push_image("docker", "regctl", "registry.cloudflare.com/account/image:tag", directory)
        self.assertEqual(run.call_count, 2)
        self.assertEqual(run.call_args.args[0], ['docker', 'builder', 'prune', '--all', '--force'])
        self.assertIn('image_bytes=100000', str(failure.exception))
        self.assertIn('free_bytes=0', str(failure.exception))
        self.assertIn(f'required_bytes={100000 + docker_push.RESERVE_BYTES}', str(failure.exception))

    def test_unused_build_cache_is_reclaimed_only_under_disk_pressure(self):
        commands = []
        def run(arguments, **kwargs):
            commands.append(arguments)
            return subprocess.CompletedProcess(arguments, 0, stdout='1024\n')
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(docker_push.subprocess, 'run', side_effect=run), \
                patch.object(docker_push.shutil, 'disk_usage', side_effect=[
                    type('Disk', (), {'free': 0})(), type('Disk', (), {'free': 10**10})()]):
            docker_push.push_image('docker', 'regctl', 'registry.cloudflare.com/account/image:tag', directory)
        self.assertEqual(commands[1], ['docker', 'builder', 'prune', '--all', '--force'])
        self.assertEqual(commands[2][1:4], ['image', 'save', '--output'])
        self.assertEqual(commands[-1][1:4], ['image', 'import', 'registry.cloudflare.com/account/image:tag'])
        self.assertFalse(any(command[1:3] in [['system', 'prune'], ['image', 'prune'], ['volume', 'prune']] for command in commands))

    def test_failed_cache_cleanup_prevents_export_or_upload(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(docker_push.subprocess, 'run', side_effect=[
                    subprocess.CompletedProcess([], 0, stdout='1024\n'),
                    subprocess.CompletedProcess([], 2)]) as run, \
                patch.object(docker_push.shutil, 'disk_usage', return_value=type('Disk', (), {'free': 0})()):
            with self.assertRaisesRegex(docker_push.PushError, 'build-cache cleanup.*2'):
                docker_push.push_image('docker', 'regctl', 'registry.cloudflare.com/account/image:tag', directory)
        self.assertEqual(run.call_count, 2)

    def test_uploader_failure_never_reports_success(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(docker_push.subprocess, "run", side_effect=[
                    subprocess.CompletedProcess([], 0, stdout="1024\n"),
                    subprocess.CompletedProcess([], 3)]), \
                patch.object(docker_push.shutil, "disk_usage", return_value=type("Disk", (), {"free": 10**10})()):
            with self.assertRaisesRegex(docker_push.PushError, "image export.*3"):
                docker_push.push_image("docker", "regctl", "registry.cloudflare.com/account/image:tag", directory)


if __name__ == "__main__":
    unittest.main()
