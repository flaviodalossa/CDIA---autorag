import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

from cloudflare import deploy, docker_push

ACCOUNT = "a" * 32
IMAGE = f"registry.cloudflare.com/{ACCOUNT}/image:tag"
PASSWORD = "synthetic-registry-password-never-log"


def credentials(**changes):
    return json.dumps({"account_id": ACCOUNT, "registry_host": docker_push.REGISTRY,
                       "username": "synthetic-user", "password": PASSWORD, **changes})


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
            if arguments[1:5] == ["pnpm", "exec", "wrangler", "containers"]:
                return subprocess.CompletedProcess(arguments, 0, stdout=credentials())
            return subprocess.CompletedProcess(arguments, 0)
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(os.environ, {"DOCKER_CONFIG": "private-config", "CLOUDFLARE_ACCOUNT_ID": ACCOUNT}), \
                patch.object(docker_push.shutil, "which", return_value="corepack"), \
                patch.object(docker_push.subprocess, "run", side_effect=run), \
                patch.object(docker_push.shutil, "disk_usage", return_value=type("Disk", (), {"free": 10**10})()):
            docker_push.push_image("docker-real", "regctl-verified", IMAGE, directory)
        commands = [args for args, _ in invocations]
        self.assertEqual(commands[1][1:4], ["image", "save", "--output"])
        self.assertEqual(commands[2][1:8], ["pnpm", "exec", "wrangler", "containers", "registries", "credentials", docker_push.REGISTRY])
        self.assertEqual(commands[2][-5:], ["--push", "--pull", "--expiration-minutes", "60", "--json"])
        self.assertEqual(commands[3][1:3], ["login", "--password-stdin"])
        self.assertEqual(invocations[3][1]["input"], PASSWORD + "\n")
        self.assertTrue(invocations[2][1]["capture_output"])
        self.assertTrue(invocations[3][1]["capture_output"])
        self.assertNotIn(PASSWORD, str(commands))
        settings = commands[5]
        self.assertIn("--blob-max", settings)
        self.assertEqual(settings[settings.index("--blob-max") + 1], "1")
        self.assertEqual(settings[settings.index("--blob-chunk") + 1], str(16 * 1024 * 1024))
        self.assertEqual(commands[4][-1], str(32 * 1024 * 1024))
        self.assertEqual(settings[settings.index("--req-concurrent") + 1], "1")
        self.assertEqual(commands[-1][1:4], ["image", "import", IMAGE])
        self.assertTrue(commands[-1][-1].endswith("image.tar"))
        for _, options in invocations:
            self.assertEqual(options["env"]["DOCKER_CONFIG"], "private-config")
            self.assertNotIn("shell", options)

    def test_insufficient_disk_prevents_export_or_upload(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(os.environ, {"CLOUDFLARE_ACCOUNT_ID": ACCOUNT}), \
                patch.object(docker_push.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, stdout="100000\n")) as run, \
                patch.object(docker_push.shutil, "disk_usage", return_value=type("Disk", (), {"free": 0})()):
            with self.assertRaisesRegex(docker_push.PushError, "disk space") as failure:
                docker_push.push_image("docker", "regctl", IMAGE, directory)
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
                patch.dict(os.environ, {"CLOUDFLARE_ACCOUNT_ID": ACCOUNT}), \
                patch.object(docker_push, "refresh_registry_login") as refresh, \
                patch.object(docker_push.subprocess, 'run', side_effect=run), \
                patch.object(docker_push.shutil, 'disk_usage', side_effect=[
                    type('Disk', (), {'free': 0})(), type('Disk', (), {'free': 10**10})()]):
            docker_push.push_image('docker', 'regctl', IMAGE, directory)
        refresh.assert_called_once()
        self.assertEqual(commands[1], ['docker', 'builder', 'prune', '--all', '--force'])
        self.assertEqual(commands[2][1:4], ['image', 'save', '--output'])
        self.assertEqual(commands[-1][1:4], ['image', 'import', IMAGE])
        self.assertFalse(any(command[1:3] in [['system', 'prune'], ['image', 'prune'], ['volume', 'prune']] for command in commands))

    def test_failed_cache_cleanup_prevents_export_or_upload(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(os.environ, {"CLOUDFLARE_ACCOUNT_ID": ACCOUNT}), \
                patch.object(docker_push.subprocess, 'run', side_effect=[
                    subprocess.CompletedProcess([], 0, stdout='1024\n'),
                    subprocess.CompletedProcess([], 2)]) as run, \
                patch.object(docker_push.shutil, 'disk_usage', return_value=type('Disk', (), {'free': 0})()):
            with self.assertRaisesRegex(docker_push.PushError, 'build-cache cleanup.*2'):
                docker_push.push_image('docker', 'regctl', IMAGE, directory)
        self.assertEqual(run.call_count, 2)

    def test_uploader_failure_never_reports_success(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(os.environ, {"CLOUDFLARE_ACCOUNT_ID": ACCOUNT}), \
                patch.object(docker_push.subprocess, "run", side_effect=[
                    subprocess.CompletedProcess([], 0, stdout="1024\n"),
                    subprocess.CompletedProcess([], 3)]), \
                patch.object(docker_push.shutil, "disk_usage", return_value=type("Disk", (), {"free": 10**10})()):
            with self.assertRaisesRegex(docker_push.PushError, "image export.*3"):
                docker_push.push_image("docker", "regctl", IMAGE, directory)

    def test_refreshed_login_never_logs_credential_output(self):
        output = io.StringIO()
        with patch.object(docker_push.shutil, "which", return_value="corepack"), \
                patch.object(docker_push.subprocess, "run", side_effect=[
                    subprocess.CompletedProcess([], 0, stdout=credentials(), stderr=PASSWORD),
                    subprocess.CompletedProcess([], 0, stdout=PASSWORD, stderr=PASSWORD),
                ]) as run, redirect_stdout(output), redirect_stderr(output):
            environment = {"DOCKER_CONFIG": "private-config"}
            docker_push.refresh_registry_login("docker", IMAGE, environment)
        self.assertNotIn(PASSWORD, output.getvalue())
        self.assertEqual(environment["CLOUDFLARE_ACCOUNT_ID"], ACCOUNT)
        self.assertEqual(run.call_args.kwargs["input"], PASSWORD + "\n")
        self.assertNotIn(PASSWORD, str(run.call_args.args))
        self.assertNotIn("shell", run.call_args.kwargs)

    def test_invalid_credential_responses_fail_closed_without_logging(self):
        cases = [
            subprocess.CompletedProcess([], 1, stdout=credentials(), stderr=PASSWORD),
            subprocess.CompletedProcess([], 0, stdout=PASSWORD),
            subprocess.CompletedProcess([], 0, stdout=credentials(account_id="b" * 32)),
            subprocess.CompletedProcess([], 0, stdout=credentials(registry_host="evil.example")),
            subprocess.CompletedProcess([], 0, stdout=credentials(username="")),
            subprocess.CompletedProcess([], 0, stdout=credentials(password="")),
            subprocess.CompletedProcess([], 0, stdout=credentials(password=PASSWORD + "\n")),
            subprocess.CompletedProcess([], 0, stdout="[]"),
            subprocess.TimeoutExpired(["corepack"], 120, output=PASSWORD, stderr=PASSWORD),
            OSError(PASSWORD),
        ]
        for result in cases:
            with self.subTest(response=type(result).__name__), \
                    patch.object(docker_push.shutil, "which", return_value="corepack"), \
                    patch.object(docker_push.subprocess, "run", side_effect=[result]) as run:
                output = io.StringIO()
                with redirect_stdout(output), redirect_stderr(output), \
                        self.assertRaises(docker_push.PushError) as failure:
                    docker_push.refresh_registry_login("docker", IMAGE, {})
                self.assertEqual(run.call_count, 1)
                self.assertNotIn(PASSWORD, str(failure.exception) + output.getvalue())

    def test_login_failure_is_opaque_and_prevents_upload(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(os.environ, {"CLOUDFLARE_ACCOUNT_ID": ACCOUNT}), \
                patch.object(docker_push.shutil, "which", return_value="corepack"), \
                patch.object(docker_push.shutil, "disk_usage", return_value=type("Disk", (), {"free": 10**10})()), \
                patch.object(docker_push.subprocess, "run", side_effect=[
                    subprocess.CompletedProcess([], 0, stdout="1024\n"),
                    subprocess.CompletedProcess([], 0),
                    subprocess.CompletedProcess([], 0, stdout=credentials()),
                    subprocess.CompletedProcess([], 1, stdout=PASSWORD, stderr=PASSWORD),
                ]) as run:
            output = io.StringIO()
            with redirect_stdout(output), redirect_stderr(output), \
                    self.assertRaisesRegex(docker_push.PushError, "login failed") as failure:
                docker_push.push_image("docker", "regctl", IMAGE, directory)
        self.assertEqual(run.call_count, 4)
        self.assertNotIn(PASSWORD, str(failure.exception) + output.getvalue())
        self.assertFalse(any(call.args[0][1:3] == ["image", "import"] for call in run.call_args_list))

    def test_wrong_registry_or_account_is_rejected_without_credentials(self):
        for target, environment in [
            ("evil.example/" + ACCOUNT + "/image:tag", {}),
            ("registry.cloudflare.com/not-an-account/image:tag", {}),
            (IMAGE, {"CLOUDFLARE_ACCOUNT_ID": "b" * 32}),
        ]:
            with self.subTest(target=target), patch.object(docker_push.subprocess, "run") as run:
                with self.assertRaises(docker_push.PushError):
                    docker_push.refresh_registry_login("docker", target, environment)
                run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
