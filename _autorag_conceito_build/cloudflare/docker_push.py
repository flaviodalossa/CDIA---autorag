#!/usr/bin/env python3
"""Docker CLI proxy: replace only managed-registry pushes with chunked uploads.

Wrangler still performs its own Docker login. regctl reads those credentials
from Docker's configuration; this module neither reads nor prints that file.
All other Docker commands preserve their arguments and standard input/output.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


REGISTRY = "registry.cloudflare.com"
CHUNK_BYTES = 16 * 1024 * 1024
RESERVE_BYTES = 64 * 1024 * 1024
PUSH_CREDENTIAL_MINUTES = 60
ROOT = Path(__file__).resolve().parent
ACCOUNT_ID = re.compile(r"[a-f0-9]{32}")


class PushError(RuntimeError):
    pass


def real_docker():
    candidate = os.getenv("AUTORAG_REAL_DOCKER_BIN") or shutil.which("docker")
    if not candidate or Path(candidate).resolve() == Path(__file__).resolve():
        raise PushError("The real Docker executable is not configured.")
    return str(Path(candidate).resolve())


def run_step(arguments, environment, stage):
    result = subprocess.run(arguments, env=environment, check=False)
    if result.returncode:
        raise PushError(f"The {stage} step failed with exit code {result.returncode}.")


def registry_account(image, environment):
    parts = image.split("/", 2)
    if (len(parts) != 3 or parts[0] != REGISTRY or not ACCOUNT_ID.fullmatch(parts[1])
            or not parts[2] or any(ord(character) <= 32 for character in image)):
        raise PushError("The upload target must use the managed registry and a valid account namespace.")
    account = parts[1]
    configured = environment.get("CLOUDFLARE_ACCOUNT_ID")
    if configured and configured != account:
        raise PushError("The upload target differs from the configured Cloudflare account.")
    return account


def refresh_registry_login(docker, image, environment):
    """Refresh the same push/pull permissions immediately before a long upload.

    Wrangler's initial registry login lasts 15 minutes. Image export consumes
    part of that window, and the chunked upload can exceed the remainder.
    Capture all credential-related output and pass the password only on stdin.
    """
    account = registry_account(image, environment)
    corepack = shutil.which("corepack")
    if not corepack:
        raise PushError("Corepack is required to refresh the temporary registry login.")
    environment["CLOUDFLARE_ACCOUNT_ID"] = account
    try:
        result = subprocess.run(
            [corepack, "pnpm", "exec", "wrangler", "containers", "registries", "credentials",
             REGISTRY, "--push", "--pull", "--expiration-minutes", str(PUSH_CREDENTIAL_MINUTES), "--json"],
            cwd=ROOT, env=environment, check=False, capture_output=True, text=True,
            encoding="utf-8", timeout=120,
        )
        if result.returncode or len(result.stdout) > 64 * 1024:
            raise ValueError
        credentials = json.loads(result.stdout)
        if (not isinstance(credentials, dict) or credentials.get("account_id") != account
                or credentials.get("registry_host") != REGISTRY):
            raise ValueError
        username, password = credentials.get("username"), credentials.get("password")
        if any(not isinstance(value, str) or not value or len(value) > 16 * 1024
               or any(ord(character) < 32 or ord(character) == 127 for character in value)
               for value in (username, password)):
            raise ValueError
    except (OSError, subprocess.SubprocessError, ValueError, TypeError, AttributeError):
        raise PushError("Unable to refresh the temporary registry credentials; upload was not attempted.") from None
    try:
        result = subprocess.run(
            [docker, "login", "--password-stdin", "--username", username, REGISTRY],
            input=password + "\n", env=environment, check=False, capture_output=True,
            text=True, encoding="utf-8", timeout=120,
        )
        if result.returncode:
            raise ValueError
    except (OSError, subprocess.SubprocessError, ValueError):
        raise PushError("The refreshed temporary registry login failed; upload was not attempted.") from None
    print(f"Refreshed temporary registry login for {PUSH_CREDENTIAL_MINUTES} minutes with push/pull permissions.", flush=True)


def ensure_archive_space(docker, environment, directory, image_bytes):
    required_bytes = image_bytes + RESERVE_BYTES
    free_bytes = shutil.disk_usage(directory).free
    print(f"Registry archive space: image_bytes={image_bytes} free_bytes={free_bytes} required_bytes={required_bytes}.", flush=True)
    if free_bytes >= required_bytes:
        return
    # BuildKit retains the intermediate model-assets and dependency stages.
    # Reclaim only unused build cache; the tagged final image, other images,
    # containers, volumes and the Docker credential configuration are retained.
    print("Reclaiming unused Docker build cache before exporting the tagged image.", flush=True)
    run_step([docker, "builder", "prune", "--all", "--force"], environment, "build-cache cleanup")
    free_bytes = shutil.disk_usage(directory).free
    print(f"Registry archive space after cache cleanup: image_bytes={image_bytes} free_bytes={free_bytes} required_bytes={required_bytes}.", flush=True)
    if free_bytes < required_bytes:
        raise PushError("The build host has insufficient temporary disk space for the image archive "
                        f"(image_bytes={image_bytes}, free_bytes={free_bytes}, required_bytes={required_bytes}).")


def push_image(docker, binary, image, directory):
    directory = Path(directory)
    environment = os.environ.copy()
    # Isolate uploader settings while retaining DOCKER_CONFIG and any credential
    # helper inherited from the successful Wrangler Docker login.
    environment["REGCTL_CONFIG"] = str(directory / "regctl.json")
    registry_account(image, environment)
    inspected = subprocess.run(
        [docker, "image", "inspect", "--format", "{{.Size}}", image],
        env=environment, check=False, capture_output=True, text=True,
    )
    try:
        image_bytes = int(inspected.stdout.strip())
    except (ValueError, TypeError):
        raise PushError("Unable to determine image archive space requirements.") from None
    if inspected.returncode or image_bytes < 1:
        raise PushError("Unable to inspect the image before its registry upload.")
    ensure_archive_space(docker, environment, directory, image_bytes)
    archive = directory / "image.tar"
    print("Preparing a temporary Docker archive for the chunked registry upload.", flush=True)
    run_step([docker, "image", "save", "--output", str(archive), image], environment, "image export")
    refresh_registry_login(docker, image, environment)
    run_step([binary, "config", "set", "--docker-cred", "--blob-limit", str(CHUNK_BYTES * 2)],
             environment, "uploader configuration")
    run_step([binary, "registry", "set", REGISTRY, "--blob-max", "1",
              "--blob-chunk", str(CHUNK_BYTES), "--req-concurrent", "1", "--skip-check"],
             environment, "registry configuration")
    print("Uploading image layers in 16 MiB chunks using the refreshed Docker login.", flush=True)
    # regctl import requires a seekable file; stdin/pipes are not supported.
    run_step([binary, "image", "import", image, str(archive)], environment, "chunked registry upload")


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    docker = real_docker()
    if not (len(args) == 2 and args[0] == "push" and args[1].startswith(REGISTRY + "/")):
        os.execv(docker, [docker, *args])
        return 0
    binary = os.getenv("AUTORAG_REGCTL_BIN", "")
    if not binary or not Path(binary).is_file():
        raise PushError("Run the Cloudflare deployment through deploy.py to install the verified uploader.")
    with tempfile.TemporaryDirectory(prefix="autorag-registry-push-") as temporary:
        push_image(docker, str(Path(binary).resolve()), args[1], temporary)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PushError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)
    except Exception:
        print("The managed-registry upload failed. No deployment success was reported.", file=sys.stderr)
        raise SystemExit(1)
