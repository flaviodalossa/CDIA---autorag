#!/usr/bin/env python3
"""Docker CLI proxy: replace only managed-registry pushes with chunked uploads.

Wrangler still performs its own Docker login. regctl reads those credentials
from Docker's configuration; this module neither reads nor prints that file.
All other Docker commands preserve their arguments and standard input/output.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


REGISTRY = "registry.cloudflare.com"
CHUNK_BYTES = 8 * 1024 * 1024
RESERVE_BYTES = 64 * 1024 * 1024


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
    run_step([binary, "config", "set", "--docker-cred", "--blob-limit", str(CHUNK_BYTES * 2)],
             environment, "uploader configuration")
    run_step([binary, "registry", "set", REGISTRY, "--blob-max", "1",
              "--blob-chunk", str(CHUNK_BYTES), "--req-concurrent", "1", "--skip-check"],
             environment, "registry configuration")
    print("Uploading image layers in 8 MiB chunks using the existing Docker login.", flush=True)
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
