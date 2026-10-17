"""Deploy through Wrangler with a checksum-pinned, chunked registry uploader.

Run from Workers Builds with ``python3 deploy.py``. This changes only the build
host's push transport; the container image and its runtime files are unchanged.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.request


REGCTL_VERSION = "0.11.6"
REGCTL_URL = "https://github.com/regclient/regclient/releases/download/v0.11.6/regctl-linux-amd64"
REGCTL_SHA256 = "8e0e62a497fcdb8048d18aa927a139613176ba0531f412bc541044e28f9856bd"
MAX_BINARY_BYTES = 32 * 1024 * 1024
ROOT = Path(__file__).resolve().parent


class DeployError(RuntimeError):
    pass


def install_regctl(destination, opener=None):
    """Install the official Linux AMD64 binary only after verifying its SHA256."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".download")
    opener = opener or urllib.request.urlopen
    request = urllib.request.Request(REGCTL_URL, headers={"User-Agent": "AutoRAG-Cloudflare-deploy/1"})
    digest = hashlib.sha256()
    size = 0
    try:
        with opener(request, timeout=120) as response, temporary.open("wb") as output:
            while block := response.read(1024 * 1024):
                size += len(block)
                if size > MAX_BINARY_BYTES:
                    raise DeployError("The registry uploader exceeds the allowed binary size.")
                output.write(block)
                digest.update(block)
        if digest.hexdigest() != REGCTL_SHA256:
            raise DeployError("The official registry uploader checksum does not match its pinned release.")
        temporary.chmod(0o700)
        os.replace(temporary, destination)
        return destination
    finally:
        temporary.unlink(missing_ok=True)


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if sys.platform != "linux" or platform.machine().lower() not in {"x86_64", "amd64"}:
        raise DeployError("This deployment entry point requires a Linux AMD64 build host.")
    docker = shutil.which("docker")
    corepack = shutil.which("corepack")
    if not docker or not corepack:
        raise DeployError("Docker and Corepack must be available on the build host.")
    wrapper = ROOT / "docker_push.py"
    wrapper.chmod(wrapper.stat().st_mode | 0o111)
    with tempfile.TemporaryDirectory(prefix="autorag-regctl-") as temporary:
        binary = install_regctl(Path(temporary) / "regctl")
        environment = os.environ.copy()
        environment.update({
            "AUTORAG_REAL_DOCKER_BIN": str(Path(docker).resolve()),
            "AUTORAG_REGCTL_BIN": str(binary.resolve()),
            "WRANGLER_DOCKER_BIN": str(wrapper.resolve()),
        })
        print(f"Deploying with verified regctl v{REGCTL_VERSION}; credentials remain in Docker configuration.", flush=True)
        result = subprocess.run(
            [corepack, "pnpm", "exec", "wrangler", "deploy", *args],
            cwd=ROOT, env=environment, check=False,
        )
        return result.returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        # Avoid serializing environment, request URLs, or credential-related errors.
        print("Cloudflare deployment bootstrap failed. Check the build tools and pinned uploader download.", file=sys.stderr)
        raise SystemExit(1)
