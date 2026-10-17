"""Extract only CPU files from a checksum-pinned official Linux Ollama release."""
from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request

from .download_models import load_lock, sha256_file


def cpu_member(name):
    path = PurePosixPath(name.removeprefix("./"))
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        return None
    if path.as_posix() == "bin/ollama":
        return path
    if len(path.parts) != 3 or path.parts[:2] != ("lib", "ollama"):
        return None
    lower = path.name.lower()
    if any(gpu in lower for gpu in ("cuda", "rocm", "vulkan", "mlx")):
        return None
    if path.name == "llama-server" or "license" in lower:
        return path
    if lower.startswith(("libggml", "libllama", "libmtmd", "libgomp", "libomp", "libstdc++", "libgcc")) and ".so" in lower:
        return path
    return None


def extract_cpu(archive, root):
    root = Path(root)
    links = []
    installed = set()
    for member in archive:
        relative = cpu_member(member.name)
        if relative is None or member.isdir():
            continue
        if relative.as_posix() in installed:
            raise ValueError("Duplicate CPU archive member.")
        destination = root.joinpath(*relative.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if member.isfile():
            with archive.extractfile(member) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)
            destination.chmod(0o755 if relative.name in {"ollama", "llama-server"} else 0o644)
        elif member.issym():
            target = PurePosixPath(member.linkname)
            if target.is_absolute() or len(target.parts) != 1 or cpu_member((relative.parent / target).as_posix()) is None:
                raise ValueError("Unsafe Ollama CPU library symlink.")
            links.append((destination, member.linkname))
        else:
            raise ValueError("Unsupported Ollama CPU archive member.")
        installed.add(relative.as_posix())
    for destination, target in links:
        destination.symlink_to(target)
    for destination, target in links:
        if not destination.is_file() or not destination.resolve().is_relative_to(root.resolve()):
            raise ValueError("CPU library symlink target is absent or unsafe.")
    required = {"bin/ollama", "lib/ollama/llama-server", "lib/ollama/libggml-base.so", "lib/ollama/libllama.so"}
    if not required.issubset(installed) or not any("libggml-cpu" in path for path in installed):
        raise ValueError("The pinned Ollama release is missing its CPU runner or libraries.")
    return sorted(installed)


def install(root):
    lock = load_lock()
    release = lock["ollama"]
    expected_url = "https://github.com/ollama/ollama/releases/download/v" + release["version"] + "/ollama-linux-amd64.tar.zst"
    if release["url"] != expected_url:
        raise ValueError("Only a pinned official Ollama release is accepted.")
    with tempfile.TemporaryDirectory(prefix="autorag-ollama-build-") as temporary:
        package = Path(temporary) / "ollama.tar.zst"
        request = urllib.request.Request(release["url"], headers={"User-Agent": "AutoRAG-Cloudflare-build/1"})
        with urllib.request.urlopen(request, timeout=120) as response, package.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        if sha256_file(package) != release["sha256"]:
            raise ValueError("Official Ollama release checksum mismatch.")
        process = subprocess.Popen(["zstd", "--decompress", "--stdout", str(package)], stdout=subprocess.PIPE)
        try:
            with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
                files = extract_cpu(archive, root)
            process.stdout.close()
            if process.wait() != 0:
                raise RuntimeError("Unable to decompress Ollama release.")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
        receipt = Path(root) / "lib" / "ollama" / "AUTORAG_INSTALL.txt"
        receipt.write_text("Official Ollama " + release["version"] + "\nArchive SHA256: " + release["sha256"] + "\nCPU files only:\n" + "\n".join(files) + "\n", encoding="utf-8")
        license_url = "https://raw.githubusercontent.com/ollama/ollama/v" + release["version"] + "/LICENSE"
        with urllib.request.urlopen(license_url, timeout=30) as response:
            license_text = response.read(65537)
        if len(license_text) > 65536 or b"MIT License" not in license_text:
            raise ValueError("The pinned Ollama runtime license is missing or unexpected.")
        (receipt.parent / "OLLAMA_LICENSE").write_bytes(license_text)
    print("Installed pinned Ollama CPU release:", release["version"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/opt/ollama-cpu")
    install(parser.parse_args().root)
