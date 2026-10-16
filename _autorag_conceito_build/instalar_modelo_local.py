"""Download explícito do E5-small de 384 dimensões; inferência posterior é offline."""
from pathlib import Path
import os
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent/".cache"/"huggingface"))
from huggingface_hub import snapshot_download
from autorag.core import Project
from autorag.service import register_profile

if __name__ == "__main__":
    project = Project(Path(__file__).resolve().parent)
    model = "intfloat/multilingual-e5-small"
    destination = project.root/"modelos"/model.replace("/", "--")
    snapshot_download(model, local_dir=str(destination), allow_patterns=["*.json", "*.txt", "*.model", "*.safetensors", "*.bin", "*.md"], ignore_patterns=["onnx/*", "openvino/*"])
    profile = register_profile(project, "sentence_transformers", model, 384, models_dir=str(project.root/"modelos"))
    print("Modelo local instalado:", profile["perfil_id"])
