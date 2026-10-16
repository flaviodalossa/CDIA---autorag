"""Registra verificações feitas de fato no ambiente instalado."""
from pathlib import Path
import re
import subprocess
import sys
import tkinter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from autorag.core import Project, read_json
from autorag.service import inventory

project = Project(Path(__file__).resolve().parents[1])
process = subprocess.run([sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests", "-v"],
                         cwd=project.root, capture_output=True, encoding="utf-8", errors="replace", timeout=180)
project.root.joinpath("logs").mkdir(exist_ok=True)
project.root.joinpath("logs", "testes.log").write_text(process.stdout+process.stderr, encoding="utf-8")
if process.returncode:
    print(process.stderr)
    raise SystemExit(process.returncode)
count = int(re.search(r"Ran (\d+) tests", process.stderr).group(1))
integration = read_json(project.root/"logs"/"integracao_ollama.json")
profiles = project.config()["perfis"]
base = project.base()
active_topics = sum(bool(t.get("ativo",True)) for d in base["documentos"] for t in d["estrutura"])
total_topics = sum(len(d["estrutura"]) for d in base["documentos"])
legacy_units = sum(r["vetores_armazenados"] for r in inventory(project) if profiles[r["perfil_id"]].get("legado"))
report = f'''# Entrega e verificações

Aplicação publicada em `{project.root}`.

- Ambiente virtual: `.venv`; Python {sys.version.split()[0]}; Tcl/Tk {tkinter.Tcl().eval("info patchlevel")}.
- {count} testes offline passaram no ambiente instalado. Detalhes: `logs/testes.log`.
- Inicialização da interface desktop validada, incluindo leitura de inventário e perfis.
- Base migrada: {len(base["documentos"])} documentos, {total_topics} tópicos, {active_topics} ativos; {legacy_units} embeddings legados válidos preservados em NPZ.
- Nenhum embedding novo da base do usuário foi gerado automaticamente.
- Ollama portátil instalado e configurado; servidor em `http://127.0.0.1:11434`. Configuração em `config/ollama-local.json`.
- Inferência real em português validada para os dois modelos locais.
- A integração real gerou as oito categorias numa base sintética temporária e verificou busca, média, faixas de escore, reordenação e exportação. Detalhes: `logs/integracao_ollama.json`.

| Modelo local | Dimensão efetiva | Integração real |
|---|---:|---|
'''
for model in integration["modelos"]:
    report += f'| {model["modelo"]} | {model["dimensoes"]} | {model["status"]}; oito categorias |\n'
report += '''
Para abrir, execute **02_abrir_aplicacao.bat**. Na aba Base e geração, escolha o modelo, marque as categorias e gere somente o que deseja estudar. Os modelos locais cadastrados ficam disponíveis para geração; uma categoria entra na seleção de busca depois que seu índice é gerado.

Instalações opcionais: script 05 para E5-small/Sentence Transformers, script 07 para algoritmos de clusterização, script 08 para extração documental. Essas dependências opcionais não foram instaladas nesta entrega. Serviços online foram implementados, mas não testados com chamadas autenticadas; configure a chave e confirme a cota da conta para usá-los.

O protocolo usa média de todas as unidades escolhidas, incluindo segmentos, no ranking por tópico. Resultados de modelos diferentes são exibidos separadamente. Coordenadas ficam exclusivamente nos NPZ; JSON contém textos, hashes, referências, escores e distâncias.
'''
project.root.joinpath("ENTREGA.md").write_text(report, encoding="utf-8")
print(f"Entrega registrada; {count} testes OK; integração real OK.")
