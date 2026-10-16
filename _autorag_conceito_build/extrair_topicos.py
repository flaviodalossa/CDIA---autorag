"""Extrai via pipeline original e importa o resultado preservando índices.

Saída intermediária isolada em dados/base_extracao.json; a publicação de uma
nova base é feita pelo serviço de importação, mantendo versões de embeddings.
"""
from pathlib import Path
import shutil
import sys

from autorag.core import Project
from autorag.service import import_base


def main():
    try:
        import pipeline_etica as pipeline
    except ImportError as exc:
        print(f"Instale 08_instalar_extracao.bat antes de usar a extração: {exc}", file=sys.stderr)
        return 1
    project = Project(Path(__file__).resolve().parent)
    intermediate = project.data/"base_extracao.json"
    # O extrator usa uma cópia da base atual; não substitui a base de busca em andamento.
    with project.lock():
        if project.base_path.exists():
            shutil.copy2(project.base_path, intermediate)
        args = pipeline.build_parser().parse_args(["topicos", "--base-json", str(intermediate),
                                                   "--documentos-dir", str(project.root/"documentos"), *sys.argv[1:]])
        if Path(args.base_json).resolve() != intermediate:
            raise ValueError("Use autorag_cli.py importar para uma base externa; este comando publica apenas sua saída intermediária.")
        pipeline.process_topics(args)
    result = import_base(project, intermediate)
    print("Base atualizada:", result)
    print("Embeddings preservados. Selecione as categorias e modelos que deseja atualizar.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
