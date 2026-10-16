"""Atalho: python gerar_embeddings.py --perfil ID --categorias resumo metadados."""
import sys
from autorag.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["gerar", *sys.argv[1:]]))
