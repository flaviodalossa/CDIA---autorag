"""Atalho: python buscar_topicos.py --perfil ID --categorias resumo --consulta TEXTO."""
import sys
from autorag.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["buscar", *sys.argv[1:]]))
