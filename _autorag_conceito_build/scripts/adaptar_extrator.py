"""Utilitário de migração do extrator original; nunca imprime credenciais."""
import ast
from pathlib import Path
import re
import sys

source = Path(sys.argv[1]).read_text(encoding="utf-8-sig")
tree = ast.parse(source)
lines = source.splitlines(keepends=True)
replacements = []
for node in tree.body:
    if isinstance(node, ast.Assign) and any(isinstance(n, ast.Name) and "API_KEY" in n.id for n in node.targets):
        replacements.append((node.lineno-1, node.end_lineno, f'{node.targets[0].id} = ""\n'))
    if isinstance(node, ast.FunctionDef) and node.name == "get_openai_api_key":
        replacements.append((node.lineno-1, node.end_lineno, '''def get_openai_api_key() -> str:
    from embedding_credentials import obter_chave
    key = obter_chave("openai")
    if not key:
        raise ValueError("Configure OPENAI_API_KEY ou o campo direto em embedding_credentials.py.")
    return key
'''))
    if isinstance(node, ast.FunctionDef) and node.name == "process_embeddings":
        replacements.append((node.lineno-1, node.end_lineno, '''def process_embeddings(args: argparse.Namespace) -> None:
    from autorag.core import Project, CATEGORIES
    from autorag.service import import_base, register_profile, generate
    project = Project(Path(__file__).resolve().parent)
    if Path(args.base_json).resolve() != project.base_path or not project.base_path.exists():
        import_base(project, args.base_json)
    profile = next((p for p in project.config()["perfis"].values()
                    if p["provedor"] == "openai" and p["modelo"] == args.modelo
                    and (args.dimensoes is None or args.dimensoes == p["dimensoes"])), None)
    if profile is None:
        profile = register_profile(project, "openai", args.modelo, args.dimensoes)
    ids = read_id_values(args.ids, args.ids_arquivo)
    result = generate(project, profile["perfil_id"], [CATEGORIES[0]],
                      "regenerar" if ids else "atualizar", topic_ids=ids or None,
                      batch_size=args.tamanho_lote, simulate=args.simular,
                      progress=logging.info)
    logging.info("%s", result)
'''))
for left, right, content in sorted(replacements, reverse=True):
    lines[left:right] = [content]
result = "".join(lines)
result = re.sub(r"sk-[A-Za-z0-9_-]{20,}", "CHAVE_REMOVIDA_NA_MIGRACAO", result)
ast.parse(result)
Path(sys.argv[2]).write_text(result, encoding="utf-8")
print("Extrator adaptado; credenciais centralizadas; embeddings delegados ao armazenamento NPZ.")
