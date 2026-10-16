from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .core import CATEGORIES, Project, read_json, topics, uid
from .service import generate, import_base, inventory, register_profile
from .providers import available_models
from .experiments import (export_experiment, list_experiments, load_experiment,
                          rerank, score_groups, search)


def output(value):
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def limits(parser):
    parser.add_argument("--visao", choices=["topicos", "vetores"], default="topicos")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--top-k", type=int, default=5)
    group.add_argument("--sem-top-k", action="store_true")
    parser.add_argument("--minimo", type=float)
    parser.add_argument("--maximo", type=float)
    parser.add_argument("--cobertura-parcial", action="store_true")


def main(argv=None):
    parser = argparse.ArgumentParser(description="AutoRAG Conceito — laboratório de embeddings")
    parser.add_argument("--projeto", default=str(Path(__file__).resolve().parents[1]))
    sub = parser.add_subparsers(dest="comando", required=True)
    imp = sub.add_parser("importar", help="Importa/atualiza JSON sem gerar embeddings")
    imp.add_argument("--base-json", required=True)
    imp.add_argument("--npz-legado")
    model = sub.add_parser("modelos", help="Lista modelos instalados e opções online")
    model.add_argument("--endpoint", default="http://127.0.0.1:11434")
    model.add_argument("--pasta-modelos")
    reg = sub.add_parser("cadastrar", help="Registra perfil imutável de modelo")
    reg.add_argument("--provedor", required=True, choices=["ollama", "sentence_transformers", "openai", "gemini"])
    reg.add_argument("--modelo", required=True)
    reg.add_argument("--dimensoes", type=int)
    reg.add_argument("--endpoint")
    reg.add_argument("--pasta-modelos")
    sub.add_parser("inventario")
    gen = sub.add_parser("gerar")
    gen.add_argument("--perfil", required=True)
    gen.add_argument("--categorias", nargs="+", required=True, choices=CATEGORIES)
    gen.add_argument("--operacao", choices=["faltantes", "atualizar", "regenerar"], default="atualizar")
    gen.add_argument("--topicos", nargs="+")
    gen.add_argument("--documentos", nargs="+")
    gen.add_argument("--lote", type=int, default=16)
    gen.add_argument("--simular", action="store_true")
    find = sub.add_parser("buscar")
    find.add_argument("--perfil", nargs="+", required=True, help="Um ou vários perfis; rankings separados")
    find.add_argument("--categorias", nargs="+", required=True, choices=CATEGORIES)
    query_group = find.add_mutually_exclusive_group(required=True)
    query_group.add_argument("--consulta")
    query_group.add_argument("--arquivo-consulta", help="Lê texto exato de arquivo UTF-8")
    find.add_argument("--regenerar-entrada", action="store_true")
    find.add_argument("--limiares-individuais", help='Objeto JSON, ex.: {"resumo":0.5}')
    find.add_argument("--regra-individual", choices=["todos", "qualquer"], default="todos")
    limits(find)
    sub.add_parser("execucoes")
    reopen = sub.add_parser("abrir")
    reopen.add_argument("execucao")
    ranges = sub.add_parser("faixas")
    ranges.add_argument("execucao")
    ranges.add_argument("--limites", nargs="+", required=True, type=float)
    ranges.add_argument("--base-escore", choices=["individual", "media"], default="individual")
    rank = sub.add_parser("reordenar")
    rank.add_argument("execucao")
    rank.add_argument("--categorias", nargs="+", choices=CATEGORIES)
    limits(rank)
    export = sub.add_parser("exportar")
    export.add_argument("execucao")
    export.add_argument("--destino", required=True)
    sub.add_parser("validar")
    analyze = sub.add_parser("analisar", help="Manipula referências de um experimento salvo")
    analyze.add_argument("execucao")
    analyze.add_argument("--categorias", nargs="+", choices=CATEGORIES)
    analyze.add_argument("--minimo", type=float)
    analyze.add_argument("--maximo", type=float)
    analyze.add_argument("--operacao", required=True, choices=["centroide", "soma", "matriz", "cluster"])
    analyze.add_argument("--metodo", default="kmeans", choices=["kmeans", "minibatch", "aglomerativo", "dbscan", "hdbscan", "espectral"])
    analyze.add_argument("--parametros", default="{}", help="Parâmetros JSON do algoritmo científico")
    args = parser.parse_args(argv)
    project = Project(args.projeto)
    try:
        command = args.comando
        if command == "importar":
            result = import_base(project, args.base_json, args.npz_legado)
        elif command == "modelos":
            result = available_models(args.endpoint, args.pasta_modelos or str(project.root/"modelos"))
        elif command == "cadastrar":
            result = register_profile(project, args.provedor, args.modelo, args.dimensoes,
                                      endpoint=args.endpoint, models_dir=args.pasta_modelos or str(project.root/"modelos"))
        elif command == "inventario":
            result = inventory(project)
        elif command == "gerar":
            result = generate(project, args.perfil, args.categorias, args.operacao, args.topicos,
                              args.documentos, args.lote, args.simular, progress=lambda m: print(m, file=sys.stderr))
        elif command == "buscar":
            query = Path(args.arquivo_consulta).read_text(encoding="utf-8-sig") if args.arquivo_consulta else args.consulta
            comparison = uid("comparacao") if len(args.perfil)>1 else None
            result = [search(project, profile, args.categorias, query, args.visao,
                             None if args.sem_top_k else args.top_k, args.minimo, args.maximo,
                             args.cobertura_parcial, json.loads(args.limiares_individuais or "{}"),
                             args.regra_individual, regenerate_input=args.regenerar_entrada, comparison_id=comparison)
                      for profile in args.perfil]
            if len(result) == 1:
                result = result[0]
        elif command == "execucoes":
            result = [{key: row.get(key) for key in ("execucao_id", "perfil_id", "data", "status", "texto_original_entrada", "quantidade_resultados")}
                      for row in list_experiments(project)]
        elif command == "abrir":
            result = load_experiment(project, args.execucao, verify=True)
        elif command == "faixas":
            result = score_groups(project, args.execucao, args.limites, args.base_escore)
        elif command == "reordenar":
            result = rerank(project, args.execucao, args.categorias, args.visao,
                            None if args.sem_top_k else args.top_k, args.minimo, args.maximo, args.cobertura_parcial)
        elif command == "exportar":
            result = {"arquivo": export_experiment(project, args.execucao, args.destino)}
        elif command == "validar":
            all_topics = list(topics(project.base(), active=False))
            result = {"status": "ok", "topicos": len(all_topics), "inventario": inventory(project)}
            for experiment in list_experiments(project):
                load_experiment(project, experiment["execucao_id"], verify=True)
            result["experimentos_verificados"] = len(list_experiments(project))
        elif command == "analisar":
            from .analysis import cluster, derive, experiment_references, pairwise
            refs = experiment_references(project, args.execucao, args.categorias, args.minimo, args.maximo)
            if args.operacao == "matriz":
                result = pairwise(project, refs)
            elif args.operacao == "cluster":
                result = cluster(project, refs, args.metodo, json.loads(args.parametros))
            else:
                result = derive(project, refs, args.operacao)
        output(result)
        return 0
    except (ValueError, RuntimeError, OSError, KeyError, TypeError, ImportError) as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
