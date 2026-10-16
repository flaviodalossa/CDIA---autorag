"""Teste real opcional: python tests/integracao_ollama.py.

Usa uma base sintética temporária; não modifica os índices da base do usuário.
"""
from pathlib import Path
import json
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from autorag.core import CATEGORIES, Project, write_json
from autorag.service import generate, import_base, inventory
from autorag.experiments import export_experiment, rerank, score_groups, search
from test_autorag import fixture


def main():
    root = Path(__file__).resolve().parents[1]
    project = Project(root)
    local_profiles = {k: p for k, p in project.config()["perfis"].items() if p["provedor"]=="ollama"}
    if not local_profiles:
        raise ValueError("Nenhum perfil Ollama cadastrado; execute 04_configurar_ollama.bat.")
    cache = root/".cache"
    cache.mkdir(exist_ok=True)
    report = {"base_de_teste": "sintetica_temporaria", "modelos": []}
    with tempfile.TemporaryDirectory(prefix="integracao_", dir=cache) as temporary:
        lab = Project(Path(temporary)/"laboratorio")
        source = Path(temporary)/"fixture.json"
        write_json(source, fixture())
        import_base(lab, source)
        lab.save_config({"perfis": local_profiles, "perfil_ativo_id": next(iter(local_profiles))})
        for profile_id, profile in local_profiles.items():
            started = time.perf_counter()
            print("Verificando oito categorias:", profile["modelo"], flush=True)
            generate(lab, profile_id, list(CATEGORIES), batch_size=3)
            rows = [row for row in inventory(lab) if row["perfil_id"]==profile_id]
            assert len(rows)==8 and all(row["atuais"]==3 for row in rows)
            result = search(lab, profile_id, ["resumo", "conteudo_fonte", "palavras_chave"],
                            "Como estudar o consentimento em pesquisa?", top_k=2)
            assert result["avaliados"]==9 and len(result["resultados"])==2
            for row in result["resultados"]:
                assert len(row["comparacoes"])==3
            grouped = score_groups(lab, result["execucao_id"], [-1,0,.5,1])
            assert sum(row["quantidade"] for row in grouped["faixas"])==9
            reranked = rerank(lab, result["execucao_id"], view="vetores", top_k=None)
            assert len(reranked["resultados"])==9
            export_experiment(lab, result["execucao_id"], Path(temporary)/f"{profile_id}.zip")
            report["modelos"].append({"modelo": profile["modelo"], "dimensoes": profile["dimensoes"],
                                      "categorias": 8, "topicos_por_categoria": 3,
                                      "vetores_avaliados": 9, "segundos": round(time.perf_counter()-started,2), "status": "ok"})
    path = root/"logs"/"integracao_ollama.json"
    write_json(path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
