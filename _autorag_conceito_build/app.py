"""Interface desktop Windows. Inferência ocorre em thread; Tk fica na principal."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import queue
import threading
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

from autorag.core import CATEGORIES, LABELS, Project, uid
from autorag.providers import available_models
from autorag.service import generate, import_base, inventory, register_profile
from autorag.experiments import (export_experiment, list_experiments, load_experiment,
                                 rerank, score_groups, search)


class Application(tk.Tk):
    def __init__(self, project=None):
        super().__init__()
        self.project = project or Project(Path(__file__).resolve().parent)
        self.title("AutoRAG Conceito — Laboratório de embeddings")
        self.geometry("1320x850")
        self.minsize(1000, 680)
        self.events = queue.Queue()
        self.running = False
        self.catalog = []
        self.profiles = {}
        self.result_rows = {}
        self.current_experiment = None
        self.protocol("WM_DELETE_WINDOW", self.close)
        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=10, pady=8)
        base_tab, search_tab, history_tab = [ttk.Frame(notebook, padding=8) for _ in range(3)]
        for frame, name in zip([base_tab, search_tab, history_tab], ["Base e geração", "Recuperação e comparação", "Experimentos salvos"]):
            notebook.add(frame, text=name)
        self.build_base(base_tab)
        self.build_search(search_tab)
        self.build_history(history_tab)
        self.status = tk.StringVar(value="Pronto")
        ttk.Label(self, textvariable=self.status).pack(anchor="w", padx=12)
        self.log = tk.Text(self, height=5, wrap="word", state="disabled")
        self.log.pack(fill="x", padx=10, pady=(2, 8))
        self.refresh()
        self.after(100, self.poll)

    def tree(self, parent, columns, headings, height=8):
        frame = ttk.Frame(parent)
        frame.pack(fill="both", expand=True, pady=5)
        tree = ttk.Treeview(frame, columns=columns, show="headings", height=height, selectmode="extended")
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        horizontal = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=scrollbar.set, xscrollcommand=horizontal.set)
        tree.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        for name, heading in zip(columns, headings):
            tree.heading(name, text=heading)
            tree.column(name, width=150, minwidth=80)
        return tree

    def build_base(self, frame):
        actions = ttk.Frame(frame)
        actions.pack(fill="x")
        ttk.Button(actions, text="Importar / atualizar base JSON", command=self.import_dialog).pack(side="left")
        ttk.Button(actions, text="Atualizar inventário", command=self.refresh).pack(side="left", padx=6)
        self.base_status = tk.StringVar()
        ttk.Label(actions, textvariable=self.base_status).pack(side="left", padx=8)
        model_frame = ttk.LabelFrame(frame, text="Modelos de embeddings", padding=6)
        model_frame.pack(fill="x", pady=8)
        self.endpoint = tk.StringVar(value="http://127.0.0.1:11434")
        ttk.Label(model_frame, text="Ollama:").grid(row=0, column=0)
        ttk.Entry(model_frame, textvariable=self.endpoint, width=30).grid(row=0, column=1, padx=6)
        ttk.Button(model_frame, text="Descobrir modelos", command=self.discover).grid(row=0, column=2)
        self.model_combo = ttk.Combobox(model_frame, state="readonly", width=58)
        self.model_combo.grid(row=0, column=3, padx=6)
        ttk.Label(model_frame, text="Dimensão:").grid(row=0, column=4)
        self.dimensions = tk.StringVar()
        ttk.Entry(model_frame, textvariable=self.dimensions, width=7).grid(row=0, column=5)
        self.model_combo.bind("<<ComboboxSelected>>", lambda _: self.dimensions.set(str(self.catalog[self.model_combo.current()].get("dimensoes") or "")))
        ttk.Button(model_frame, text="Cadastrar perfil", command=self.register).grid(row=0, column=6, padx=6)
        ttk.Label(model_frame, text="Perfis separam modelo, versão dos pesos, dimensão e preparação do texto. Opções online dependem de chave e cota.").grid(row=1, column=0, columnspan=7, sticky="w", pady=4)
        gen = ttk.LabelFrame(frame, text="Geração independente por modelo e categoria", padding=7)
        gen.pack(fill="x")
        self.profile_combo = ttk.Combobox(gen, state="readonly", width=70)
        self.profile_combo.grid(row=0, column=0, columnspan=3, sticky="w")
        self.operation = tk.StringVar(value="atualizar")
        ttk.Combobox(gen, textvariable=self.operation, values=["faltantes", "atualizar", "regenerar"], state="readonly", width=14).grid(row=0, column=3, padx=5)
        ttk.Label(gen, text="Lote:").grid(row=0, column=4)
        self.batch = tk.StringVar(value="16")
        ttk.Entry(gen, textvariable=self.batch, width=6).grid(row=0, column=5)
        self.generate_categories = {}
        for number, category in enumerate(c for c in CATEGORIES if c != "enriquecimento_recuperacao"):
            variable = tk.BooleanVar(value=category == CATEGORIES[0])
            self.generate_categories[category] = variable
            ttk.Checkbutton(gen, text=LABELS[category], variable=variable).grid(row=1+number//4, column=number%4, sticky="w", pady=2)
        ttk.Label(gen, text="IDs de tópicos (espaço ou vírgula; vazio = todos):").grid(row=3, column=0, sticky="w", pady=4)
        self.scope_topics = tk.StringVar()
        ttk.Entry(gen, textvariable=self.scope_topics, width=52).grid(row=3, column=1, columnspan=2, sticky="ew")
        ttk.Label(gen, text="IDs de documentos:").grid(row=4, column=0, sticky="w")
        self.scope_docs = tk.StringVar()
        ttk.Entry(gen, textvariable=self.scope_docs, width=52).grid(row=4, column=1, columnspan=2, sticky="ew")
        ttk.Button(gen, text="Simular seleção", command=lambda: self.generate_action(True)).grid(row=3, column=3, padx=5)
        ttk.Button(gen, text="Gerar selecionados", command=lambda: self.generate_action(False)).grid(row=4, column=3, padx=5)
        self.inventory_tree = self.tree(frame, ["modelo", "categoria", "atuais", "stale", "missing", "empty", "stored"],
                                        ["Modelo / dimensões", "Categoria", "Tópicos atuais", "Desatualizados", "Faltantes", "Sem texto", "Unidades NPZ"], 9)
        self.inventory_tree.column("modelo", width=310)
        self.inventory_tree.column("categoria", width=310)

    def build_search(self, frame):
        ttk.Label(frame, text="Selecione pares de modelo e categoria com Ctrl/Shift. Cada modelo produz seu próprio ranking.").pack(anchor="w")
        self.search_pairs = self.tree(frame, ["modelo", "perfil", "categoria", "count"],
                                      ["Modelo / dimensões", "Perfil", "Categoria disponível", "Tópicos atuais"], 5)
        self.search_pairs.column("modelo", width=320)
        self.search_pairs.column("categoria", width=330)
        ttk.Label(frame, text="Texto original da consulta:").pack(anchor="w")
        self.query = tk.Text(frame, height=4, wrap="word")
        self.query.pack(fill="x", pady=3)
        controls = ttk.Frame(frame)
        controls.pack(fill="x", pady=6)
        self.view = tk.StringVar(value="topicos")
        ttk.Radiobutton(controls, text="Tópicos (média dos vetores)", variable=self.view, value="topicos").pack(side="left")
        ttk.Radiobutton(controls, text="Vetores individuais", variable=self.view, value="vetores").pack(side="left", padx=8)
        self.top_k = tk.StringVar(value="5")
        ttk.Label(controls, text="Top-k:").pack(side="left")
        ttk.Entry(controls, textvariable=self.top_k, width=5).pack(side="left", padx=3)
        self.no_top_k = tk.BooleanVar(value=False)
        ttk.Checkbutton(controls, text="Sem top-k", variable=self.no_top_k).pack(side="left")
        self.minimum, self.maximum = tk.StringVar(), tk.StringVar()
        for label, variable in [("Escore mínimo:", self.minimum), ("Máximo:", self.maximum)]:
            ttk.Label(controls, text=label).pack(side="left", padx=(8, 2))
            ttk.Entry(controls, textvariable=variable, width=6).pack(side="left")
        self.partial = tk.BooleanVar()
        ttk.Checkbutton(controls, text="Cobertura parcial", variable=self.partial).pack(side="left", padx=6)
        more = ttk.Frame(frame)
        more.pack(fill="x")
        ttk.Label(more, text='Mínimos por categoria (JSON opcional, ex.: {"resumo": 0.5}):').pack(side="left")
        self.individual = tk.StringVar()
        ttk.Entry(more, textvariable=self.individual, width=38).pack(side="left", padx=5)
        self.individual_rule = tk.StringVar(value="todos")
        ttk.Combobox(more, values=["todos", "qualquer"], textvariable=self.individual_rule, width=8, state="readonly").pack(side="left")
        ttk.Button(more, text="Buscar / comparar", command=self.search_action).pack(side="left", padx=8)
        self.results_tree = self.tree(frame, ["modelo", "rank", "topic", "category", "score", "distance", "title"],
                                      ["Modelo", "Posição", "Tópico", "Categoria / combinação", "Similaridade", "Distância cosseno", "Título"], 8)
        self.results_tree.column("title", width=360)
        self.results_tree.column("modelo", width=240)
        self.results_tree.bind("<Double-1>", self.result_detail)
        ttk.Label(frame, text="Duplo clique mostra textos recuperados, escores individuais, média e referências aos vetores.").pack(anchor="w")

    def build_history(self, frame):
        bar = ttk.Frame(frame)
        bar.pack(fill="x")
        for label, callback in [("Atualizar", self.refresh_history), ("Reabrir resultados", self.reopen),
                                ("Todos os escores", self.universe_dialog), ("Separar faixas", self.ranges_dialog),
                                ("Reordenar sem inferência", self.rerank_dialog), ("Exportar ZIP", self.export_dialog),
                                ("Centroide / matriz / clusters", self.analysis_dialog)]:
            ttk.Button(bar, text=label, command=callback).pack(side="left", padx=3)
        self.history_tree = self.tree(frame, ["id", "model", "query", "view", "count", "status"],
                                      ["Execução", "Modelo", "Consulta", "Visão", "Resultados", "Status"], 17)
        self.history_tree.column("id", width=320)
        self.history_tree.column("query", width=340)
        self.history_tree.bind("<Double-1>", lambda _: self.reopen())
        ttk.Label(frame, text="As faixas são calculadas sobre o universo completo da consulta, por modelo e categoria, inclusive abaixo do mínimo ou fora do top-k.").pack(anchor="w")

    def emit(self, message):
        self.events.put(("log", str(message)))

    def start(self, label, function, done=None):
        if self.running:
            messagebox.showinfo("Operação em andamento", "Aguarde a operação atual terminar.")
            return
        self.running = True
        self.status.set(label)
        def worker():
            try:
                self.events.put(("done", (function(), done)))
            except Exception as exc:
                self.events.put(("error", (str(exc), traceback.format_exc())))
        threading.Thread(target=worker, daemon=True).start()

    def poll(self):
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == "log":
                    self.log.configure(state="normal")
                    self.log.insert("end", value+"\n")
                    self.log.see("end")
                    self.log.configure(state="disabled")
                elif kind == "done":
                    self.running = False
                    self.status.set("Pronto")
                    result, callback = value
                    if callback:
                        callback(result)
                    self.refresh()
                elif kind == "error":
                    self.running = False
                    self.status.set("Operação não concluída")
                    self.emit(value[0])
                    path = self.project.root/"logs"/"ultimo_erro.log"
                    path.parent.mkdir(exist_ok=True)
                    path.write_text(value[1], encoding="utf-8")
                    messagebox.showerror("Erro", value[0])
                    self.refresh()
        except queue.Empty:
            pass
        self.after(100, self.poll)

    def refresh(self):
        self.profiles = self.project.config()["perfis"]
        profile_values = [f"{p['modelo']} | {p['dimensoes']}d | {p['perfil_id']}" for p in self.profiles.values()]
        self.profile_combo["values"] = profile_values
        if profile_values and self.profile_combo.get() not in profile_values:
            active = self.project.config().get("perfil_ativo_id")
            ids = list(self.profiles)
            self.profile_combo.current(ids.index(active) if active in ids else 0)
        for tree in (self.inventory_tree, self.search_pairs):
            tree.delete(*tree.get_children())
        if self.project.base_path.exists():
            try:
                rows = inventory(self.project)
                for number, row in enumerate(rows):
                    label = f"{row['modelo']} ({row['dimensoes']}d)"
                    self.inventory_tree.insert("", "end", values=[label, LABELS[row["categoria"]], row["atuais"], row["desatualizados"], row["faltantes"], row["sem_texto"], row["vetores_armazenados"]])
                    if row["revisao"]:
                        self.search_pairs.insert("", "end", iid=f"{row['perfil_id']}|{row['categoria']}", values=[label, row["perfil_id"], LABELS[row["categoria"]], row["atuais"]])
                self.base_status.set(f"{len(self.project.base()['documentos'])} documentos; {len(self.profiles)} perfis")
            except Exception as exc:
                self.base_status.set(str(exc))
        else:
            self.base_status.set("Importe uma base JSON para começar")
        self.refresh_history()

    def refresh_history(self):
        self.history_tree.delete(*self.history_tree.get_children())
        for row in list_experiments(self.project):
            self.history_tree.insert("", "end", iid=row["execucao_id"], values=[row["execucao_id"], row["perfil"]["modelo"], row["texto_original_entrada"].replace("\n", " ")[:160], row["parametros"]["visao"], row.get("quantidade_resultados", 0), row["status"]])

    def import_dialog(self):
        source = filedialog.askopenfilename(title="Base JSON existente", filetypes=[("Base JSON", "*.json")])
        if source:
            self.start("Importando base e índice legado...", lambda: import_base(self.project, source),
                       lambda value: self.emit(json.dumps(value, ensure_ascii=False)))

    def discover(self):
        endpoint = self.endpoint.get().strip()
        def done(rows):
            self.catalog = rows
            self.model_combo["values"] = [f"{r['modelo']} | {r['dimensoes']}d | {'disponível' if r['disponivel'] else r['detalhe']}" for r in rows]
            if rows:
                self.model_combo.current(0)
                self.dimensions.set(str(rows[0]["dimensoes"]))
        self.start("Descobrindo modelos...", lambda: available_models(endpoint, str(self.project.root/"modelos")), done)

    def register(self):
        selected = self.model_combo.current()
        if selected < 0:
            messagebox.showinfo("Modelo", "Use Descobrir modelos e selecione uma opção.")
            return
        row = self.catalog[selected]
        try:
            dimensions = int(self.dimensions.get())
        except ValueError:
            messagebox.showerror("Dimensão", "Informe uma dimensão inteira positiva.")
            return
        endpoint = self.endpoint.get().strip()
        self.start("Cadastrando perfil...", lambda: register_profile(self.project, row["provedor"], row["modelo"], dimensions,
                   endpoint=endpoint if row["provedor"]=="ollama" else None, models_dir=str(self.project.root/"modelos")),
                   lambda value: self.emit(f"Perfil cadastrado: {value['perfil_id']}"))

    def generate_action(self, simulate):
        selected = self.profile_combo.current()
        if selected < 0:
            messagebox.showinfo("Perfil", "Cadastre ou importe um perfil de modelo primeiro.")
            return
        profile_id = list(self.profiles)[selected]
        categories = [name for name, value in self.generate_categories.items() if value.get()]
        try:
            batch_size = int(self.batch.get())
        except ValueError:
            messagebox.showerror("Lote", "Informe um número inteiro positivo.")
            return
        ids = lambda value: value.replace(",", " ").split() or None
        topic_ids, document_ids = ids(self.scope_topics.get()), ids(self.scope_docs.get())
        operation = self.operation.get()
        self.start("Simulando seleção..." if simulate else "Gerando embeddings...",
                   lambda: generate(self.project, profile_id, categories, operation, topic_ids, document_ids,
                                    batch_size, simulate, progress=self.emit),
                   lambda value: self.emit(json.dumps(value, ensure_ascii=False)))

    def search_options(self):
        number = lambda value: float(value.replace(",", ".")) if value.strip() else None
        return dict(view=self.view.get(), top_k=None if self.no_top_k.get() else int(self.top_k.get()),
                    minimum=number(self.minimum.get()), maximum=number(self.maximum.get()), partial=self.partial.get())

    def search_action(self):
        pairs = defaultdict(list)
        for selected in self.search_pairs.selection():
            profile, category = selected.split("|", 1)
            pairs[profile].append(category)
        if not pairs:
            messagebox.showinfo("Seleção", "Selecione os pares de modelo e categoria para buscar.")
            return
        query = self.query.get("1.0", "end-1c")
        try:
            options = self.search_options()
            individual = json.loads(self.individual.get() or "{}")
            if not isinstance(individual, dict):
                raise ValueError("Mínimos individuais devem ser um objeto JSON.")
        except ValueError as exc:
            messagebox.showerror("Parâmetros", str(exc))
            return
        rule = self.individual_rule.get()
        comparison_id = uid("comparacao") if len(pairs)>1 else None
        def run():
            executions = []
            for profile, categories in pairs.items():
                self.emit(f"Buscando em {self.profiles[profile]['modelo']}: {', '.join(categories)}")
                result = search(self.project, profile, categories, query, **options,
                                individual_limits={k: v for k, v in individual.items() if k in categories},
                                individual_rule=rule, comparison_id=comparison_id)
                executions.append(result)
                self.emit(f"Experimento salvo: {result['execucao_id']}")
            return executions
        self.start("Calculando buscas por modelo...", run, self.display_results)

    def display_results(self, experiments):
        self.results_tree.delete(*self.results_tree.get_children())
        self.result_rows = {}
        for experiment in experiments:
            self.current_experiment = experiment["execucao_id"]
            for row in experiment["resultados"]:
                score = row.get("similaridade_media", row.get("similaridade_cosseno"))
                distance = row.get("distancia_cosseno_media", row.get("distancia_cosseno"))
                category = row.get("categoria", " + ".join(experiment["parametros"]["categorias"]))
                item = self.results_tree.insert("", "end", values=[experiment["perfil"]["modelo"], row["posicao"], row["topico_id"], category, f"{score:.6f}", f"{distance:.6f}", row["titulo"]])
                self.result_rows[item] = {"execucao_id": experiment["execucao_id"], "perfil": experiment["perfil"], **row}
            self.emit(f"{experiment['perfil']['modelo']}: {len(experiment['resultados'])} resultados / {experiment['avaliados']} vetores avaliados; {experiment['motivo_termino']}")

    def text_dialog(self, title, value):
        window = tk.Toplevel(self)
        window.title(title)
        window.geometry("1000x650")
        text = tk.Text(window, wrap="word")
        text.pack(fill="both", expand=True, padx=8, pady=8)
        scrollbar = ttk.Scrollbar(text, orient="vertical", command=text.yview)
        scrollbar.pack(side="right", fill="y")
        text.configure(yscrollcommand=scrollbar.set)
        text.insert("end", json.dumps(value, ensure_ascii=False, indent=2))
        text.configure(state="disabled")

    def result_detail(self, _=None):
        selected = self.results_tree.selection()
        if selected:
            self.text_dialog("Detalhes e similaridades individuais", self.result_rows[selected[0]])

    def selected_experiment(self):
        selected = self.history_tree.selection()
        if not selected:
            messagebox.showinfo("Experimento", "Selecione um experimento salvo.")
            return None
        return selected[0]

    def reopen(self):
        selected = self.selected_experiment()
        if selected:
            self.start("Reabrindo e verificando artefatos...", lambda: [load_experiment(self.project, selected, True)], self.display_results)

    def universe_dialog(self):
        selected = self.selected_experiment()
        if not selected:
            return
        def show(experiment):
            window = tk.Toplevel(self)
            window.title(f"Todos os escores — {selected}")
            window.geometry("1200x650")
            tree = self.tree(window, ["category", "topic", "segment", "score", "distance", "title"],
                             ["Categoria", "Tópico", "Segmento", "Similaridade", "Distância", "Título"], 20)
            rows = {}
            for row in sorted(experiment["universo"], key=lambda r: (r["categoria"], -r["similaridade_cosseno"])):
                item = tree.insert("", "end", values=[row["categoria"], row["topico_id"], row["segmento"], f"{row['similaridade_cosseno']:.6f}", f"{row['distancia_cosseno']:.6f}", row["titulo"]])
                rows[item] = row
            tree.bind("<Double-1>", lambda _: self.text_dialog("Vetor avaliado", rows[tree.selection()[0]]) if tree.selection() else None)
        self.start("Carregando escores completos...", lambda: load_experiment(self.project, selected, True), show)

    def ranges_dialog(self):
        selected = self.selected_experiment()
        if not selected:
            return
        answer = simpledialog.askstring("Faixas de escore", "Limites em ordem crescente, separados por ponto e vírgula:", initialvalue="-1;0.4;0.7;1")
        if answer is None:
            return
        try:
            boundaries = [float(value.strip().replace(",", ".")) for value in answer.split(";")]
        except ValueError:
            messagebox.showerror("Faixas", "Informe limites numéricos separados por ponto e vírgula.")
            return
        basis = simpledialog.askstring("Base do escore", "Use individual (por categoria) ou media (combinação de categorias):", initialvalue="individual")
        if basis is not None:
            self.start("Separando faixas sem inferência...", lambda: score_groups(self.project, selected, boundaries, basis.strip()),
                       lambda value: self.text_dialog("Conjunto de vetores por faixa", value))

    def rerank_dialog(self):
        selected = self.selected_experiment()
        if not selected:
            return
        answer = simpledialog.askstring("Categorias para reordenar", "Categorias já avaliadas, separadas por espaços; vazio mantém todas:", initialvalue="")
        if answer is None:
            return
        try:
            options = self.search_options()
        except ValueError as exc:
            messagebox.showerror("Parâmetros", str(exc))
            return
        self.emit("Reordenação utiliza visão, top-k, mínimos, máximo e cobertura definidos na aba Recuperação.")
        self.start("Reordenando escores salvos...", lambda: [rerank(self.project, selected, answer.split() or None, **options)], self.display_results)

    def export_dialog(self):
        selected = self.selected_experiment()
        if selected:
            destination = filedialog.asksaveasfilename(title="Exportar experimento", defaultextension=".zip", initialfile=selected+".zip", filetypes=[("Arquivo ZIP", "*.zip")])
            if destination:
                self.start("Exportando JSON e NPZ...", lambda: export_experiment(self.project, selected, destination), self.emit)

    def analysis_dialog(self):
        selected = self.selected_experiment()
        if not selected:
            return
        operation = simpledialog.askstring("Análise do universo completo", "Operação: centroide, soma, matriz ou cluster (um único perfil):", initialvalue="centroide")
        if operation is None:
            return
        operation = operation.strip()
        method, parameters = "kmeans", {}
        if operation == "cluster":
            method = simpledialog.askstring("Clusterização", "Método: kmeans, minibatch, aglomerativo, dbscan, hdbscan, espectral", initialvalue="kmeans")
            raw = simpledialog.askstring("Parâmetros científicos", "Objeto JSON dos parâmetros do algoritmo:", initialvalue='{"n_clusters": 8}')
            if method is None or raw is None:
                return
            try:
                parameters = json.loads(raw)
            except ValueError as exc:
                messagebox.showerror("Parâmetros", str(exc))
                return
        try:
            options = self.search_options()
        except ValueError as exc:
            messagebox.showerror("Parâmetros", str(exc))
            return
        def run():
            from autorag.analysis import cluster, derive, experiment_references, pairwise
            refs = experiment_references(self.project, selected, minimum=options["minimum"], maximum=options["maximum"])
            if operation == "matriz":
                return pairwise(self.project, refs)
            if operation == "cluster":
                return cluster(self.project, refs, method, parameters)
            return derive(self.project, refs, operation)
        self.start("Calculando análise e salvando proveniência...", run,
                   lambda value: self.text_dialog("Análise salva", value))

    def close(self):
        if self.running:
            messagebox.showinfo("Operação em andamento", "Aguarde o processamento terminar antes de fechar a aplicação.")
            return
        self.destroy()


if __name__ == "__main__":
    Application().mainloop()
