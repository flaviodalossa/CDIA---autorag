# AutoRAG Web v0.2

A aplicação fica em _autorag_conceito_build. A versão web compara até três experimentos por entrada, salva configurações reutilizáveis, registra avaliações opcionais de LLM e permite baixar os resultados e os vetores numéricos usados. Os aplicativos desktop, CLI e experimentos anteriores continuam disponíveis.

## Consulta e laboratório

Sem experimentos selecionados no laboratório, a consulta usa distância cosseno sobre os vetores de conteudo_fonte, com Top K 6 e o perfil ativo do projeto. Não há agrupamento nesse padrão.

No laboratório, cada experimento tem nome, modelos de embeddings, categorias e parâmetros próprios. Os métodos são:

- Similaridade vetorial: ordena cada tópico pela menor distância dos seus vetores selecionados.
- UMAP + HDBSCAN: inclui a consulta no agrupamento, seleciona tópicos pela cobertura de categorias distintas no grupo da consulta e ordena o conjunto pela distância no espaço original. Ruído e dados insuficientes são informados; a recuperação do conteúdo original continua disponível.
- Laboratório de vetores: usa a média das distâncias dos vetores selecionados de cada tópico. Permite cobertura parcial e mínimos de similaridade por categoria.

Distâncias disponíveis: cosseno, euclidiana e Manhattan. Top K, similaridade cosseno mínima, distância máxima, profundidade da fusão, semente e parâmetros UMAP/HDBSCAN são ajustáveis. O dendrograma é opcional. Os filtros atuam em cada modelo antes de Top K e fusão.

Escolha categorias comuns aos modelos ou personalize categorias por modelo, como resumo com Gemma e perguntas com MiniLM. Modelos com dimensões distintas são consultados separadamente; os tópicos são combinados por Reciprocal Rank Fusion (RRF, k=60). A pontuação RRF não é uma distância vetorial. Os escores e evidências de cada modelo acompanham o resultado.

É necessário gerar os índices das categorias escolhidas em Preparar base. Um modelo sem índice adequado produz um erro identificado no experimento, preservando os demais resultados. A fusão não declara sucesso com apenas parte dos modelos solicitados.

## Salvar, reutilizar e reproduzir

1. Edite até três experimentos. Guardar alterações atualiza o rascunho.
2. Dê um nome ao conjunto e use Salvar configuração para disponibilizá-lo no projeto. Configurações JSON também podem ser baixadas e importadas.
3. Usar estes experimentos na consulta salva os parâmetros que serão utilizados em Consulta. Usar consulta padrão restaura a similaridade vetorial do conteúdo original. A seleção é compartilhada pelos usuários do mesmo projeto.
4. Execute uma comparação com uma entrada comum. Configurações, resultados, referências dos índices e avaliações são salvos.
5. Reusar configuração aplica os parâmetros à base corrente. Reproduzir recuperação salva utiliza a base, os índices e os vetores de entrada registrados na execução original, sem novas chamadas de embeddings ou LLM. Hashes e disponibilidade dos artefatos são verificados. Diferenças de versões de bibliotecas são informadas.

Os retornos podem ser baixados em JSON, CSV ou JSONL, e a configuração pode ser exportada separadamente. Os registros preservam a proveniência dos perfis, revisões de índices, parâmetros e vetor da entrada.

Baixar vetores e métricas (ZIP) fornece os vetores numéricos da entrada e dos correspondentes recuperados, organizados por perfil/dimensão em NPZ, com manifesto JSON e métricas CSV. As associações identificam experimento, tópico, categoria, vetor, posição e pontuações de recuperação/fusão. Os arquivos NPZ não exigem pickle. São incluídos os vetores dos retornos e baselines registrados, sem regenerar embeddings.

## Análise automática opcional

A análise começa desmarcada. Quando ativada, o modelo recebe a entrada e os conteúdos originais retornados, classifica pertinência, comenta e atribui prioridade global de utilidade. Conteúdos repetidos entre experimentos são avaliados uma única vez e recebem a mesma anotação.

O padrão é OpenAI gpt-6-luna. Também é possível escolher outro modelo OpenAI, um modelo generativo Ollama local/remoto ou uma API compatível. Credenciais e URLs dos provedores são configuradas no servidor, nunca nos arquivos exportados. Chamadas de API podem ter cobrança.

Cada linha registra classificação, comentário, prioridade, provedor/modelo e hashes de proveniência. Falhas deixam os conteúdos sem classificação, mantêm as recuperações e registram a mensagem de erro. Não se transforma erro em “não pertinente”.

Limites de avaliação: até 20 retornos por experimento, 60 no conjunto, 12.000 caracteres por conteúdo/entrada e 120.000 no total. Conteúdos não são truncados silenciosamente; uma entrada acima dos limites mantém a recuperação e informa que a análise não foi concluída.

## Preparar e importar uma base

Em Preparar base, forneça URLs HTTPS públicas, caminhos autorizados do servidor ou envie PDF, TXT, Markdown, HTML e DOCX. Metadados são opcionais e editáveis, sem planilha obrigatória. O pré-processamento pode usar LLM para perguntas, respostas e resumo; o modo extrativo funciona sem LLM. Revise a prévia e incorpore os documentos antes de gerar os vetores por perfil/categoria. PDFs digitalizados exigem OCR prévio.

Importar base já preparada aceita o JSON estruturado e um NPZ legado opcional. Preserva IDs, textos, metadados, representações e tópicos inativos. A importação não chama LLM. Índices de modelos diferentes ficam separados; um perfil nunca reutiliza coordenadas de outro modelo.

Caminhos locais referem-se ao servidor. AUTORAG_ALLOWED_SOURCE_ROOTS habilita diretórios autorizados; arquivos do computador de quem usa a interface devem ser enviados por upload. URLs privadas/localhost são recusadas para ingestão. Limites existentes: 20 MiB por documento, 25 MiB por requisição e 30 fontes por lote.

## Execução local e acesso

No Windows, abra _autorag_conceito_build/09_abrir_web.bat. Alternativa na pasta da aplicação:

~~~powershell
py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-web.txt
.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8766 --workers 1
~~~

Abra http://127.0.0.1:8766. AUTORAG_PROJECT_DIR permite usar outra pasta de projeto. O servidor Uvicorn não carrega .env.example automaticamente. OPENAI_API_KEY habilita OpenAI; AUTORAG_LLM_OLLAMA_BASE_URL e AUTORAG_LLM_BASE_URL/AUTORAG_LLM_API_KEY configuram os outros serviços generativos.

O token da aplicação autentica sua API; não é a chave do Cloudflare nem do OpenAI. Na interface, ele fica na sessão do navegador. Projetos compartilhados usam a mesma base central e seleção de consulta; esta versão não cria contas individuais ou permissões por usuário.

Na Cloudflare, configurações e comparações fazem parte dos snapshots privados do R2. Operações confirmam persistência antes do sucesso. Os limites existentes do armazenamento são 90 MiB de snapshot compactado, 1 GiB descompactado e 8 MiB por registro de job; excessos são informados. Código, pesos instalados e dados privados permanecem separados na publicação.

## API da nova versão

- GET/POST /api/experiment-configs: configurações nomeadas.
- GET/PUT /api/consultation-config: parâmetros compartilhados da consulta.
- POST /api/comparisons: até três experimentos e análise opcional; responde com job.
- GET /api/comparisons e GET /api/comparisons/{id}: histórico e registros.
- GET /api/comparisons/{id}/export?format=json|csv|jsonl|config|vectors: downloads autenticados.
- POST /api/comparisons/{id}/reproduce: reprodução registrada sem inferência.
- POST /api/search com apenas query: parâmetros salvos ou consulta vetorial padrão.

Configurações enviadas explicitamente por integrações anteriores continuam disponíveis. Os fontes estão preparados para GitHub e Cloudflare; datasets, índices, chaves e registros privados não integram o pacote público.
