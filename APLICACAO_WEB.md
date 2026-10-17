# AutoRAG Web — primeira versão

A aplicação web fica em `_autorag_conceito_build`. O aplicativo desktop, a CLI, os oito tipos de vetores, perfis separados por modelo, índices versionados e os experimentos anteriores continuam disponíveis.

## Executar localmente

No Windows, abra `_autorag_conceito_build/09_abrir_web.bat`. Alternativa, dentro da pasta da aplicação:

```powershell
py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-web.txt
.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8766 --workers 1
```

Abra http://127.0.0.1:8766. O lançador reutiliza o ambiente virtual existente quando disponível. A porta local padrão é 8766, alterável por `AUTORAG_PORT` no lançador. O serviço usa os dados e índices existentes da pasta da aplicação. Para um projeto vazio ou outro órgão, defina `AUTORAG_PROJECT_DIR` com uma pasta separada antes de iniciar. Não copie chaves para o navegador. As variáveis de configuração do backend precisam estar no ambiente do processo; o arquivo `.env.example` é referência e não é carregado automaticamente pelo Uvicorn. O resolvedor existente continua aceitando chaves de modelos em `.env.local`.

## Preparar uma base

1. Em **Preparar base**, informe URLs HTTPS públicas, caminhos do servidor autorizados ou envie PDF, TXT, Markdown, HTML e DOCX. Metadados são opcionais e editáveis no formulário: título, órgão, eixo, idioma e campos extras JSON. Não há dependência de planilha separada.
2. Gere a prévia e revise tópicos, fonte e metadados. A extração inicial preserva o texto, produz resumo extrativo e pergunta em template; os documentos registram essas características e a necessidade de revisão. Essa etapa não chama um LLM. Documentos PDF digitalizados precisam de OCR prévio. A API Python de ingestão aceita um callback `enrich` para representações geradas por modelo e revisadas.
3. Incorpore a prévia. A API usa seu `preview_id` para gravar exatamente os documentos revisados, sem baixar novamente a fonte. Fontes repetidas no mesmo lote são rejeitadas; outra incorporação da mesma fonte atualiza o documento e conserva a versão anterior da base.
4. Escolha/cadastre um perfil no laboratório e gere as categorias. Geração é explícita e usa o provedor configurado. Modelos OpenAI/Gemini exigem chave no servidor; Ollama exige o modelo e endpoint disponíveis no servidor Python. Os vetores não são gerados automaticamente ao importar documentos. Na base local existente, só a categoria `topico_total_sem_palavras_chave` tem índice; gere as quatro categorias do novo padrão antes da primeira consulta semântica.

Um caminho como `C:\normas\lei.pdf` se refere ao computador do backend. Para habilitar caminhos locais, configure `AUTORAG_ALLOWED_SOURCE_ROOTS` com diretórios autorizados (` ; ` no Windows; ` : ` no Linux). Para arquivos do computador de quem está navegando, use upload. URLs privadas/localhost são recusadas. Limites: 20 MiB por documento, 25 MiB por requisição HTTP e 30 fontes por lote. Órgão/idioma/vigência não são inferidos quando ausentes.

## Recuperação padrão

A consulta e todos os vetores atuais das categorias selecionadas do **mesmo perfil** participam de um novo ajuste UMAP → HDBSCAN. Padrão: `perguntas_sugeridas`, `resposta_sintese`, `resumo`, `conteudo_fonte`. Perguntas e respostas são categorias independentes preservadas do modelo anterior. Qualquer combinação das oito categorias pode ser usada.

No agrupamento que contém a consulta:

1. Cada tópico recebe a contagem de **categorias distintas** presentes. Vários segmentos da mesma categoria contam uma vez.
2. São selecionados até **seis tópicos**, priorizando a maior contagem. O desempate usa a menor distância entre a consulta e um vetor membro do tópico.
3. O conjunto selecionado é apresentado em ordem crescente dessa distância, calculada no espaço original normalizado L2. UMAP serve para agrupamento e projeção, sem substituir a distância original no ranking.

A coluna de comparação recupera tópicos usando exclusivamente a distância dos vetores de `conteudo_fonte`, independentemente do agrupamento. Se essa categoria ainda não foi gerada, a tela informa que a comparação está indisponível. Categorias ausentes são sinalizadas; vetores inativos/desatualizados são excluídos.

Consulta classificada como ruído (`-1`) ou base insuficiente não recebe agrupamento artificial: a recuperação semântica fica vazia e a comparação direta permanece disponível. O resultado registra parâmetros efetivos, sementes, referências dos índices e do vetor de consulta e revisão da base. A projeção mostra as duas primeiras coordenadas da redução usada; quando a redução tem mais dimensões, o desenho 2D não representa todas as distâncias.

## Laboratório

O laboratório permite escolher perfil, categorias, métrica cosseno/euclidiana, top-k e parâmetros UMAP/HDBSCAN, com dendrograma opcional da árvore real HDBSCAN em SVG e JSON. O modo de recuperação tradicional mantém médias por tópico, vetores individuais, intervalos de similaridade e cobertura parcial da implementação existente, via API e aplicativo desktop. Modelos distintos continuam produzindo rankings separados.

A consulta e os resultados podem ser exportados em JSON. O histórico web guarda execuções em `estudos/web`; o histórico original permanece em `estudos/execucoes`. As exportações ZIP completas e análises originais continuam na CLI/desktop. Operações de ingestão, geração e busca usam uma fila Python e polling de jobs; reiniciar o serviço marca jobs interrompidos para reenvio.

## Publicação completa no Cloudflare

Os fontes estão em [flaviodalossa/CDIA---autorag](https://github.com/flaviodalossa/CDIA---autorag). A configuração em `cloudflare/` hospeda frontend e backend no Cloudflare usando Workers, Containers e R2. Consulte [o guia de Cloudflare](./_autorag_conceito_build/cloudflare/README.md) para autenticação, recursos e publicação pelo GitHub integrado a Workers Builds.

A alternativa abaixo usa Pages com um backend Python hospedado separadamente.

## Alternativa: Cloudflare Pages

Para gerar um ZIP somente com os fontes, execute `python scripts/package_source.py` dentro da pasta da aplicação. O arquivo `autorag-web-v0.1.zip` será criado na pasta anterior.

O repositório usa uma lista de inclusão em `.gitignore`: datasets, índices, chaves, ambientes virtuais e históricos locais ficam fora da publicação. Não há dados normativos embutidos no frontend. A pasta da aplicação contém `Dockerfile`, `compose.yaml`, `wrangler.jsonc` e testes/CI.

1. Crie um repositório GitHub e envie os fontes desta pasta. Na raiz:

```powershell
git init
git add .
git status --short
git commit -m "Primeira versão web AutoRAG"
git branch -M main
git remote add origin https://github.com/SEU_USUARIO/SEU_REPOSITORIO.git
git push -u origin main
```

2. Hospede o **backend Python** em um serviço que execute contêineres/Python, com disco persistente montado em `/data`. Contexto de build: `_autorag_conceito_build`; Dockerfile: `Dockerfile`. Configure `AUTORAG_ENV=production`, `AUTORAG_PROJECT_DIR=/data`, `AUTORAG_API_TOKEN` com pelo menos 32 caracteres e as chaves dos provedores usados. Use **um worker e uma réplica** nesta primeira versão: fila e trava são locais. Preserve o volume no redeploy. A rota `/health` serve à verificação de disponibilidade.
3. Crie um projeto **Cloudflare Pages**, conecte o repositório GitHub e configure:

| Campo | Valor |
|---|---|
| Framework | None |
| Root directory | `_autorag_conceito_build` |
| Build command | deixar vazio |
| Build output directory | `web` |
| Variável de ambiente | `BACKEND_URL=https://ORIGEM_DO_BACKEND` |

`BACKEND_URL` deve ser apenas a origem HTTPS, sem caminho, credenciais ou parâmetros. `web/_worker.js` encaminha `/api/*` ao backend e serve os arquivos estáticos pelo binding `ASSETS`. O visitante informa o token de acesso nas configurações da aplicação; ele fica em sessionStorage da aba. O proxy encaminha esse token ao backend. Chaves de provedores permanecem no servidor. Não coloque segredos em `wrangler.jsonc` ou arquivos públicos.

4. Faça deploy pelo GitHub integrado ao Pages. Abra o site, informe o token e confirme o estado da base. Importe documentos e gere embeddings no projeto persistente do backend. Para migrar a base atual, transfira `dados`, `indices` e, se necessário, `estudos` diretamente ao volume privado, mantendo caminhos relativos.

O processamento científico e os modelos locais rodam no backend Python. Na alternativa Pages, publicar apenas os arquivos estáticos exige configurar o backend separado. A instalação principal usa Workers + Containers + R2, com frontend e backend no mesmo endereço Cloudflare.

As instruções seguem a documentação de [integração Git do Cloudflare Pages](https://developers.cloudflare.com/pages/configuration/git-integration/) e [Functions em modo avançado](https://developers.cloudflare.com/pages/functions/advanced-mode/). O agrupamento segue as APIs oficiais de [UMAP para clustering](https://umap-learn.readthedocs.io/en/latest/clustering.html) e [HDBSCAN](https://hdbscan.readthedocs.io/en/latest/api.html).

## Validação

Dentro da pasta da aplicação:

```powershell
python -m unittest discover -s tests -p 'test_*.py' -v
npm run check
npm test
```

Os testes usam documentos/vetores sintéticos temporários e provedores falsos; não consomem APIs pagas. A integração científica usa UMAP/HDBSCAN reais quando as dependências de `requirements-web.txt` estão instaladas. A CI repete esses testes no GitHub. Os JSONs normativos e embeddings locais não são alterados pelos testes.

## LLM para elaboração dos tópicos

A tela Preparar base usa OpenAI `gpt-6-luna` por padrão. Cadastre `OPENAI_API_KEY` como segredo do servidor (no Cloudflare, Runtime variables and secrets → Production → Secret). A chave permanece no backend. A presença da chave não comprova saldo, validade nem acesso ao modelo; isso é verificado ao processar uma prévia. O aplicativo também oferece outros modelos OpenAI, Ollama e APIs compatíveis com saídas JSON estruturadas. Não há troca automática de provedor se o modelo escolhido falhar.

No Ollama, informe um modelo generativo previamente instalado no serviço definido por `AUTORAG_LLM_OLLAMA_BASE_URL` (padrão: Ollama local). Essa configuração separada preserva os embeddings locais ao usar um LLM remoto. BGE-M3 e EmbeddingGemma geram embeddings e não elaboram tópicos. Para vLLM, LM Studio ou outro serviço compatível, configure `AUTORAG_LLM_BASE_URL` (incluindo `/v1` quando aplicável) e, se necessário, `AUTORAG_LLM_API_KEY`. URLs de serviços e credenciais são configuradas somente no servidor; serviços remotos exigem HTTPS.

Cada trecho preserva o texto original e seus offsets. O LLM produz resumo, perguntas, resposta, palavras-chave e referências normativas; sua identificação e a versão do prompt ficam registradas na prévia. O usuário revisa antes de confirmar a importação. A opção Extrativo é explícita e não faz elaboração generativa. Confirmar uma prévia já criada não chama o LLM novamente.

No laboratório, o catálogo informa configuração e disponibilidade dos embeddings, permite cadastrar os perfis e testar os modelos locais com entradas sintéticas. Testes automáticos de embeddings não chamam APIs pagas. Na imagem Cloudflare ficam os mesmos pesos locais: BGE-M3 (1.024 dimensões) e EmbeddingGemma 300M QAT Q4_0 (768 dimensões nativas, redução configurada para 256). Os pesos não são enviados ao GitHub nem ao snapshot R2.
