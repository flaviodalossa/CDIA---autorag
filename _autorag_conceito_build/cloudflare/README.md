# Cloudflare: frontend, backend e estado durável

Este diretório publica o frontend em Workers Static Assets e o backend Python em Cloudflare Containers. Um único container atende o projeto; o bucket R2 privado guarda a base, os índices, os experimentos, as prévias e os registros da fila. Não é necessário um servidor externo nem credenciais S3.

Containers exige Workers Paid, atualmente US$ 5/mês mais consumo. R2 e requisições também seguem a cobrança da conta. Não ative um plano pago automaticamente. Documentação oficial: https://developers.cloudflare.com/containers/platform/pricing/.

## Publicar pelo repositório GitHub

1. Ative Containers pelo plano Workers Paid e habilite a assinatura de R2 na conta Cloudflare. Essas ativações e seus termos são feitos no painel da conta. Em seguida, crie um bucket R2 privado chamado autorag-normativo-state. Se escolher outro nome, ajuste bucket_name em wrangler.jsonc.
2. Conecte o repositório em Workers & Pages > Create application > Import a repository, usando o produto Workers. Use a branch main.
3. Use a raiz de build _autorag_conceito_build, que contém o contexto completo do Dockerfile. O nome do Worker deve ser autorag-normativo.
4. Build command: cd cloudflare && corepack pnpm install --frozen-lockfile && corepack pnpm run build
5. Deploy command: cd cloudflare && corepack pnpm exec wrangler deploy
6. Cadastre AUTORAG_API_TOKEN como segredo de runtime do Worker, com pelo menos 32 caracteres aleatórios. Use esse mesmo token nas configurações da interface. O nome existente api_token_autorag também é aceito como alias; se ambos estiverem cadastrados, AUTORAG_API_TOKEN prevalece. O Worker repassa o valor resolvido ao backend com o nome canônico. Cadastre OPENAI_API_KEY ou GEMINI_API_KEY como segredo de runtime se usar esses provedores. Não coloque as chaves em variáveis públicas, no código ou nos assets.
7. Execute o build de produção. Workers Builds consegue construir o Dockerfile; Docker local é necessário somente para desenvolvimento/deploy local.
8. Aguarde a primeira provisão do container. Abra a URL workers.dev e verifique Configurações > Conexão. Uma resposta autenticada de /api/status confirma que o backend está atendendo.

Documentação do build de containers: https://developers.cloudflare.com/containers/guides/deploy/.

## Desenvolvimento e validação

No diretório cloudflare:

- corepack pnpm install --frozen-lockfile
- corepack pnpm run check
- corepack pnpm test
- corepack pnpm run build
- corepack pnpm run dev — exige Docker em execução

Na raiz _autorag_conceito_build, com as dependências Python do projeto: python -m unittest discover -s cloudflare/tests -p 'test_*.py' -v.

Use .dev.vars apenas para segredos locais. Esse arquivo, node_modules, dist e .wrangler são ignorados pelo Git. O arquivo pnpm-lock.yaml fixa as dependências JavaScript do deploy.

## Persistência e suspensão

O disco de Containers é efêmero. O adapter restaura o R2 antes de construir a API. Uma operação recebe HTTP 202 apenas depois de seu registro queued estar confirmado no R2. Ao terminar, o adapter publica um snapshot consistente da base e dos índices antes de confirmar succeeded e finished_at no registro durável. Esse trabalho ocorre no servidor mesmo se o navegador fechar.

Os registros da fila ficam separados do snapshot para permitir novas submissões durante o processamento. Se o container for interrompido, a aplicação restaura esses registros e marca operações incompletas como falhas, permitindo reenvio. Operações não são retomadas automaticamente. Falhas de persistência não são apresentadas como sucesso.

Após dez minutos sem requisições, o Durable Object verifica a fila. Enquanto houver processamento, mantém o container ativo. Antes de suspender, conclui um checkpoint, bloqueando a entrada de novas requisições durante a decisão de parada. Os dados ficam no bucket; as chaves de modelos ficam apenas nos secrets.

O hostname http://storage.autorag funciona somente no handler de saída do container e usa o binding STATE do Worker. Nenhuma rota pública expõe o bucket ou /internal/*. Guia oficial: https://developers.cloudflare.com/containers/configuration/workers-connections/.

## Modelos instalados na imagem

A imagem inclui somente os dois modelos encontrados na instalação local auditada: Ollama `bge-m3` e `embeddinggemma:300m-qat-q4_0`. As opções OpenAI e Gemini continuam disponíveis por API. E5, MiniLM, Sentence Transformers, PyTorch e modelos de geração adicionais não são instalados.

`models.lock.json` fixa a versão oficial Ollama 0.40.5, o SHA256 do pacote Linux e os digests completos dos dois manifestos locais. O build baixa os artefatos do GitHub oficial e do registro Ollama; recusa diferenças de digest ou tamanho. Todos os blobs são verificados, incluindo as camadas de licença. Apenas executáveis e bibliotecas CPU entram na imagem final; CUDA, ROCm, MLX e Vulkan são excluídos. Pesos não são enviados ao GitHub nem incorporados ao frontend.

Os modelos ficam em `/opt/models/ollama`, separados de `/data` e do snapshot R2. Cada reinício começa com os mesmos artefatos presentes na imagem. O supervisor inicia Ollama exclusivamente em `127.0.0.1:11434`, confere seus digests e depois inicia o backend. Sinais de parada são encaminhados aos dois processos. Nenhum perfil é registrado automaticamente.

O container mantém `standard-1`, uma fila e um modelo residente por vez. `AUTORAG_OLLAMA_INPUT_BATCH=1` limita batches de textos, `AUTORAG_OLLAMA_NUM_THREAD=1` limita as threads de inferência e `AUTORAG_OLLAMA_KEEP_ALIVE=0` descarrega o modelo ao concluir a operação. São mantidos os limites de contexto, a preparação dos textos e as dimensões existentes: BGE-M3 com 1024 dimensões e EmbeddingGemma com redução Matryoshka para 256, a partir de 768 dimensões nativas. O processamento continua sujeito ao limite de 4 GiB de RAM; o CI verifica ambos com textos curtos e esse limite, sem afirmar uma medição do pico de memória para entradas de contexto máximo.

As licenças originais estão preservadas nas camadas do modelo e em `/opt/models/ollama/licenses/`, com `NOTICE.txt`. BGE-M3 usa MIT; EmbeddingGemma usa os [termos Gemma](https://ai.google.dev/gemma/terms). O download usa os artefatos Ollama já adotados na instalação local, sem novo acesso ao repositório Google no Hugging Face. Os avisos e restrições da licença continuam aplicáveis ao serviço hospedado.

O workflow de CI verifica tamanho descompactado da imagem menor que 8 GB e executa os modelos com rede desabilitada, 4 GiB de RAM, sem swap e 0,5 CPU. Esse teste confere todos os hashes, gera embeddings reais de ambos e valida dimensões nativas e perfis da aplicação. Não utiliza R2 nem provedores pagos. Para executar a mesma verificação: `docker run --rm --network none --memory 4g --memory-swap 4g --cpus 0.5 --entrypoint python autorag-cloudflare:ci -m cloudflare.verify_models --smoke`.

## Limites desta primeira versão

- Uma instância compartilhada; sem isolamento entre vários órgãos ou usuários.
- Snapshot de até 90 MiB compactados e 1 GiB descompactado. Cada resultado de operação pode ter até 8 MiB.
- Os dois modelos Ollama estão fixados na imagem. Novos modelos requerem atualizar o manifesto e reconstruir a imagem; downloads feitos apenas durante uma sessão não são persistidos. Os perfis de API OpenAI/Gemini funcionam com os secrets de runtime correspondentes.
- Interrupções durante uma operação podem exigir reenvio. Publicações da aplicação devem acontecer com a fila ociosa.
- O limite max_instances=1 contém a escala desta implantação; consumo de CPU, memória, disco, R2 e duração dos trabalhos continua sendo cobrado conforme uso.
