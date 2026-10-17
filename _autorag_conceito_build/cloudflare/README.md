# Cloudflare: frontend, backend e estado durável

Este diretório publica o frontend em Workers Static Assets e o backend Python em Cloudflare Containers. Um único container atende o projeto; o bucket R2 privado guarda a base, os índices, os experimentos, as prévias e os registros da fila. Não é necessário um servidor externo nem credenciais S3.

Containers exige Workers Paid, atualmente US$ 5/mês mais consumo. R2 e requisições também seguem a cobrança da conta. Não ative um plano pago automaticamente. Documentação oficial: https://developers.cloudflare.com/containers/platform/pricing/.

## Publicar pelo repositório GitHub

1. Ative Containers pelo plano Workers Paid e habilite a assinatura de R2 na conta Cloudflare. Essas ativações e seus termos são feitos no painel da conta. Em seguida, crie um bucket R2 privado chamado autorag-normativo-state. Se escolher outro nome, ajuste bucket_name em wrangler.jsonc.
2. Conecte o repositório em Workers & Pages > Create application > Import a repository, usando o produto Workers. Use a branch main.
3. Use a raiz de build _autorag_conceito_build, que contém o contexto completo do Dockerfile. O nome do Worker deve ser autorag-normativo.
4. Build command: cd cloudflare && corepack pnpm install --frozen-lockfile && corepack pnpm run build
5. Deploy command: cd cloudflare && python3 deploy.py
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

A imagem inclui Ollama `bge-m3` e `embeddinggemma:300m-qat-q4_0`, encontrados na instalação local auditada, e Sentence Transformers `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. As opções OpenAI e Gemini continuam disponíveis por API. E5 e modelos de geração adicionais não são instalados. Instalar um modelo não altera o perfil ativo nem gera novos índices automaticamente.

`models.lock.json` fixa a versão oficial Ollama 0.40.5, o SHA256 do pacote Linux e os digests completos dos dois manifestos locais. O build baixa os artefatos do GitHub oficial e do registro Ollama; recusa diferenças de digest ou tamanho. Todos os blobs são verificados, incluindo as camadas de licença. Apenas executáveis e bibliotecas CPU entram na imagem final; CUDA, ROCm, MLX e Vulkan são excluídos. Pesos não são enviados ao GitHub nem incorporados ao frontend.

Os modelos Ollama ficam em `/opt/models/ollama`, separados de `/data` e do snapshot R2. Cada reinício começa com os mesmos artefatos presentes na imagem. O supervisor inicia Ollama exclusivamente em `127.0.0.1:11434`, confere seus digests e depois inicia o backend. Sinais de parada são encaminhados aos dois processos. Nenhum perfil é registrado automaticamente.

O MiniLM fica em `/opt/models/sentence-transformers`, localizado pelo fallback `AUTORAG_MODELS_DIR`. Seus 12 arquivos somam 499.561.295 bytes, com revisão `e8f8c211226b894fcb81acc59f3b34ba3efd5f42`, tamanhos e SHA256 fixados no mesmo lock. O build baixa somente os pesos safetensors, os arquivos de tokenização/configuração e o model card originais; não baixa cópias ONNX, TensorFlow ou PyTorch `.bin`. O downloader recusa conteúdo maior ou diferente do artefato fixado e publica cada arquivo por troca atômica. O model card, a licença Apache 2.0 integral e o aviso da distribuição ficam preservados na imagem. Para repetir a instalação em outra pasta: `python -m cloudflare.download_sentence_models --root CAMINHO`.

O runtime usa `torch==2.10.0+cpu`, instalado primeiro pelo índice oficial CPU, `sentence-transformers==5.2.0`, `transformers==4.57.6` e `sentencepiece==0.2.1`. Não são instaladas bibliotecas CUDA. `AUTORAG_SENTENCE_DEVICE=cpu`, `AUTORAG_SENTENCE_BATCH_SIZE=1` e `AUTORAG_SENTENCE_NUM_THREADS=1` limitam o processamento; sem essas variáveis, os defaults locais anteriores são mantidos. O carregamento é exclusivamente local, sem código remoto ou downloads em produção. O MiniLM mantém 384 dimensões nativas, mean pooling e seu limite real de 128 tokens; a preparação e a verificação contra truncamento silencioso permanecem as mesmas.

O container mantém `standard-1`, uma fila e um modelo residente por vez. `AUTORAG_OLLAMA_INPUT_BATCH=1` limita batches de textos, `AUTORAG_OLLAMA_NUM_THREAD=1` limita as threads de inferência e `AUTORAG_OLLAMA_KEEP_ALIVE=0` descarrega o modelo ao concluir a operação. São mantidos os limites de contexto, a preparação dos textos e as dimensões existentes: BGE-M3 com 1024 dimensões e EmbeddingGemma com redução Matryoshka para 256, a partir de 768 dimensões nativas. O processamento continua sujeito ao limite de 4 GiB de RAM; o CI verifica ambos com textos curtos e esse limite, sem afirmar uma medição do pico de memória para entradas de contexto máximo.

As licenças originais estão preservadas nas camadas do modelo e em `/opt/models/ollama/licenses/`, com `NOTICE.txt`. BGE-M3 usa MIT; EmbeddingGemma usa os [termos Gemma](https://ai.google.dev/gemma/terms). O download usa os artefatos Ollama já adotados na instalação local, sem novo acesso ao repositório Google no Hugging Face. Os avisos e restrições da licença continuam aplicáveis ao serviço hospedado.

O workflow de CI verifica tamanho descompactado da imagem menor que 8 GB e executa os três modelos com rede desabilitada, 4 GiB de RAM, sem swap e 0,5 CPU. Esse teste confere todos os hashes, gera embeddings reais e valida dimensões nativas e perfis da aplicação. Ollama é encerrado antes do smoke do MiniLM, que verifica textos curtos de consulta e documento em CPU. Esse teste não mede o pico de memória de contextos máximos ou modelos simultâneos. Não utiliza R2 nem provedores pagos. Para executar a mesma verificação: `docker run --rm --network none --memory 4g --memory-swap 4g --cpus 0.5 --entrypoint python autorag-cloudflare:ci -m cloudflare.verify_models --smoke`.

## Limites desta primeira versão

- Uma instância compartilhada; sem isolamento entre vários órgãos ou usuários.
- Snapshot de até 90 MiB compactados e 1 GiB descompactado. Cada resultado de operação pode ter até 8 MiB.
- Os dois modelos Ollama e o MiniLM estão fixados na imagem. Novos modelos requerem atualizar o manifesto e reconstruir a imagem; downloads feitos apenas durante uma sessão não são persistidos. Os perfis de API OpenAI/Gemini funcionam com os secrets de runtime correspondentes.
- Interrupções durante uma operação podem exigir reenvio. Publicações da aplicação devem acontecer com a fila ociosa.
- O limite max_instances=1 contém a escala desta implantação; consumo de CPU, memória, disco, R2 e duração dos trabalhos continua sendo cobrado conforme uso.

## Envio da imagem em blocos

O comando `python3 deploy.py` usa o Wrangler com o uploader oficial regctl 0.11.6, fixado por SHA256. Ele envia as camadas ao registro Cloudflare em blocos de 8 MiB, com uma requisição por vez, para evitar timeouts no envio dos pesos. A autenticação continua sendo a configuração Docker criada pelo Wrangler com o token de build já autorizado. Não são criadas novas credenciais.

O ajuste atua somente no ambiente Linux AMD64 de build. Os pesos, seus digests, o formato da imagem e o runtime permanecem os mesmos. Antes de exportar o TAR, o wrapper confere espaço livre para a imagem e uma reserva. Se faltar espaço, executa `docker builder prune --all --force` para remover somente cache de build não utilizado e repete a conferência; imagens finais, containers, volumes e credenciais são preservados. Se ainda faltar espaço, o envio é interrompido com os tamanhos necessários no diagnóstico. O TAR temporário da imagem é removido após o envio; regctl e suas configurações ficam em diretórios temporários fora do contexto da aplicação. Para deploy local direto, o comando Wrangler convencional continua disponível.
