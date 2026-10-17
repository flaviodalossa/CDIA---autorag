# Cloudflare: frontend, backend e estado durável

Este diretório publica o frontend em Workers Static Assets e o backend Python em Cloudflare Containers. Um único container atende o projeto; o bucket R2 privado guarda a base, os índices, os experimentos, as prévias e os registros da fila. Não é necessário um servidor externo nem credenciais S3.

Containers exige Workers Paid, atualmente US$ 5/mês mais consumo. R2 e requisições também seguem a cobrança da conta. Não ative um plano pago automaticamente. Documentação oficial: https://developers.cloudflare.com/containers/platform/pricing/.

## Publicar pelo repositório GitHub

1. Ative Containers pelo plano Workers Paid e habilite a assinatura de R2 na conta Cloudflare. Essas ativações e seus termos são feitos no painel da conta. Em seguida, crie um bucket R2 privado chamado autorag-normativo-state. Se escolher outro nome, ajuste bucket_name em wrangler.jsonc.
2. Conecte o repositório em Workers & Pages > Create application > Import a repository, usando o produto Workers. Use a branch main.
3. Use a raiz de build _autorag_conceito_build, que contém o contexto completo do Dockerfile. O nome do Worker deve ser autorag-normativo.
4. Build command: cd cloudflare && corepack pnpm install --frozen-lockfile && corepack pnpm run build
5. Deploy command: cd cloudflare && corepack pnpm exec wrangler deploy
6. Cadastre AUTORAG_API_TOKEN como segredo de runtime do Worker, com pelo menos 32 caracteres aleatórios. Use esse mesmo token nas configurações da interface. Cadastre OPENAI_API_KEY ou GEMINI_API_KEY como segredo de runtime se usar esses provedores. Não coloque as chaves em variáveis públicas, no código ou nos assets.
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

## Limites desta primeira versão

- Uma instância compartilhada; sem isolamento entre vários órgãos ou usuários.
- Snapshot de até 90 MiB compactados e 1 GiB descompactado. Cada resultado de operação pode ter até 8 MiB.
- Modelos baixados e caches locais são descartáveis. O container não oferece um serviço Ollama local. Os perfis de API OpenAI/Gemini funcionam com os secrets de runtime correspondentes.
- Interrupções durante uma operação podem exigir reenvio. Publicações da aplicação devem acontecer com a fila ociosa.
- O limite max_instances=1 contém a escala desta implantação; consumo de CPU, memória, disco, R2 e duração dos trabalhos continua sendo cobrado conforme uso.
