# Roteiro resumido de implementação

Data: 30/09/2026. Situação: planejamento; implementação e instalação ainda não realizadas.

Objetivo: transformar a pipeline em um ambiente de exploração dos embeddings dos conteúdos, com geração independente por modelo/categoria, buscas reproduzíveis e preparação para cálculos e clusterização. Detalhamento: [PLANO_EMBEDDINGS.md](PLANO_EMBEDDINGS.md).

## Alterações principais

- Vetores numéricos exclusivamente em **NPZ**, incluindo conteúdos, entradas de consulta e derivados. JSON guarda textos, hashes, configurações, estados, referências e resultados escalares.
- Nova estrutura JSON com perfis dos modelos e metadados das unidades de embedding, migrando o objeto único `embedding` para `embeddings.perfis`.
- Oito categorias independentes: total sem palavras-chave; total com palavras-chave; resumo; conteúdo-fonte; perguntas sugeridas; resposta-síntese; palavras-chave; metadados.
- Geração/atualização/regeneração escolhidas por **modelo × categoria × escopo da base**, sem exigir os outros vetores. Seleção de categorias para geração e para recuperação são controles distintos.
- Preservação das gerações anteriores: atualização da base marca divergências, mas não apaga vetores nem inicia geração automática. Publicar substitutos somente depois de validá-los.
- Busca por pares modelo/categoria, com escores individuais, médias das combinações, limiar, top-k e grupos por faixa de escore associados à consulta.
- Armazenamento permanente das entradas e execuções para reexplorar resultados sem regerar embeddings.
- Opção de chave direta no código e ponto de integração para provisão segura. Segurança/exposição de chaves fora do escopo atual.

## Etapas de implementação

| Etapa | Trabalho | Critério de conclusão |
| --- | --- | --- |
| **1. Consolidar a arquitetura** | Extrair contratos compartilhados dos scripts atuais: provedores, composição de textos, armazenamento, geração, recuperação, entradas e experimentos. Separar configuração dos modelos de configuração da busca. | Os comandos usam os mesmos contratos; cada modelo possui identificação/configuração rastreável. |
| **2. Reestruturar JSON e NPZ** | Migrar a base textual para schema 3.0, criar schema de índice 2.0, IDs por unidade e NPZ por modelo/categoria. Implementar referências, hashes, revisões e publicação validada. | Ler/migrar o legado sem perder dados, sem coordenadas vetoriais em JSON; atualizar uma categoria preserva as demais. |
| **3. Implementar os provedores e a configuração de chaves** | Manter OpenAI; acrescentar Ollama, Sentence Transformers local e Gemini com modalidade gratuita disponível. Descobrir modelos instalados e respeitar dimensões/preparação/limites próprios. Centralizar chave direta e alternativa de provisão. | Geração e consulta usam o mesmo perfil; execução local funciona sem chave; contratos testáveis com provedores simulados. |
| **4. Gerar independentemente os oito tipos** | Construir cada texto, tratar campos vazios/textos longos, gerar faltantes, atualizar alterados ou forçar regeneração para os pares/IDs escolhidos. Preservar o total sem palavras-chave como referência. | Gerar só resumo ou só palavras-chave em qualquer modelo, manter outros pares intactos e retomar após falha sem perder a geração anterior. |
| **5. Persistir entradas e experimentos** | Salvar texto original e embedding de consulta por modelo; reutilizar gerações compatíveis. Registrar execução, parâmetros, revisões, textos recuperados, referências e distâncias. | Reabrir uma busca e localizar seus vetores exatos sem chamar o modelo novamente, mesmo após atualização da base. |
| **6. Adaptar busca e exploração por escore** | Implementar seleção modelo/categoria, visões de vetores e tópicos, média aritmética, limiar/top-k, cobertura e faixas de escore. Preservar o universo completo dos escores avaliados. | Escores individuais e médias conferem; limites têm escopo explícito; faixas podem ser redefinidas sobre resultados salvos. |
| **7. Integrar menus, exportação e testes offline** | Atualizar atalhos Windows, parâmetros CLI, inventário modelo/categoria, listagem/reabertura de experimentos e exportação de conjuntos. Atualizar dependências e documentação. | Executar o fluxo com dados de teste e provedores simulados; validar migração, independência, persistência e ranking antes da instalação dos modelos. |
| **8. Instalar modelos e validar a operação real** | Instalar/configurar Ollama no D:, baixar BGE-M3 e EmbeddingGemma Q4 com perfil de 256 dimensões. Disponibilizar E5 Small de 384 dimensões como alternativa local. Executar piloto em português e depois somente as gerações escolhidas. | Modelos maior/leve produzem vetores consistentes; geração, consulta, armazenamento, faixas e comparação funcionam com dados reais. |
| **9. Ampliar a bancada analítica** | Após a exploração inicial, acrescentar comparação de pares/conjuntos, operações vetoriais, matrizes de distâncias, centroides, projeções e clusterização por vários métodos. | Resultados derivados têm NPZ próprio e metadados de origem, parâmetros e revisões; operações não alteram a base fonte automaticamente. |

## Configuração das chaves

Prever no código um módulo central `embedding_credentials.py`, com os pontos de alimentação direta abaixo:

```python
OPENAI_API_KEY_DIRETA = ""
GEMINI_API_KEY_DIRETA = ""
```

Esses campos são locais previstos para preenchimento; o roteiro não copia valores das chaves existentes. A função `obter_chave(provedor)` deve aceitar a configuração direta e preservar as alternativas já existentes de variável de ambiente/configuração. Reservar um resolvedor para entrada protegida ou outro mecanismo de provisão segura, sem alterar os demais componentes quando essa opção for adotada. Documentar a precedência e manter o funcionamento atual durante a transição. Nenhuma configuração de chave será necessária para Ollama ou outros modelos locais.

## Fluxo esperado da primeira entrega

1. Abrir o inventário de modelos/categorias e selecionar exatamente quais pares gerar ou atualizar.
2. Simular a operação, executar e publicar as novas revisões dos NPZ selecionados.
3. Informar o texto de entrada e salvar seus embeddings nos modelos escolhidos.
4. Selecionar as categorias disponíveis em cada modelo, a visão de resultado e os limites da busca.
5. Exibir vetores individuais por modelo/categoria ou tópicos com as médias dos vetores selecionados.
6. Salvar execução, textos, referências, distâncias e o universo de escores calculados.
7. Separar conjuntos por faixa de escore, reabrir resultados e exportá-los para estudo posterior.

Palavras-chave isoladas e tópico total com palavras-chave só participarão da recuperação quando selecionados. Comparações entre modelos terão avaliações separadas, usando o embedding da consulta correto em cada modelo.

## Organização prevista dos dados

| Dados | Armazenamento |
| --- | --- |
| Conteúdos e metadados dos embeddings | Base JSON reestruturada |
| Vetores dos conteúdos | `indices/<perfil>/<categoria>/<revisao>/vetores.npz` |
| Texto original de entrada e suas referências | `estudos/entradas/<entrada>/entrada.json` |
| Vetores da entrada, por modelo/configuração/papel | NPZ versionados dentro da pasta da entrada |
| Resultados, textos recuperados, escores/distâncias e parâmetros | `estudos/execucoes/<execucao>/resultado.json` |
| Universo completo de escores calculados | `estudos/execucoes/<execucao>/escores.npz`, com referências/manifesto JSON |
| Grupos por modelo/categoria/faixa | JSON de conjuntos com IDs/referências ligados à execução |
| Vetores derivados, centroides e projeções futuras | NPZ próprios em `estudos/analises/`, com metadados JSON |

Na visão de tópicos, ordenar pela média aritmética dos escores selecionados e retornar até top-k resultados que atinjam o limiar. Na visão de vetores, aplicar os limites ao escore individual de cada grupo modelo/categoria. Vetores desatualizados permanecem armazenados e identificados, mas não são associados silenciosamente ao conteúdo novo.

## Validação e entrega

A primeira entrega deve comprovar: migração sem perda, geração independente, preservação após falhas/atualizações, combinação correta, médias/distâncias corretas, limiar/top-k, faixas sem duplicação nas fronteiras e reprodução de execuções antigas. O piloto comparará os modelos em português, inicialmente pelo total sem palavras-chave, medindo qualidade de recuperação, latência, geração e tamanho dos índices.

A estrutura para manipulação e clusterização entra desde o início; a implementação completa de K-Means, métodos hierárquicos, DBSCAN/HDBSCAN, espectral e projeções integra a etapa 9, após explorar os resultados de recuperação. Critérios e referências técnicas desses métodos estão no plano detalhado.
