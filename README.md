# AutoRAG — aplicação web e pipeline incremental

A primeira versão web está em `_autorag_conceito_build`. Consulte [APLICACAO_WEB.md](APLICACAO_WEB.md) para execução, ingestão genérica, busca UMAP/HDBSCAN e publicação GitHub → Cloudflare. O pipeline original está documentado abaixo. No repositório publicado, os comandos Python do pipeline devem ser executados dentro de `_autorag_conceito_build`, que reúne os fontes preservados.

# Pipeline incremental — Assistente de Ética em Pesquisa

O programa mantém um único JSON cumulativo e separa dois processos:

1. `topicos`: importa os metadados na ordem da planilha, obtém os documentos,
   extrai o texto e usa a API da OpenAI para decompô-los em tópicos;
2. `embeddings`: gera vetores somente para tópicos novos ou para uma lista
   explícita de IDs.

O JSON é salvo atomicamente depois de cada documento ou lote de embeddings. Uma
interrupção não destrói o que já foi concluído.

## Instalação no Windows

Para o primeiro uso, execute os arquivos na seguinte ordem:

1. `01_configurar_ambiente.bat`, para usar o Python existente, criar ou
   reutilizar `.venv`, conferir as dependências, instalar somente o necessário
   e abrir um terminal com o ambiente ativado;
2. `02_executar_piloto.bat`, para solicitar a chave e processar somente
   `despacho_1_2026`;
3. `04_validar_resultado.bat`, caso queira repetir a validação.

O arquivo `03_abrir_terminal_ambiente.bat` abre um terminal com o ambiente
virtual ativado. Os `.bat` sempre entram automaticamente na pasta do projeto,
independentemente da pasta em que o PowerShell estava aberto.

O primeiro arquivo não instala o Python e não acessa a API. O `pip` compara o
conteúdo de `requirements.txt` com o ambiente virtual e reutiliza os pacotes
que já atendem às versões solicitadas.

### Alternativa manual

No PowerShell, dentro desta pasta:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Depois da instalação manual, use `02_executar_piloto.bat`, que solicita a chave
em campo protegido. Não digite a chave diretamente em comandos, nem a grave
dentro do script ou do JSON.

Se uma chave aparecer em captura de tela, repositório ou mensagem, revogue-a e
gere outra. O piloto solicita a chave em campo protegido e a remove do ambiente
ao terminar.

## 1. Importar metadados e gerar tópicos

```powershell
python pipeline_etica.py topicos `
  --metadados "D:\projetoCDIA\bases\metadados.xlsx" `
  --base-json "D:\projetoCDIA\bases\base_etica.json" `
  --documentos-dir "D:\projetoCDIA\bases\documentos"
```

As colunas são normalizadas para caixa baixa, sem acentos e com `_`. São
obrigatórias:

```text
caminho_arquivo, url, descricao_documento, tipo_documento,
identificacao_documento, data_documento, eixo, inicio_vigencia, fim_vigencia
```

O programa também aceita `data_download` ou a grafia anterior `data_dowload`.
CSV/TSV e XLS/XLSX são aceitos. A ordem das linhas é registrada em
`ordem_documentos` e em `ordem_documento`.

Quando a célula `eixo` contém mais de um valor separado por `;`, o documento é
armazenado uma única vez em `documentos`. Cada entrada de `eixos` referencia o
mesmo `documento_id` por meio de `documentos_ids`, sem duplicar tópicos ou
embeddings.

Para verificar apenas a importação, sem acessar documentos nem a API:

```powershell
python pipeline_etica.py topicos `
  --metadados "D:\projetoCDIA\bases\metadados.xlsx" `
  --base-json "D:\projetoCDIA\bases\base_etica.json" `
  --somente-metadados
```

Para fazer um piloto com apenas um documento, sem processar os demais:

```powershell
python pipeline_etica.py topicos `
  --metadados "D:\projetoCDIA\bases\metadados_atualizada.xlsx" `
  --base-json "D:\projetoCDIA\bases\base_etica.json" `
  --documentos-dir "D:\projetoCDIA\bases\documentos" `
  --somente-documentos-ids despacho_1_2026
```

O filtro também aceita `--somente-documentos-ids-arquivo piloto.txt`. Ele é
diferente de `--reprocessar-ids`: o primeiro limita quais documentos são
processados; o segundo força nova extração de documentos cujo arquivo não mudou.

Um documento cujo hash não mudou é ignorado. Para forçar nova decomposição:

```powershell
python pipeline_etica.py topicos `
  --metadados "D:\projetoCDIA\bases\metadados.xlsx" `
  --base-json "D:\projetoCDIA\bases\base_etica.json" `
  --reprocessar-ids decreto_12651_2025 lei_14874_2024
```

Também se pode usar `--reprocessar-ids-arquivo documentos.txt`.

## Fontes aceitas

A resolução segue esta prioridade:

1. arquivo local existente em `caminho_arquivo`;
2. link do Google Drive em `caminho_arquivo`, baixado para a pasta de documentos;
3. URL comum em `caminho_arquivo`;
4. campo `url` como alternativa.

PDFs precisam conter texto extraível. Se forem digitalizações, o programa marca
o documento com erro e solicita OCR, sem inventar conteúdo. Páginas HTML são
baixadas e limpas antes da decomposição.

## 2. Gerar embeddings

Sem lista de IDs, somente tópicos novos com status `pendente` são processados:

```powershell
python pipeline_etica.py embeddings `
  --base-json "D:\projetoCDIA\bases\base_etica.json"
```

Quando um tópico muda, ele fica `desatualizado`. A atualização ocorre somente
quando seu ID é fornecido:

```powershell
python pipeline_etica.py embeddings `
  --base-json "D:\projetoCDIA\bases\base_etica.json" `
  --ids tp_decreto_12651_2025_art_7_consentimento
```

Ou por arquivo, com um ID por linha (vírgulas e ponto e vírgula também são
aceitos):

```powershell
python pipeline_etica.py embeddings `
  --base-json "D:\projetoCDIA\bases\base_etica.json" `
  --ids-arquivo "D:\projetoCDIA\bases\topicos_atualizar.txt"
```

Antes de gastar créditos, confira a seleção:

```powershell
python pipeline_etica.py embeddings `
  --base-json "D:\projetoCDIA\bases\base_etica.json" `
  --ids-arquivo "D:\projetoCDIA\bases\topicos_atualizar.txt" `
  --simular
```

## Estados do embedding

- `pendente`: tópico novo, ainda sem vetor;
- `atualizado`: vetor corresponde ao hash atual do tópico;
- `desatualizado`: o conteúdo mudou e o vetor anterior foi preservado apenas
  para auditoria;
- tópicos não encontrados num reprocessamento ficam `ativo: false` e não são
  enviados para embeddings.

## Validação local

```powershell
python pipeline_etica.py validar `
  --base-json "D:\projetoCDIA\bases\base_etica.json"
```

O comando verifica IDs duplicados, hashes dos tópicos e embeddings marcados como
atualizados com conteúdo divergente.

Os testes offline podem ser executados sem consumir a API:

```powershell
python -m unittest discover -s tests -v
```

## Modelos

Os padrões são `gpt-5-mini` para decomposição e `text-embedding-3-small` para
embeddings. Ambos podem ser trocados com `--modelo` ou pelas variáveis
`OPENAI_EXTRACTION_MODEL` e `OPENAI_EMBEDDING_MODEL`.

## Observação regulatória

O prompt limita a extração ao texto fornecido e inclui defesa básica contra
instruções inseridas nos documentos. Ainda assim, tópicos e respostas sintéticas
devem passar por revisão humana antes de uso decisório.
