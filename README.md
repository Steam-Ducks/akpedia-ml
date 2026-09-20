# akpedia-ml

Repositório destinado ao serviço de **Machine Learning e processamento de documentos** do projeto
**Akpedia**, construído com **Python 3.12 + FastAPI**.

O `akpedia-ml` é um serviço **stateless**: recebe dados por HTTP (a partir do `akpedia-server`) e
devolve o resultado do processamento. Ele **não acessa o banco de dados**, apenas o `akpedia-server`
tem acesso ao PostgreSQL.

---

## Requisitos

- **Só Docker** (recomendado): [Docker](https://www.docker.com/) + Docker Compose.
- **Desenvolvimento local**: Python 3.12 e o [`uv`](https://docs.astral.sh/uv/) (gerenciador de
  dependências e ambientes). O `uv` cuida do ambiente virtual e das versões — nada mais precisa ser
  instalado manualmente.

---

## Rodando com Docker (qualquer máquina)

```bash
docker compose up --build
```

- API disponível em `http://localhost:8000`

Para parar: `docker compose down`.

As configurações podem ser ajustadas copiando `.env.example` para `.env`.

### Endpoint de verificação

- `GET http://localhost:8000/health` deve retornar `{"status":"UP"}`

---

## Estrutura

```bash
akpedia-ml/
├── pyproject.toml # dependências e config de ferramentas
├── uv.lock # versões travadas das dependências
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .gitignore
├── .env.example
├── README.md
├── src/
│   └── app/
│       ├── __init__.py
│       ├── main.py # app FastAPI, rotas registradas e handlers de erro
│       ├── config.py # configurações lidas de variáveis de ambiente
│       ├── embeddings.py # contrato Embedder + modelo e5 + get_embedder()
│       ├── api/
│       │   ├── __init__.py
│       │   └── documents.py # POST /api/v1/documents/process
│       └── documents/ # leitura e extração de texto dos documentos
│           ├── __init__.py # API pública do pacote
│           ├── base.py # contrato DocumentTextExtractor + normalização do texto
│           ├── chunking.py # split_text(): divide o texto em chunks com overlap
│           ├── errors.py # exceções do processamento
│           ├── pdf.py # extrator de PDF (pypdf)
│           ├── registry.py # mapeia extensão/media type -> extrator
│           └── service.py # extract_text(): ponto de entrada usado pela aplicação
└── tests/
    ├── __init__.py
    ├── pdf_builder.py # gera PDFs mínimos para os testes
    ├── test_chunking.py # divisão em chunks e overlap
    ├── test_health.py # teste do endpoint /health
    ├── test_extractor_registry.py # registro de formatos e ponto de extensão
    ├── test_pdf_extraction.py # extração de texto de PDF
    ├── test_process_document.py # teste da rota de processamento
    └── test_embeddings_model.py # checagens do modelo real (marcadas como slow)
```

---

## Formatos de documento suportados

A extração de texto fica em `src/app/documents`. O restante da aplicação usa apenas
`extract_text()` e recebe uma `str` — nenhum consumidor conhece formatos concretos.

| Formato | Extensão | Media type | Biblioteca |
|---------|----------|------------|------------|
| PDF     | `.pdf`   | `application/pdf` | [`pypdf`](https://pypdf.readthedocs.io) |

```python
from app.documents import extract_text

texto = extract_text(conteudo_em_bytes, filename="manual.pdf")
```

O formato é identificado pela extensão do `filename` e, na falta dela, pelo `media_type`.
Erros: `UnsupportedDocumentFormatError` (nenhum extrator atende o formato) e
`DocumentReadError` (arquivo corrompido, truncado ou protegido por senha) — ambos
herdam de `DocumentProcessingError`.

> Só a camada de texto do PDF é lida. Um PDF apenas de imagens (digitalização sem OCR)
> é lido com sucesso e devolve texto vazio.

### Adicionando um formato novo

1. Crie um extrator em `src/app/documents/` herdando de `DocumentTextExtractor`,
   declarando `extensions` / `media_types` e implementando `extract_text(source)`.
2. Normalize a saída com `normalize_text_parts(...)`, para que todo formato entregue
   texto no mesmo padrão ao pipeline de indexação.
3. Registre-o no `default_registry`, em `service.py`.

Nenhum código existente muda — é só o passo 3 que liga o formato novo à aplicação.

---

## Divisão do texto em chunks

Depois de extraído, o texto é dividido em **chunks** por `split_text()`
(`src/app/documents/chunking.py`). São os chunks que passam pelo modelo de embedding
e vão para o índice.

```python
from app.documents import extract_text, split_text

texto = extract_text(conteudo_em_bytes, filename="manual.pdf")
chunks = split_text(texto)  # defaults abaixo
chunks = split_text(texto, chunk_size=800, chunk_overlap=100)  # ajustado
```

| Parâmetro       | Default | O que é |
|-----------------|---------|---------|
| `chunk_size`    | `1000`  | Tamanho **máximo** de cada chunk, em caracteres. Limite rígido. |
| `chunk_overlap` | `200`   | Trecho, em caracteres, repetido do fim de um chunk no início do seguinte. |

Os tamanhos são em **caracteres**, não em tokens, para o divisor não depender do
tokenizador do modelo. Os defaults foram escolhidos para o chunk caber com folga na
janela de 512 tokens do `multilingual-e5-small` mesmo com o prefixo `passage:`
(português fica em torno de 4–5 caracteres por token, então 1000 caracteres ≈ 200–250
tokens).

### Frases não são cortadas ao meio

O texto é primeiro separado em frases (pontuação final seguida de espaço, ou quebra de
parágrafo). Os chunks são montados com frases inteiras: se a próxima frase não cabe,
ela abre o chunk seguinte. Quebras de linha simples **não** são fronteira de frase —
texto de PDF vem com uma quebra por linha visual, quase sempre no meio da frase — e
são tratadas como espaço.

Única exceção: uma frase sozinha maior que `chunk_size`. Ela é dividida em pedaços
por palavra (e, se uma palavra sozinha for maior que `chunk_size`, por caractere) para
o limite nunca ser ultrapassado.

### Como o overlap funciona

O overlap existe para o contexto não se perder exatamente na fronteira: as frases que
fecham o chunk *n* são as mesmas que abrem o chunk *n+1*, então um trecho que cruza a
fronteira está inteiro em pelo menos um dos dois. Regras:

- O overlap é **alinhado por frase**: repete-se o maior conjunto de frases finais
  inteiras que caiba em `chunk_overlap` caracteres. Por isso ele é um **teto**, não um
  valor exato — se a última frase tem 250 caracteres e `chunk_overlap` é 200, nada é
  repetido.
- `chunk_size` sempre vence: se carregar o overlap estouraria o limite, as frases mais
  antigas do overlap são descartadas primeiro.
- `chunk_overlap = 0` desliga o overlap; deve ser menor que `chunk_size`.

Exemplo com `chunk_size=120` e `chunk_overlap=45`, frases `A`–`F`:

```text
chunk 1: A B
chunk 2:   B C        <- B repetida (cabe em 45 caracteres)
chunk 3:     C D E
chunk 4:         E F  <- E repetida; "D E" não caberia em 45
```

---

## API

### `GET /health`

Verificação de saúde. Responde `{"status":"UP"}`.

### `POST /api/v1/documents/process`

Recebe um arquivo, extrai o texto, divide em chunks e devolve cada chunk com sua
representação numérica (embedding). O serviço é **stateless**: nada é gravado aqui —
quem persiste é o `akpedia-server`.

**Requisição** — `multipart/form-data` com um único campo:

| Campo  | Tipo    | Obrigatório | Observação |
|--------|---------|-------------|------------|
| `file` | arquivo | sim         | Formato identificado pela extensão e, na falta dela, pelo `Content-Type` |

O upload é limitado a `MAX_UPLOAD_BYTES` (25 MiB por padrão). A leitura do arquivo é
truncada nesse limite, então um upload grande demais é recusado com `413` em vez de
ser carregado inteiro na memória só para ser rejeitado depois.

```bash
curl -X POST http://localhost:8000/api/v1/documents/process \
  -F "file=@manual.pdf"
```

**Resposta `200`**

```json
{
  "filename": "manual.pdf",
  "model": {
    "name": "intfloat/multilingual-e5-small",
    "dimensions": 384,
    "normalized": true
  },
  "chunk_count": 2,
  "chunks": [
    { "index": 0, "text": "Primeiro trecho do documento...", "embedding": [0.012, -0.045, "... 384 floats"] },
    { "index": 1, "text": "...segundo trecho.", "embedding": [0.031, 0.008, "..."] }
  ]
}
```

- `index` mantém a ordem dos chunks explícita, sem depender da posição no array.
- `model.dimensions` é o tamanho de todo vetor — é o `N` da coluna `vector(N)` no banco.
- `model.normalized` indica que os vetores têm comprimento 1, então similaridade de
  cosseno equivale a produto escalar.
- `text` é o chunk limpo: o prefixo `passage:` exigido pelo modelo e5 é detalhe interno
  e não aparece na resposta.

**Erros** — todos com o mesmo corpo:

```json
{
  "code": "unsupported_format",
  "message": "Unsupported format for 'planilha.xlsx'. Supported: .pdf.",
  "supported_extensions": [".pdf"]
}
```

| Status | `code`                | Quando |
|--------|-----------------------|--------|
| `413`  | `file_too_large`      | Upload acima de `MAX_UPLOAD_BYTES` |
| `415`  | `unsupported_format`  | Nenhum extrator atende a extensão/media type enviados |
| `422`  | `unreadable_document` | Formato suportado, mas o arquivo está corrompido, truncado ou protegido por senha |
| `422`  | `no_extractable_text` | Arquivo lido com sucesso, mas sem texto (PDF digitalizado — OCR ainda não é suportado) |
| `422`  | *(validação FastAPI)* | Requisição sem o campo `file` |

O campo `supported_extensions` só aparece no `415`: nos demais erros ele é omitido do
corpo, e não devolvido como `null`.

A documentação interativa fica em `http://localhost:8000/docs`.

---

## Embeddings

Cada chunk é convertido em um vetor pelo modelo
[`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small)
(384 dimensões, multilíngue, roda em CPU).

```python
from app.embeddings import get_embedder

embedder = get_embedder()  # carrega o modelo uma única vez
vetores = embedder.embed(["primeiro chunk", "segundo chunk"])
```

| Detalhe | Por quê |
|---------|---------|
| Prefixo `passage: ` nos textos | Os modelos e5 são treinados com prefixo por papel: `passage:` no conteúdo indexado e `query:` na busca. Sem isso a qualidade cai |
| Vetores normalizados | Comprimento 1, então cosseno = produto escalar no índice vetorial |
| `@lru_cache` em `get_embedder()` | Carregar o modelo custa centenas de MB e vários segundos; acontece uma vez por processo |
| Import de `sentence_transformers` dentro do `__init__` | Importar `app.main` não carrega o PyTorch — só carrega quem realmente instancia o modelo |

O contrato `Embedder` é o ponto de extensão: a rota depende só dele, o que permite
trocar o modelo real por um dublê nos testes (veja `tests/test_process_document.py`).

O modelo é baixado do Hugging Face na primeira execução, para o diretório apontado por
`HF_HOME`. No Docker isso é o volume `model-cache`, então o download acontece uma vez só.

Qual modelo carregar vem de `EMBEDDING_MODEL` — qualquer modelo do
`sentence-transformers` serve. Trocá-lo muda o contrato que o `akpedia-server` consome,
então vale conferir dois pontos: `model.dimensions` (é o `N` da coluna `vector(N)`, e
reindexar é obrigatório se mudar) e o prefixo `passage: `, que é específico da família
e5. Os tamanhos padrão de chunk também assumem a janela de 512 tokens do e5.

---

## Configuração

A aplicação lê as configurações a partir de variáveis de ambiente (com defaults para dev):

| Variável           | Default                          | O que é |
|--------------------|----------------------------------|---------|
| `APP_PORT`         | `8000`                           | Porta publicada pela API |
| `HF_HOME`          | `/cache/hf`                      | Onde o modelo de embedding fica em cache (definido no Docker) |
| `EMBEDDING_MODEL`  | `intfloat/multilingual-e5-small` | Modelo usado para gerar os embeddings |
| `MAX_UPLOAD_BYTES` | `26214400` (25 MiB)              | Maior upload aceito pela rota de processamento |

As variáveis são lidas uma vez, na subida do processo (`src/app/config.py`): mudar
qualquer uma delas exige reiniciar o serviço. Um valor inválido em `MAX_UPLOAD_BYTES`
derruba a aplicação no start, em vez de silenciosamente voltar ao default.

---

## Rodando testes e lint localmente

Antes de abrir um PR, rode localmente as mesmas checagens do CI.

### Lint e formatação (Ruff)

```bash
uv run ruff check . # aponta violações de lint
uv run ruff format --check . # verifica a formatação
```

> Para corrigir automaticamente: `uv run ruff check --fix .` e `uv run ruff format .`.

### Testes

```bash
uv run pytest
```

A suíte usa um dublê no lugar do modelo de embedding, então roda em segundos e sem
rede. As checagens contra o modelo real (dimensões e vetores normalizados) estão
marcadas como `slow` e ficam de fora por padrão — e do CI, que não deve baixar
centenas de MB a cada PR:

```bash
uv run pytest -m slow # baixa o modelo na primeira execução
```

---

## Padrões de contribuição (CI)

Todo Pull Request com destino à branch `develop` passa por um pipeline no GitHub Actions
(`.github/workflows/ci.yml`). O merge só é liberado após o pipeline passar **e** a aprovação de outro
membro da equipe.

### Nome da branch

Deve seguir o padrão `AKP-<número>`:

```text
AKP-12
```

### Mensagens de commit

Conventional Commits com o escopo do ticket — `type(AKP-<número>): descrição`:

```text
feat(AKP-12): adiciona endpoint de processamento
fix(AKP-15): corrige extração de texto do PDF
```

Tipos aceitos: `feat`, `fix`, `docs`, `chore`, `refactor`, `test`, `style`, `perf`, `build`, `ci`, `revert`.

### O que o pipeline verifica

| Etapa | O que faz |
|-------|-----------|
| **Padrão de branch & commits** | Valida o nome da branch (`AKP-<número>`) e o padrão dos commits. Roda sempre. |
| **Lint Python (Ruff)** | Roda o Ruff (lint + formatação). Executa apenas quando há mudanças em `src/**`. |
| **Testes de qualidade** | Roda `uv run pytest`. Executa apenas quando há mudanças em `src/**`. |
| **Solicitar aprovação da equipe** | Após tudo passar, solicita a revisão de um membro da equipe. |

> Commits que só alteram configuração/estrutura (fora de `src/**`) não disparam lint nem testes de qualidade.

### Abrindo o PR

A descrição do PR é pré-preenchida pelo template em `.github/PULL_REQUEST_TEMPLATE.md`.
Preencha os campos e marque o checklist antes de solicitar revisão.