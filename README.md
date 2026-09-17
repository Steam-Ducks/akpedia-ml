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
│       ├── main.py # app FastAPI + endpoint /health
│       └── documents/ # leitura e extração de texto dos documentos
│           ├── __init__.py # API pública do pacote
│           ├── base.py # contrato DocumentTextExtractor + normalização do texto
│           ├── errors.py # exceções do processamento
│           ├── pdf.py # extrator de PDF (pypdf)
│           ├── registry.py # mapeia extensão/media type -> extrator
│           └── service.py # extract_text(): ponto de entrada usado pela aplicação
└── tests/
    ├── __init__.py
    ├── pdf_builder.py # gera PDFs mínimos para os testes
    ├── test_health.py # teste do endpoint /health
    ├── test_extractor_registry.py # registro de formatos e ponto de extensão
    └── test_pdf_extraction.py # extração de texto de PDF
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

## Configuração

A aplicação lê as configurações a partir de variáveis de ambiente (com defaults para dev):

| Variável    | Default |
|-------------|---------|
| `APP_PORT`  | `8000`  |

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