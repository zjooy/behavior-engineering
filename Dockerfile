# Estágio 1: instala as dependências com o uv. Nada daqui vai para a imagem final além da .venv.
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.12.9 /uv /usr/local/bin/uv

WORKDIR /app

# Dependências em camada própria: mudar o código não reinstala tudo
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project

COPY src ./src
RUN uv sync --locked --no-dev --no-editable

# Estágio 2: imagem final, sem uv e sem cache de instalação, rodando como usuário sem privilégios.
FROM python:3.12-slim AS runtime

# GIT_PYTHON_REFRESH: sem Git na imagem, evita avisos do MLflow a cada treino
ENV PATH="/app/.venv/bin:$PATH" \
    GIT_PYTHON_REFRESH=quiet

# libgomp1: exigida pelo XGBoost e ausente na imagem slim
# UID 1000: o mesmo do usuário do host, para gravar nos volumes montados
RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 1000 --user-group --no-create-home --shell /usr/sbin/nologin app

WORKDIR /app

COPY --from=builder /app/.venv ./.venv

# Código e parâmetros pertencem ao root: o processo não consegue alterá-los.
# O src/ é dependência dos estágios no dvc.yaml; o pacote executado é o instalado na .venv.
COPY src ./src
COPY configs ./configs
COPY dvc.yaml .dvcignore ./

# Só o que o pipeline grava pertence ao usuário da aplicação
COPY --chown=app:app dvc.lock ./
COPY --chown=app:app .dvc/config ./.dvc/config
COPY --chown=app:app data/online_shoppers_intention.csv.dvc ./data/
RUN mkdir -p models \
    && dvc config --system core.no_scm true \
    && chown app:app . data models \
    && chown -R app:app .dvc

USER app

# Dados e modelos entram por volume e credenciais só em tempo de execução (ver README)
CMD ["dvc", "repro"]
