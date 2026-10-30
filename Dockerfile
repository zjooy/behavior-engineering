# Imagens fixadas por digest: a tag pode mudar de conteúdo, o digest não.
# Para atualizar: docker buildx imagetools inspect python:3.12-slim-trixie

# Estágio 1: instala as dependências com o uv. Nada daqui vai para a imagem final além da .venv.
FROM python:3.12-slim-trixie@sha256:aa943ce411423c7582288f18707ff66eddcbaeb510502d747a72aeb34f30d159 AS builder

COPY --from=ghcr.io/astral-sh/uv:0.12.9@sha256:8b940d3a9d65bed080436972241af2e21c84b5e8c9193f7014ed71479ee795ff /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependências em camada própria: mudar o código não reinstala tudo
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project

COPY src ./src
RUN uv sync --locked --no-dev --no-editable

# Estágio 2: imagem final, sem uv e sem cache de instalação, rodando como usuário sem privilégios.
FROM python:3.12-slim-trixie@sha256:aa943ce411423c7582288f18707ff66eddcbaeb510502d747a72aeb34f30d159 AS runtime

# UID igual ao do usuário do host, para gravar nos volumes montados (--build-arg para outro valor)
ARG APP_UID=1000
ARG APP_GID=1000

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    GIT_PYTHON_REFRESH=quiet

# libgomp1: exigida pelo XGBoost e ausente na imagem slim
RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid "$APP_GID" app \
    && useradd --uid "$APP_UID" --gid app --no-create-home --shell /usr/sbin/nologin app

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
    && dvc config --system core.analytics false \
    && chown app:app . data models \
    && chown -R app:app .dvc

USER app

# Dados e modelos entram por volume e credenciais só em tempo de execução (ver README)
CMD ["dvc", "repro"]
