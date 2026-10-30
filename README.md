# Behavior Engineering

> Projeto de machine learning para estimar a intenção de compra em sessões de e-commerce com o conjunto **Online Shoppers Purchasing Intention**.

## Objetivo

O alvo `Revenue` indica se uma sessão resultou em compra. O projeto contém análise exploratória, um modelo baseline e experimentos de classificação registrados com MLflow.

## Dados, notebooks e modelos

O arquivo `data/online_shoppers_intention.csv` (versionado pelo DVC; obtenha com `uv run python -m dvc pull`) contém 12.330 sessões com atributos de navegação, duração, origem de tráfego e tipo de visitante. A variável alvo é desbalanceada: a maioria das sessões não resulta em compra.

| Caminho | Conteúdo |
| --- | --- |
| `notebooks/eda.ipynb` | Análise exploratória dos dados |
| `notebooks/baseline_model.ipynb` | Regressão logística baseline; registra acurácia, F1, precisão, recall e PR-AUC no MLflow |
| `notebooks/model_experiments.ipynb` | Compara árvore de decisão, Random Forest e XGBoost; otimiza hiperparâmetros do XGBoost com Optuna e validação cruzada estratificada de 5 folds, usando PR-AUC como objetivo |
| `models/baseline_model.joblib` | Classificador baseline treinado |
| `models/champion_model.joblib` | Pipeline campeão treinado, incluindo pré-processamento e classificador |
| `artifacts/` | Artefatos e métricas registrados pelo MLflow |

Os dois modelos persistidos têm interfaces diferentes. O baseline contém apenas o classificador: codificação, transformações e escalonamento são feitos no notebook e não estão incluídos no arquivo. O campeão é um pipeline do scikit-learn com pré-processador e XGBoost, portanto inclui as etapas de pré-processamento usadas no treinamento.

## Ambiente

Requisitos: Python 3.12 ou superior e [uv](https://docs.astral.sh/uv/). O projeto usa uv, que o enunciado aceita como alternativa ao Poetry: `uv sync` faz o mesmo papel de `poetry install` e o lock file é o `uv.lock`.

| Comando | Instala |
| --- | --- |
| `uv sync` | Produção + grupo `dev` (pytest, ruff, pre-commit). Uso diário |
| `uv sync --no-dev` | Só as dependências de produção (usado na imagem Docker) |
| `uv sync --group notebook` | Adiciona Jupyter, Optuna e bibliotecas de gráficos |

Instalação do uv (se `uv --version` não funcionar):

```bash
python -m pip install --user uv
```

Se depois disso `uv` ainda não for reconhecido, a pasta de scripts do Python não está no PATH. Use `python -m uv` no lugar de `uv` ou adicione a pasta ao PATH pelo PowerShell (prompt começando com `PS`) e reabra o terminal:

```powershell
$scripts = python -c "import sysconfig; print(sysconfig.get_path('scripts', 'nt_user'))"
[Environment]::SetEnvironmentVariable('Path', [Environment]::GetEnvironmentVariable('Path', 'User') + ";$scripts", 'User')
```

Primeira configuração do projeto (na pasta do repositório):

```bash
uv sync --group notebook
cp .env.example .env               # no cmd do Windows: copy .env.example .env
uv run pre-commit install          # roda ruff automaticamente antes de cada commit
```

Ajuste os valores do `.env` para a sua máquina. Ele não vai para o Git. Em seguida, configure o DVC ([Dados e pipeline com DVC](#dados-e-pipeline-com-dvc)).

Testes e lint:

```bash
uv run pytest
uv run ruff check .
```

No Windows, se o Controle de Aplicativo bloquear o `pytest.exe`, use `uv run python -m pytest`. Se o projeto estiver numa pasta do OneDrive, defina `UV_LINK_MODE=copy` antes do `uv sync`.

## Notebooks e MLflow

Inicie o servidor MLflow na raiz do repositório:

```bash
uv run mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./artifacts \
  --host 127.0.0.1 \
  --port 5000
```

A interface fica disponível em `http://127.0.0.1:5000`. Em outro terminal, inicie o Jupyter a partir da pasta `notebooks`:

```bash
cd notebooks
uv run jupyter lab
```

Os notebooks leem os dados usando caminhos relativos à pasta `notebooks`. Execute `baseline_model.ipynb` para treinar o baseline ou `model_experiments.ipynb` para comparar e otimizar modelos, registrar resultados no experimento `Online_Shoppers_Base` e salvar o pipeline campeão. O notebook de experimentos define `device="cuda"` para a busca de hiperparâmetros e para o treino final; em ambientes sem GPU CUDA, ajuste essa configuração para CPU antes de executar.

## Pipeline em `src/`

O treino do modelo campeão também existe como código Python no pacote `behavior_engineering`, sem depender dos notebooks:

| Módulo | Responsabilidade |
| --- | --- |
| `config.py` | `Settings` (variáveis do `.env`) e `load_params` (lê `configs/params.yaml`) |
| `data/loader.py` | `DataLoader` (leitura do CSV, separação X/y) e `split_train_test` estratificado |
| `features/preprocessor.py` | `build_preprocessor`: descarta colunas de ruído e aplica one-hot nas categóricas |
| `models/trainer.py` | `Trainer`: monta e ajusta o `Pipeline` (pré-processador + XGBoost) e salva com joblib |
| `models/evaluator.py` | `Evaluator`: acurácia, F1, precisão, recall e PR-AUC |
| `models/tracking.py` | `TrainingTracker`: registra no MLflow parâmetros, métricas, modelo e as versões do Git e dos dados; cria a versão no Model Registry e promove a melhor |
| `pipelines/preprocess.py`, `pipelines/train.py` | Pontos de entrada dos estágios `preprocess` e `train` |

Seeds, colunas, hiperparâmetros (os melhores do estudo Optuna) e o dispositivo de treino (`model.device`) ficam em `configs/params.yaml`. As configurações do MLflow ficam no `.env`:

```bash
uv run python -m behavior_engineering.pipelines.preprocess   # gera data/processed/{train,test}.csv
uv run python -m behavior_engineering.pipelines.train        # gera models/model.joblib e metrics.json
```

### Registro no MLflow

O estágio `train` cria um run `XGBoost_Pipeline` no experimento de `MLFLOW_EXPERIMENT_NAME`, com os parâmetros de `configs/params.yaml`, as métricas, o modelo e estas tags:

| Tag | Conteúdo |
| --- | --- |
| `mlflow.source.git.commit` | Commit atual, registrado pelo próprio MLflow. Com `git_dirty=true`, é o commit anterior ao experimento, pois o `dvc repro` roda antes do `git commit` |
| `git_dirty` | `true` se havia alterações locais no momento do treino |
| `data_md5` | Hash de `data/processed` no `dvc.lock`. Só é confiável via `dvc repro`, que atualiza o lock antes do treino |

Cada treino registrado também cria uma nova versão do modelo `MLFLOW_REGISTERED_MODEL_NAME` no Model Registry. A versão recebe o alias `champion` só se o `test_pr_auc` dela superar o da campeã atual; a primeira versão é promovida automaticamente. Para carregar a campeã:

```python
mlflow.sklearn.load_model("models:/Online_Shoppers_Classifier@champion")
```

O `dvc repro` pula o treino quando nada mudou e, nesse caso, nenhuma versão é criada. Para treinar e registrar mesmo assim: `uv run python -m dvc repro --force --single-item train`.

O servidor MLflow precisa estar no ar ([Notebooks e MLflow](#notebooks-e-mlflow)); caso contrário, o treino falha antes de começar. Para treinar sem registrar, defina `MLFLOW_ENABLED=false` no `.env`.

## Dados e pipeline com DVC

O dataset e as saídas do pipeline são versionados pelo [DVC](https://dvc.org/). O Git guarda só as "etiquetas" (`*.dvc`, `dvc.lock`), e os arquivos ficam no remote do [DagsHub](https://dagshub.com/zjooy/behavior-engineering).

```
data/online_shoppers_intention.csv ──► preprocess ──► data/processed/ ──► train ──► models/model.joblib
                                                                                  └─► metrics.json
```

O `dvc.yaml` define os estágios. As dependências de cada um (dados, código e parâmetros de `configs/params.yaml`) determinam o que precisa rodar de novo.

### Primeira configuração (cada integrante, uma vez por máquina)

Faça antes a seção [Ambiente](#ambiente): o `uv` precisa estar funcionando e o `uv sync` precisa ter rodado.

**1. Crie sua conta no DagsHub e peça acesso ao repositório**

- Crie a conta em [dagshub.com](https://dagshub.com/) (dá para entrar com a conta do GitHub).
- Peça ao dono do repositório (`zjooy`) para adicioná-lo como colaborador em *DagsHub → behavior-engineering → Settings → Collaborators*. Sem isso, o `dvc push` é recusado.

**2. Gere um token de acesso**

No DagsHub: clique no seu avatar → *User Settings* → *Tokens*. Copie o token (ele funciona como uma senha; não compartilhe nem cole no código).

**3. Configure a credencial no projeto**

No terminal, **na pasta do projeto**, rode os três comandos trocando `<seu-usuario>` e `<seu-token>`:

```bash
uv run python -m dvc remote modify origin --local auth basic
uv run python -m dvc remote modify origin --local user <seu-usuario>
uv run python -m dvc remote modify origin --local password <seu-token>
```

O `--local` grava a credencial em `.dvc/config.local`, que **não vai para o Git**. O endereço do remote já vem configurado em `.dvc/config`.

**4. Baixe os dados e confira**

```bash
uv run python -m dvc pull       # baixa o dataset, os splits e o modelo da versão atual
uv run python -m dvc status -c  # deve responder: "Cache and remote 'origin' are in sync"
```

Pronto. O arquivo `data/online_shoppers_intention.csv` aparece na pasta `data/`.

### Problemas comuns

| Erro | Causa e solução |
| --- | --- |
| `uv` não é reconhecido como comando | A pasta do uv não está no PATH. Use `python -m uv ...` no lugar de `uv ...` ou adicione a pasta ao PATH (veja [Ambiente](#ambiente)) |
| `A sintaxe do nome do arquivo... está incorreta` | O comando de PowerShell foi digitado no cmd. Use um terminal cujo prompt começa com `PS` |
| `dvc.exe` ou `pytest.exe` bloqueado pelo Windows | O Controle de Aplicativo bloqueia o executável. Use a forma `python -m dvc` / `python -m pytest`, como nos exemplos acima |
| `401 Unauthorized` / `403 Forbidden` no push ou pull | Token errado ou expirado (refaça o passo 3) ou falta acesso de colaborador (passo 1) |
| `failed to hardlink` / `Acesso negado` no `uv sync` | Projeto dentro do OneDrive. Rode `$env:UV_LINK_MODE="copy"` (PowerShell) antes do `uv sync` |

### Uso no dia a dia

| Comando | O que faz |
| --- | --- |
| `uv run python -m dvc pull` | Baixa do remote os dados da versão atual (rode depois de cada `git pull`) |
| `uv run python -m dvc repro` | Roda o pipeline; só refaz os estágios cujas dependências mudaram |
| `uv run python -m dvc push` | Envia dados e saídas novos para o remote |
| `uv run python -m dvc metrics show` | Mostra as métricas atuais |
| `uv run python -m dvc params diff` / `metrics diff` | Compara parâmetros e métricas com o último commit |

Fluxo para um experimento:

1. Altere um valor em `configs/params.yaml` (ex.: `n_estimators`).
2. `uv run python -m dvc repro` treina de novo.
3. `uv run python -m dvc metrics diff` compara com a versão anterior.
4. Se o resultado for melhor: `git add configs/params.yaml dvc.lock metrics.json`, `git commit` e `uv run python -m dvc push`. **Sempre faça o `dvc push` junto com o `git push`**; sem ele, os colegas recebem as etiquetas novas, mas não os arquivos.

## Docker

O `Dockerfile` tem dois estágios: o primeiro instala as dependências de produção com o uv e o segundo copia só o resultado para a imagem final, que roda sem root e não contém uv, dados, modelos nem credenciais.

```bash
docker build -t behavior-engineering .
```

Dados e modelos entram por volume. Para baixá-los do remote, monte a credencial do DVC somente para leitura (ela não fica na imagem):

```bash
docker run --rm \
  -v "$PWD/.dvc/config.local:/app/.dvc/config.local:ro" \
  -v "$PWD/data:/app/data" \
  -v "$PWD/models:/app/models" \
  behavior-engineering dvc pull
```

O comando padrão da imagem é `dvc repro`. Para rodar o pipeline sem MLflow e copiar para fora o que fica fora dos volumes:

```bash
docker run --name behavior-train \
  -e MLFLOW_ENABLED=false \
  -v "$PWD/data:/app/data" \
  -v "$PWD/models:/app/models" \
  behavior-engineering

docker cp behavior-train:/app/metrics.json .
docker cp behavior-train:/app/dvc.lock .
docker rm behavior-train
```

Não monte `metrics.json` nem `dvc.lock` como arquivos: o DVC precisa removê-los e falha com `Device or resource busy`.

Para registrar no MLflow (no Linux), com o servidor da seção [Notebooks e MLflow](#notebooks-e-mlflow) no ar, use a rede do host e monte `artifacts/` **no mesmo caminho absoluto** do host, pois é nesse caminho que o cliente grava os artefatos:

```bash
docker run --rm --network host \
  -v "$PWD/data:/app/data" \
  -v "$PWD/models:/app/models" \
  -v "$PWD/artifacts:$PWD/artifacts" \
  behavior-engineering
```

Sem Git na imagem, o run fica com `git_dirty=unknown` e sem `mlflow.source.git.commit`. A imagem roda com o UID 1000: se o seu usuário tiver outro UID, ajuste as permissões de `data/` e `models/`.

## Estado atual

Concluído: análise exploratória, experimentos em notebooks, pipeline de treino em `src/` com testes, pipeline DVC (`preprocess` → `train`), Dockerfile e registro e promoção do modelo no MLflow Model Registry.
