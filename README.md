# Behavior Engineering

> Projeto de machine learning para estimar a intenção de compra em sessões de e-commerce com o conjunto **Online Shoppers Purchasing Intention**.

## Objetivo

O alvo `Revenue` indica se uma sessão resultou em compra. O projeto contém análise exploratória, um modelo baseline e experimentos de classificação registrados com MLflow. FastAPI e Pydantic estão nas dependências, mas uma API de inferência ainda não foi implementada.

## Dados, notebooks e modelos

O arquivo `data/online_shoppers_intention.csv` (versionado pelo DVC; obtenha com `uv run dvc pull`) contém 12.330 sessões com atributos de navegação, duração, origem de tráfego e tipo de visitante. A variável alvo é desbalanceada: a maioria das sessões não resulta em compra.

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

Primeira configuração da máquina:

```bash
uv sync --group notebook
cp .env.example .env               # ajuste os valores locais; o .env não vai para o Git
uv run pre-commit install          # roda ruff automaticamente antes de cada commit
```

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
| `pipelines/preprocess.py`, `pipelines/train.py` | Pontos de entrada dos estágios `preprocess` e `train` |

Seeds, colunas e hiperparâmetros (os melhores do estudo Optuna) ficam em `configs/params.yaml`. Configurações da máquina (dispositivo, MLflow) ficam no `.env`:

```bash
uv run python -m behavior_engineering.pipelines.preprocess   # gera data/processed/{train,test}.csv
uv run python -m behavior_engineering.pipelines.train        # gera models/model.joblib e metrics.json
```

## Dados e pipeline com DVC

O dataset e as saídas do pipeline são versionados pelo [DVC](https://dvc.org/). O Git guarda só as "etiquetas" (`*.dvc`, `dvc.lock`), e os arquivos ficam no remote do [DagsHub](https://dagshub.com/zjooy/behavior-engineering).

```
data/online_shoppers_intention.csv ──► preprocess ──► data/processed/ ──► train ──► models/model.joblib
                                                                                  └─► metrics.json
```

O `dvc.yaml` define os estágios. As dependências de cada um (dados, código e parâmetros de `configs/params.yaml`) determinam o que precisa rodar de novo.

Configure a credencial do DagsHub uma vez por máquina. Ela fica em `.dvc/config.local`, fora do Git. O token está em *DagsHub → Settings → Tokens*:

```bash
uv run dvc remote modify origin --local auth basic
uv run dvc remote modify origin --local user <seu-usuario>
uv run dvc remote modify origin --local password <seu-token>
```

| Comando | O que faz |
| --- | --- |
| `uv run dvc pull` | Baixa do remote o dataset e as saídas da versão atual |
| `uv run dvc repro` | Roda o pipeline, apenas os estágios cujas dependências mudaram |
| `uv run dvc push` | Envia dados e saídas novos para o remote |
| `uv run dvc metrics show` | Mostra as métricas atuais |
| `uv run dvc params diff` / `uv run dvc metrics diff` | Compara parâmetros e métricas com o último commit |

Fluxo para um experimento: altere `configs/params.yaml`, rode `dvc repro`, compare com `dvc metrics diff` e, se o resultado for bom, faça commit de `params.yaml`, `dvc.lock` e `metrics.json` e depois `dvc push`.

No Windows, se o `dvc.exe` for bloqueado pelo Controle de Aplicativo, use `uv run python -m dvc`.

## Estado atual

Concluído: análise exploratória, experimentos em notebooks, pipeline de treino em `src/` com testes e pipeline DVC (`preprocess` → `train`). Próximos passos: Dockerfile, registro do modelo no MLflow Model Registry e API de inferência.
