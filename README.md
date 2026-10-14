# Behavior Engineering

> Projeto de machine learning para estimar a intenção de compra em sessões de e-commerce com o conjunto **Online Shoppers Purchasing Intention**.

## Objetivo

O alvo `Revenue` indica se uma sessão resultou em compra. O projeto contém análise exploratória, um modelo baseline e experimentos de classificação registrados com MLflow. FastAPI e Pydantic estão nas dependências, mas uma API de inferência ainda não foi implementada.

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

## Estado atual

Concluído: análise exploratória, experimentos em notebooks, pipeline de treino em `src/` com testes e pipeline DVC (`preprocess` → `train`). Próximos passos: Dockerfile, registro do modelo no MLflow Model Registry e API de inferência.
