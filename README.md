# Behavior Engineering

> Projeto de machine learning para estimar a intenção de compra em sessões de e-commerce com o conjunto **Online Shoppers Purchasing Intention**.

## Objetivo

O alvo `Revenue` indica se uma sessão resultou em compra. O projeto contém análise exploratória, um modelo baseline e experimentos de classificação registrados com MLflow. FastAPI e Pydantic estão nas dependências, mas uma API de inferência ainda não foi implementada.

## Dados, notebooks e modelos

O arquivo `data/online_shoppers_intention.csv` contém 12.330 sessões com atributos de navegação, duração, origem de tráfego e tipo de visitante. A variável alvo é desbalanceada: a maioria das sessões não resulta em compra.

| Caminho | Conteúdo |
| --- | --- |
| `notebooks/eda.ipynb` | Análise exploratória dos dados |
| `notebooks/baseline_model.ipynb` | Regressão logística baseline; registra acurácia, F1, precisão, recall e PR-AUC no MLflow |
| `notebooks/model_experiments.ipynb` | Compara árvore de decisão, Random Forest e XGBoost; otimiza hiperparâmetros do XGBoost com Optuna e validação cruzada estratificada de 5 folds, usando PR-AUC como objetivo |
| `models/baseline_model.joblib` | Classificador baseline treinado |
| `models/champion_model.joblib` | Pipeline campeão treinado, incluindo pré-processamento e classificador |
| `artifacts/` | Artefatos e métricas registrados pelo MLflow |

Os dois modelos persistidos têm interfaces diferentes. O baseline contém apenas o classificador: codificação, transformações e escalonamento são feitos no notebook e não estão incluídos no arquivo. O campeão é um pipeline do scikit-learn com pré-processador e XGBoost, portanto inclui as etapas de pré-processamento usadas no treinamento.

## Configuração e execução

Requisitos: Python 3.12 ou superior e [uv](https://docs.astral.sh/uv/).

Instale as dependências de desenvolvimento e notebooks:

```bash
uv sync --group dev --group notebook
```

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

## Estado atual

A análise exploratória e os notebooks de treinamento estão no repositório, assim como os arquivos serializados do baseline e do pipeline campeão. O pacote `src/` ainda contém apenas o módulo inicial, sem rotina de inferência ou API, e `tests/` está vazio.

Próximos passos possíveis: implementar inferência usando o pipeline campeão, criar a API e adicionar testes para validar entradas e previsões.
