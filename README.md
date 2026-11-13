# End-to-End Data Management Pipeline for a Recommendation System

A modular, reproducible pipeline that ingests user, item and interaction data, validates it,
prepares and transforms it, manages features, trains three recommenders (a popularity baseline,
a genre-based content recommender and a matrix-factorization model), evaluates them, and runs the
whole workflow as one orchestrated DAG.

| Stage | Module | What it does |
|---|---|---|
| 0. Run metadata | `src/common/reproducibility.py` | Records config hash, seed, package versions, Python and git commit in `reports/run_metadata.json` |
| 1. Ingestion | `src/ingestion` | Reads CSVs from `data/raw`; if missing, downloads and converts MovieLens 100K (or generates documented synthetic data), writes a manifest with row counts and SHA-256 hashes |
| 2. Validation | `src/validation` | Rule-based checks: missing values, duplicates, data types, rating range, timestamps, referential integrity. Failing rows go to `data/errors` |
| 3. Preparation and transformation | `src/preparation`, `src/transformation` | Imputes and standardises fields, removes repeated user-item pairs, time-based train/test split, user activity, item popularity, user-item matrix |
| 4. Feature management | `src/features` | User, item and interaction features computed from training data only, stored in a `FeatureStore`, documented in `reports/feature_catalog.md` |
| 5. Model training | `src/training` | Popularity baseline, genre content recommender (reads the FeatureStore) and biased matrix factorization (SGD, NumPy). Parameters, per-epoch metrics and models are logged to MLflow |
| 6. Evaluation | `src/evaluation` | Precision@10, Recall@10, NDCG@10, catalog coverage, intra-list diversity, RMSE, MAE; results table in `reports/model_results.md` |
| 7. Orchestration | `src/orchestration`, `dags/` | One stage list (`src/orchestration/stages.py`) run by a dependency-aware Python runner or by Airflow |

Architecture diagram and artifact map: [`reports/architecture.md`](reports/architecture.md).

## Setup

Requirements: Python 3.11+ (tested with 3.12.3).

```bash
cd recommendation-pipeline
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` pins every package the pipeline imports. The Docker image uses
`requirements-docker.txt` on top of the Airflow 2.10.5 constraints file.

## Run the complete pipeline

```bash
python -m dags.recommendation_pipeline
```

On the first run, `data/raw/` has no CSVs, so the pipeline downloads
[MovieLens 100K](https://grouplens.org/datasets/movielens/100k/) (943 users, 1,682 movies,
100,000 ratings) and converts it to the three raw files. This needs internet access once; the zip
is kept in `data/raw/ml-100k.zip`. The source is set by `source` in `configs/pipeline.json`
(`"movielens"` or, for a synthetic dataset, remove the key). The run takes about 30 seconds, including the download. It produces:

- `reports/run_metadata.json`: config hash, seed, `reference_date`, package versions, git commit
- `reports/validation_report.md`: rule results and row counts
- `reports/feature_catalog.md`: feature definitions and cold-start counts
- `reports/model_results.md` / `.csv`: comparison table of all three models
- `reports/run_log.txt`: timestamped log of every task
- `mlflow.db` and `mlruns/`: MLflow tracking data

To view the MLflow UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

### Reproducing a run from scratch

Delete all generated files, including the raw data, and run again. The committed reports
(`final_report.md`, `architecture.md`) and screenshots in `reports/` are kept:

```bash
rm -rf data/raw data/processed data/features data/errors models mlruns mlflow.db
rm -f reports/model_results.* reports/validation_report.md reports/validation_summary.json \
      reports/evaluation_summary.json reports/feature_catalog.md reports/run_log.txt reports/run_metadata.json
python -m dags.recommendation_pipeline
```

The same code, configuration and seed produce identical results. This was checked by running the
pipeline twice from an empty state and comparing `reports/model_results.csv`,
`validation_summary.json`, `evaluation_summary.json` and `feature_catalog.md`: all matched.
`run_metadata.json` and `run_log.txt` contain timestamps, so they differ between runs by design.

### MovieLens 100K conversion

`src/ingestion/movielens.py` maps the MovieLens 100K files to the pipeline schema:

- `users.csv` from `u.user`. MovieLens has no sign-up date, so `signup_date` is the date of each
  user's first rating.
- `items.csv` from `u.item`. Genres come from the 19 genre flags, joined with `|`. Movies with no
  genre get `unknown`. Release year comes from the release date; one movie has none and is left empty.
- `interactions.csv` from `u.data`. Unix timestamps are converted to ISO 8601 UTC.

### Data source and attribution

MovieLens 100K was collected by GroupLens Research at the University of Minnesota. If you use it in
a publication, cite:

> F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context.
> ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: Article 19.

The license and terms of use are at https://grouplens.org/datasets/movielens/100k/.

### Using your own data

Place three CSV files in `data/raw/` with these columns and re-run the pipeline:

| File | Required columns |
|---|---|
| `users.csv` | `user_id`, `age`, `gender`, `occupation`, `signup_date` (YYYY-MM-DD) |
| `items.csv` | `item_id`, `title`, `genres` (pipe-separated, e.g. `Action\|Comedy`), `release_year` |
| `interactions.csv` | `user_id`, `item_id`, `rating` (1-5), `timestamp` (ISO 8601, e.g. `2020-01-01T10:00:00Z`) |

If the files exist, the ingestion stage uses them and skips the download or synthetic generation. Missing columns
in the CSVs will cause the validation stage to fail, which is intentional. Timestamps later than
`reference_date` in `configs/pipeline.json` are rejected as future timestamps; set that value to
the date your data was exported.

## Run with Apache Airflow (Docker)

The same stages run as an Airflow DAG (`recommendation_pipeline`, defined in
`dags/recommendation_dag.py`). Airflow 2.10.5 runs in Docker with PostgreSQL and the LocalExecutor.

```bash
docker compose up -d --build        # first start: build image, migrate DB, create user
```

- Web UI: http://localhost:8081 (login `airflow` / `airflow`, change in `.env`)
- Trigger the DAG from the UI (or `docker compose exec airflow-scheduler airflow dags trigger recommendation_pipeline`)
- Outputs are written to the project folder on the host (`data/`, `reports/`, `models/`, `mlflow.db`).

Stop the stack with `docker compose down` (add `-v` to also delete Airflow's metadata database).

Screenshots:
- Successful run, graph view (all 8 tasks green): [`reports/airflow_successful_run.png`](reports/airflow_successful_run.png)
- DAG overview: [`reports/airflow_dag_overview.png`](reports/airflow_dag_overview.png)
- Task durations: [`reports/airflow_task_durations.png`](reports/airflow_task_durations.png). `train_models` is the slowest task, at about 15 seconds on the MovieLens data.

Notes:
- The container uses its own MLflow database (`mlflow-airflow.db`, set by `MLFLOW_TRACKING_DB`), because the local `mlflow.db` was created by MLflow 3.x and cannot be read by the MLflow 2.x in the image.
- `MLFLOW_ENABLE_ASYNC_LOGGING=false` is set in `docker-compose.yml`. Without it, the MLflow training task exited silently inside Airflow's forked worker process.
- `AIRFLOW__CORE__EXECUTE_TASKS_NEW_PYTHON_INTERPRETER=true` is also set. The DAG file imports MLflow, so the forked task inherited MLflow state from the scheduler and `train_models` exited with code 1 and no traceback. Starting each task in a fresh interpreter fixes this.
- The Airflow image installs the versions in Airflow's constraints file, which are older than the local pins: pandas 2.1.4, numpy 1.26.4 and MLflow 2.22.5 (local: pandas 3.1.1, numpy 2.5.4, MLflow 3.18.0). Metrics from the Docker run can therefore differ from the local run in the last digits, and the timestamp dtype in the Parquet files differs. The committed reports come from the local environment in `requirements.txt`.

## Run tests

```bash
pytest
```

The tests cover:
- ranking metrics on a known example
- validation rules, including the configured `reference_date` for future timestamps
- recommender logic (seen items are never recommended; the content model ranks by genre match)
- feature logic (genre affinity is a share of positive interactions only)
- reproducibility (synthetic data and matrix factorization give identical output for the same seed)
- DAG ordering, dependency checks and cycle detection

## Scheduling

The workflow is a plain Python program, so any scheduler can run it. Example cron entry
(every day at 02:00):

```cron
0 2 * * * cd /path/to/recommendation-pipeline && .venv/bin/python -m dags.recommendation_pipeline >> reports/cron.log 2>&1
```

## Configuration

All parameters are in `configs/pipeline.json`:

- `random_seed`: seed for synthetic data, model initialisation and SGD order
- `reference_date`: fixed "today" for the future-timestamp rule (keeps validation reproducible)
- `source`: `"movielens"` to download MovieLens 100K when no raw CSVs exist; remove it to use synthetic data
- `synthetic`: size of the synthetic dataset and the share of injected errors per error type
- `validation`: valid rating range
- `split.test_fraction`: fraction of the most recent interactions used for testing
- `relevance_threshold`: a test rating at or above this value counts as relevant
- `top_k`: K for ranking metrics
- `content.popularity_weight`: weight of the popularity term in the content model score (genre match has weight 1)
- `mf`: matrix factorization hyperparameters
- `mlflow.experiment_name`: experiment name in MLflow

## Project structure

```
recommendation-pipeline/
├── configs/pipeline.json              # all pipeline parameters
├── data/
│   ├── raw/                           # input CSVs + manifest.json
│   ├── processed/                     # validated, clean, train/test, derived datasets
│   ├── features/                      # FeatureStore (parquet)
│   └── errors/                        # rejected rows with error_reasons
├── dags/
│   ├── recommendation_pipeline.py     # plain-Python entry point
│   └── recommendation_dag.py          # Airflow DAG
├── src/
│   ├── common/                        # config, logging, MLflow setup, run metadata
│   ├── ingestion/
│   ├── validation/
│   ├── preparation/
│   ├── transformation/
│   ├── features/                      # feature definitions and FeatureStore
│   ├── training/                      # models and training
│   ├── evaluation/                    # metrics and evaluation
│   └── orchestration/                 # stage list (stages.py) and task runner (workflow.py)
├── models/                            # pickled models + registry.json
├── reports/                           # deliverable reports, architecture, logs
├── tests/test_pipeline.py
├── requirements.txt
├── requirements-docker.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## Design notes

- **No data leakage.** The split is global and time-based. Features, popularity, the content model
  and matrix factorization are fitted only on the training period. Recommendations exclude items
  the user already interacted with in training.
- **Reproducibility.** A fixed seed drives data generation, model initialisation and SGD order.
  Validation uses a configured reference date, not the wall clock. Each run records its config hash
  and package versions. Each stage reads and writes files, so stages can be re-run and inspected
  independently.
- **Feature reuse.** `FeatureStore` is the single access point for features. The popularity baseline
  reads item features, and the content model reads user genre affinity and item genre flags from the
  store. Both use the same definitions that `reports/feature_catalog.md` documents.
- **One workflow definition.** The stage list in `src/orchestration/stages.py` drives both the
  plain-Python runner and the Airflow DAG, so the two cannot disagree on order or dependencies.
- **Cold start.** Users with no training history receive popularity recommendations. Items unseen
  in training are never recommended by the content or matrix factorization models.
- **Genre signal.** The content model's genre term is measured on MovieLens 100K, where genres are
  real. The `synthetic` configuration remains available for tests and generates random genres.
