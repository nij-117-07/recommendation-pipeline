# Pipeline Architecture

## Stage flow

```mermaid
flowchart LR
    M[record_run_metadata<br/>config hash, seed, versions, git commit]
    I[ingest_data<br/>CSV or synthetic + manifest]
    V[validate_data<br/>rules, error datasets]
    P[prepare_data<br/>clean, dedupe, time split]
    T[transform_data<br/>activity, popularity, user-item matrix]
    F[build_features<br/>FeatureStore]
    TR[train_models<br/>3 models, MLflow]
    E[evaluate_models<br/>ranking + rating metrics]

    M --> I --> V --> P --> T --> F --> TR --> E
```

Both runners execute this same chain. The stage list is defined once in
`src/orchestration/stages.py`:

- `python -m dags.recommendation_pipeline` runs it with the plain-Python runner
  (`src/orchestration/workflow.py`, logs to `reports/run_log.txt`).
- `dags/recommendation_dag.py` runs it as an Airflow DAG (Docker, LocalExecutor, PostgreSQL).

## Data flow and artifacts

```mermaid
flowchart TB
    subgraph raw [data/raw]
      RU[users.csv]
      RI[items.csv]
      RX[interactions.csv]
      MF1[manifest.json]
    end

    subgraph processed [data/processed]
      UV[users_valid / items_valid / interactions_valid]
      CL[users_clean / items_clean / interactions_clean]
      TRN[train.parquet]
      TST[test.parquet]
      DER[user_activity / item_popularity / user_item_matrix]
    end

    subgraph features [data/features - FeatureStore]
      UF[user_features]
      IF[item_features]
      INF[interaction_features]
    end

    ERR[data/errors/*_errors.csv]
    MOD[models/*.pkl + registry.json]
    MLF[(mlflow.db + mlruns/)]
    REP[reports/*]

    RU & RI & RX --> UV
    RU & RI & RX --> MF1
    UV --> ERR
    UV --> CL --> TRN
    CL --> TST
    TRN --> DER
    TRN & CL --> UF & IF & INF
    UF & IF --> MOD
    TRN --> MOD
    MOD --> MLF
    MOD --> REP
    TST --> REP
```

## Models

| Model | Inputs | Role |
|---|---|---|
| `popularity_baseline` | item features (positive count, average rating) | baseline, also the fallback for cold users |
| `genre_content_recommender` | user genre affinity and item genre flags (FeatureStore), item popularity | content-based hybrid |
| `biased_matrix_factorization` | training ratings (SGD) | personalised collaborative filtering |

## Guarantees

- **No leakage:** the split is one global time cutoff. Every feature, popularity count and model
  is fitted on the training split only.
- **Reproducibility:** a single seed drives synthetic data, model initialisation and SGD order.
  `reports/run_metadata.json` records the config hash and package versions for each run.
- **Validation:** the future-timestamp rule uses `reference_date` from `configs/pipeline.json`,
  not the wall clock.
