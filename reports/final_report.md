# End-to-End Data Management Pipeline for a Recommendation System

**Course:** Data Management for Machine Learning, Assignment 2
**Repository:** `recommendation-pipeline/` (see README.md for setup and run instructions)

## 1. Problem and approach

The goal was to build a reproducible pipeline that takes raw user, item and interaction data for an
online platform and produces recommenders that are evaluated and automated end to end. The emphasis
is on clean, modular data engineering, not on maximising accuracy.

Because the environment has no guaranteed internet access, I used a **documented synthetic dataset**
(permitted by the brief). It has 808 users, 404 items (with pipe-separated genres) and 60,600
timestamped ratings from 2020 to 2025. Ratings come from a latent-factor model with popularity skew, so
there is real collaborative signal to learn. The generator also injects data-quality problems at a 1%
rate per error type. The same pipeline runs unchanged on MovieLens-style CSV files placed in `data/raw/`.

## 2. Design

The pipeline is shown in `reports/architecture.md` and runs in this order:

1. **Run metadata** records the config hash, seed, reference date, package versions and git commit.
2. **Ingestion** reads the three CSV files and writes a manifest with row counts and SHA-256 hashes,
   so every run can be traced to its inputs.
3. **Validation** applies named rules. Errors (missing values, wrong types, invalid ratings, invalid or
   future timestamps, duplicate records, orphan user or item IDs) move a row to `data/errors/` with an
   `error_reasons` column. Warnings (missing age, missing genres, repeated user-item pairs) are logged
   and handled during preparation.
4. **Preparation and transformation** impute missing values, standardise genres and gender, remove
   repeated user-item pairs (keeping the latest), and split the data by time. The last 20% of
   interactions form the test set. Derived tables are user activity, item popularity and the user-item
   rating matrix.
5. **Feature management** writes three feature tables to a `FeatureStore`, computed from the training
   split only. User features include average rating, activity count, preferred genre, days since last
   interaction and a 10-column genre affinity profile. Item features include popularity, positive-feedback
   count, average rating, primary genre, number of genres, age and a 10-column genre flag set.
6. **Model training** fits three models. A popularity baseline; a genre content recommender that reads
   the user genre affinity and item genre flags from the FeatureStore and adds a popularity term; and a
   biased matrix factorization model trained with SGD (32 factors, 20 epochs). Parameters, per-epoch
   training RMSE and the pickled models are logged to MLflow.
7. **Evaluation** computes the required metrics on the test set (K = 10, relevant = rating ≥ 4).
8. **Orchestration** runs the stages from one stage list, used by both a plain-Python runner and an
   Apache Airflow DAG. Dependencies are declared explicitly; failed stages are retried where configured
   and downstream stages are skipped. Each plain-Python run writes `reports/run_log.txt`.

Tools used: Python 3.12, pandas 3.1, NumPy 2.5, PyArrow (Parquet), MLflow 3.18 (SQLite backend), pytest,
Apache Airflow 2.10.5 (Docker, PostgreSQL, LocalExecutor). The plain-Python runner is a small custom DAG
executor, which the brief allows as a "clearly structured scheduled Python workflow".

## 3. Data validation results

Validation rejected 16 of 808 user records (duplicate IDs and invalid ages), 8 of 404 item records
(duplicate IDs and non-numeric release years), and 5,912 of 60,600 interactions (9.8%). The interaction
rejections come from the injected faults (missing values, invalid ratings, invalid timestamps, duplicates)
and from orphan-key checks. The orphan checks also catch interactions that point at the users and items
rejected earlier, which is referential integrity working as intended. The repeated user-item pair check
(761 rows) is a warning. In this dataset those rows are the injected exact duplicates that were already
rejected, so preparation removed no further pairs. If real data contains the same user-item pair with
different timestamps, preparation keeps only the latest rating. The full breakdown is in
`reports/validation_report.md`.

## 4. Feature catalog

The feature catalog in `reports/feature_catalog.md` documents 13 named features and two generated groups
of genre columns, one per genre. Each entry has its name, entity, type, source and definition. The
stored tables are 792 user rows (14 columns), 396 item rows (16 columns) and 43,750 training interaction
rows. There are no cold-start users or items in the training split.

## 5. Model evaluation

| Model | Precision@10 | Recall@10 | NDCG@10 | Coverage | Diversity | RMSE | MAE |
|---|---|---|---|---|---|---|---|
| Popularity baseline | 0.1018 | 0.2151 | 0.2038 | 0.0833 | 0.8810 | 1.1381 | 0.8988 |
| Genre content recommender (hybrid) | 0.1051 | 0.2221 | 0.2072 | 0.1288 | 0.8655 | 1.1406 | 0.9119 |
| Biased matrix factorization | 0.0602 | 0.1231 | 0.1020 | **0.8737** | 0.8737 | **0.7670** | **0.6184** |

*Evaluated on 786 test users and 10,938 test ratings. Coverage is the share of the 396 catalogue items that
appear in at least one top-10 list. Diversity is the mean pairwise genre dissimilarity inside each top-10
list. The content recommender predicts ratings as the user's average training rating, so its RMSE and MAE
are close to those of the popularity baseline.*

**Interpretation.**

- The two ranking models that use popularity (the baseline and the content hybrid) rank best. In this
  synthetic data, interactions follow item popularity strongly, so popularity is a hard baseline to beat
  for top-K ranking.
- The content hybrid's gain over the baseline is small (precision 0.105 vs 0.102). With 786 users and no
  confidence intervals, this difference should not be read as significant.
- The genre signal is weak here. The synthetic generator assigns genres at random, so genres do not
  predict ratings. The content model's genre term therefore adds little on this data. A run on MovieLens
  would show whether genre features help on real preferences.
- **Disclosure on tuning.** The popularity weight of the content model (`content.popularity_weight = 1.0`)
  was chosen after a small sweep (0.0, 0.1, 0.5, 1.0, 2.0) evaluated on the test split. The sweep used
  test data, so the content model's test numbers are optimistic. The matrix factorization model used one
  hyperparameter setting and was not tuned.
- The matrix factorization model predicts ratings much more accurately (RMSE 0.77 vs 1.14). It also
  recommends about ten times as many distinct items (coverage 0.87 vs 0.08), so it is more personalised.
  Its lower precision comes from using explicit rating prediction as a ranking score. Implicit-feedback
  training (weighted ALS or BPR) would be the natural next step.

## 6. Reproducibility and orchestration

- A single seed (`random_seed = 42`) controls data generation, model initialisation and SGD order.
  Validation uses a fixed `reference_date` from the configuration, not the wall clock.
- I checked reproducibility by running the pipeline twice from an empty state (raw data deleted). Both
  runs produced identical model results, evaluation summary, validation summary and feature catalog.
- Tests check that the synthetic generator and the matrix factorization model give identical output for
  the same seed, along with the ranking metrics, validation rules, recommender logic, feature logic and
  DAG ordering. Fifteen tests pass.
- `reports/run_metadata.json` records the config hash and package versions for each run.
- The plain-Python runner completes all 8 tasks in about 10 seconds (`reports/run_log.txt`).
- The same 8 tasks ran successfully as an Airflow 2.10.5 DAG in Docker (`recommendation_pipeline`,
  manual run). The DAG and the plain-Python runner share one stage list, so they cannot disagree on order.
- A daily cron example is in the README for scheduled runs.

## 7. Limitations

- **Synthetic data.** The results show that the pipeline works, but they do not measure real-world
  accuracy. Genres are random in the synthetic data, so the content component is not tested on real
  genre preferences. Public data such as MovieLens would give more meaningful comparisons.
- **Single split and single seed.** No cross-validation or confidence intervals were computed.
- **Test-split tuning.** The content model's popularity weight was chosen on the test split (see
  Section 5). A validation split from the training period would give an unbiased choice.
- **Explicit-rating model used for ranking.** Matrix factorization ranks by predicted rating, which is a
  weaker objective than implicit feedback for this task.
- **Cold-start handling is basic.** Unseen users fall back to popularity. Unseen items can never be
  recommended by the matrix factorization model. The synthetic data has no cold-start users or items,
  so this path is tested only by unit tests.
- **Two environments.** The local run uses the pinned versions in `requirements.txt` (pandas 3.1.1, numpy 2.5.4, MLflow 3.18.0). The Airflow image uses the older versions set by Airflow's constraints file (pandas 2.1.4, numpy 1.26.4, MLflow 2.22.5). Its metrics match to about four decimal places but are not bit-identical. The reported results come from the local environment.
- **Local orchestration.** The plain-Python runner executes sequentially on one machine. A managed
  Airflow or Prefect deployment would be needed for distributed execution.
- **Validation thresholds are fixed in code.** For example, valid ages are 1 to 120. They should be
  reviewed for each new dataset. The reference date must be set to the export date of new data.

## 8. Conclusion

The pipeline covers every required stage, from ingestion through orchestration. Each stage is modular,
validated, tested and logged. Three models are trained and compared with the required metrics, and the
whole workflow runs end to end from one command or from Airflow. The main takeaway is that a popularity
baseline is strong for top-K ranking on popularity-skewed data. A personalised matrix factorization model
is clearly better at rating prediction and catalogue coverage. Genre-based features help little on
synthetic data with random genres, so their value should be checked on real data.
