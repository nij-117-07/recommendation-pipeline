# End-to-End Data Management Pipeline for a Recommendation System

**Assignment 2**

**Name:** KAPIL UNIYARA
**Bits ID No.:** 2025AG05832
**Course:** Data Management for Machine Learning
**GitHub repository:** https://github.com/nij-117-07/recommendation-pipeline (setup and run instructions in README.md)

## 1. Problem and data

The pipeline ingests user, item and rating data, validates it, prepares and transforms it, builds
features, trains and evaluates recommenders, and runs the whole workflow as one orchestrated DAG. The
emphasis is on clean, reproducible data engineering rather than accuracy.

The dataset is **MovieLens 100K**: 943 users, 1,682 movies with genre flags, and 100,000 ratings (1 to 5)
made between September 1997 and April 1998. The pipeline downloads it on the first run and converts it to
the required CSV files (`src/ingestion/movielens.py`).

## 2. Design

Stages run in this order, defined once in `src/orchestration/stages.py`:

1. **Metadata:** records config hash, seed, package versions and git commit.
2. **Ingestion:** reads the CSVs and writes a manifest with row counts and SHA-256 hashes. MovieLens has no
   sign-up date, so `signup_date` is each user's first rating date.
3. **Validation:** named rules for missing values, types, rating range, timestamps, duplicates and orphan
   keys. Failing rows go to `data/errors/` with the reasons.
4. **Preparation and transformation:** imputes missing values, removes repeated user-item pairs, and makes a
   global time-based split (last 20% of ratings = test). Derives user activity, item popularity and the
   user-item matrix.
5. **Feature management:** user, item and interaction features are computed from the training split only and
   stored in a FeatureStore shared by training and prediction (`reports/feature_catalog.md`).
6. **Training:** a popularity baseline; a genre content recommender (genre affinity plus popularity); and
   biased matrix factorization (SGD, 32 factors, 20 epochs). Parameters and metrics are logged to MLflow.
7. **Evaluation:** Precision@10, Recall@10, NDCG@10, coverage, diversity, RMSE and MAE.
8. **Orchestration:** one stage list drives a plain-Python runner and an Airflow 2.10.5 DAG (Docker,
   PostgreSQL, LocalExecutor).

Tools: Python 3.12, pandas, NumPy, PyArrow, MLflow, pytest, Apache Airflow.

## 3. Validation and features

Validation rejected no rows: all 943 users, 1,682 items and 100,000 interactions passed
(`reports/validation_report.md`). The rules were tested earlier on a synthetic dataset with injected faults,
which they caught. One movie has no release date and is imputed with the median.

The FeatureStore holds 13 named features plus 19 genre flags and 19 user genre-affinity columns. Of the 943
users, 192 have no training history (cold-start).

## 4. Results

| Model | Precision@10 | Recall@10 | NDCG@10 | Coverage | Diversity | RMSE | MAE |
|---|---|---|---|---|---|---|---|
| Popularity baseline | 0.2445 | 0.0826 | 0.2703 | 0.0440 | 0.7854 | 1.0366 | 0.8330 |
| Genre content recommender | 0.2476 | 0.0849 | 0.2740 | 0.0470 | 0.7617 | 1.1805 | 0.9573 |
| Biased matrix factorization | 0.2259 | 0.0701 | 0.2497 | **0.0957** | 0.7845 | **0.9723** | **0.7671** |

*Ranking metrics average over 290 test users with at least one test rating of 4 or higher; 192 of them are
cold-start. RMSE and MAE use 2,864 test ratings with known users and items.*

- **Ranking is dominated by popularity.** Unseen users receive popularity lists from every model, and two-thirds
  of evaluated users are cold-start. The ranking differences therefore come mainly from the 98 users with
  history.
- **The content hybrid is only marginally better than the baseline** on ranking (precision 0.248 vs 0.245). With
  290 users and no confidence intervals, this gap is not significant.
- **Matrix factorization predicts ratings best and covers the most items**, but ranks lower because it orders
  by predicted rating rather than by implicit feedback.
- **Tuning disclosure.** The content model's popularity weight (1.0) was chosen on the test split of the earlier
  synthetic data and not re-tuned on MovieLens. Matrix factorization used one untuned setting.

## 5. Reproducibility and orchestration

- A fixed seed (42) and a configured reference date make runs repeatable. Repeated runs from an empty state
  gave identical metrics, and the Airflow run matched the local run to four decimal places.
- 15 unit tests cover metrics, validation rules, recommenders, feature logic and DAG ordering. All pass.
- The plain-Python runner completes all 8 stages in about 30 seconds, including the download.
- The Airflow DAG (`recommendation_pipeline`) ran all 8 tasks green.

[INSERT SCREENSHOT: Airflow graph view of the successful run, reports/airflow_successful_run.png]

- Two settings in `docker-compose.yml` were needed: synchronous MLflow logging, and each task starting in a
  fresh interpreter. Without the second, the forked training task exited silently.

## 6. Limitations

- **Old data and a time split with many cold-start users.** The test set begins in March 1998, so most test
  users have no training history. This is realistic for new users but limits what the ranking scores show.
- **Single split and seed,** with no confidence intervals or cross-validation.
- **Test-split tuning** of the content model's popularity weight, as noted above.
- **Explicit-rating model used for ranking.** Implicit-feedback methods (weighted ALS or BPR) are the natural
  next step.
- **Basic cold-start handling.** Unseen items cannot be recommended by the content or matrix factorization models.
- **Local orchestration only.** The plain-Python runner is sequential on one machine.
- **Fixed thresholds.** Validation limits, such as valid ages of 1 to 120, must be reviewed for each new dataset.

## 7. Conclusion

The pipeline covers every required stage and runs end to end from one command or from Airflow. Matrix
factorization is the best rating predictor and covers the most of the catalogue. For top-K ranking on this data,
the popularity baseline is nearly as good as the hybrid, and both outperform matrix factorization. This reflects
the time-based split. Genre features add little to ranking on MovieLens.

**GitHub repository:** https://github.com/nij-117-07/recommendation-pipeline
