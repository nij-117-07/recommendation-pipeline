# Model Evaluation Results

K = 10, relevance threshold = rating >= 4. Evaluated users: 786 (0 cold-start). RMSE/MAE use 10938 test ratings with known users and items.

| Model | Precision@10 | Recall@10 | NDCG@10 | Coverage | Diversity | RMSE | MAE |
|---|---|---|---|---|---|---|---|
| popularity_baseline | 0.1018 | 0.2151 | 0.2038 | 0.0833 | 0.8810 | 1.1381 | 0.8988 |
| genre_content_recommender | 0.1051 | 0.2221 | 0.2072 | 0.1288 | 0.8655 | 1.1406 | 0.9119 |
| biased_matrix_factorization | 0.0602 | 0.1231 | 0.1020 | 0.8737 | 0.8737 | 0.7670 | 0.6184 |
