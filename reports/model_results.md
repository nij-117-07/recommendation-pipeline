# Model Evaluation Results

K = 10, relevance threshold = rating >= 4. Evaluated users: 290 (192 cold-start). RMSE/MAE use 2864 test ratings with known users and items.

| Model | Precision@10 | Recall@10 | NDCG@10 | Coverage | Diversity | RMSE | MAE |
|---|---|---|---|---|---|---|---|
| popularity_baseline | 0.2445 | 0.0826 | 0.2703 | 0.0440 | 0.7854 | 1.0366 | 0.8330 |
| genre_content_recommender | 0.2476 | 0.0849 | 0.2740 | 0.0470 | 0.7617 | 1.1805 | 0.9573 |
| biased_matrix_factorization | 0.2259 | 0.0701 | 0.2497 | 0.0957 | 0.7845 | 0.9723 | 0.7671 |
