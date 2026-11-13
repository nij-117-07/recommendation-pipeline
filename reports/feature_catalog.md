# Feature Catalog

Reference time for time-based features: `2024-10-20 05:28:13+00:00` (last training timestamp).

Features are computed from the training split only, to avoid leakage into the test period.

| Feature | Entity | Type | Source | Description |
|---|---|---|---|---|
| `user_avg_rating` | user | float | train interactions | Mean explicit rating given by the user. Global mean if the user has no training history. |
| `user_activity_count` | user | int | train interactions | Number of training interactions by the user (activity level). |
| `user_preferred_category` | user | string | train interactions + item genres | Genre that occurs most often among the items the user interacted with (ties broken alphabetically). |
| `user_days_since_last_interaction` | user | float | train interactions | Days between the feature reference time (last training timestamp) and the user's last interaction. -1 if no history. |
| `item_popularity_count` | item | int | train interactions | Number of training interactions with the item. |
| `item_positive_count` | item | int | train interactions | Number of training interactions where rating >= relevance threshold (positive feedback). |
| `item_avg_rating` | item | float | train interactions | Mean rating of the item in training. Global mean if the item is unseen. |
| `item_primary_genre` | item | string | items table | First genre (alphabetical) of the item. |
| `item_n_genres` | item | int | items table | Number of genres assigned to the item. |
| `item_age_years` | item | int | items table | Reference year minus release year. |
| `interaction_hour_of_day` | interaction | int | train interactions | Hour (0-23 UTC) at which the interaction happened. |
| `interaction_day_of_week` | interaction | int | train interactions | Day of week of the interaction (0 = Monday). |
| `label_is_positive` | interaction | int | train interactions | Target label (not a model input): 1 if rating >= relevance threshold, else 0. |
| `item_genre_<genre>` | item | int | items table | Multi-hot genre flag, one column per genre (10 columns: Action, Animation, Children, Comedy, Documentary, Drama, Horror, Romance, Sci-Fi, Thriller). |
| `user_genre_affinity_<genre>` | user | float | train interactions + item genres | Share of the user's positive training interactions (rating >= threshold) that fall in this genre. Shares sum to 1; 0 for users with no positive interactions. |

## Stored feature tables

| File (data/features/) | Rows | Columns | Key |
|---|---:|---:|---|
| user_features.parquet | 792 | 14 | user_id |
| item_features.parquet | 396 | 16 | item_id |
| interaction_features.parquet | 43750 | 5 | (user_id, item_id) |

Cold-start users (no training history): 0
Cold-start items (no training interactions): 0
