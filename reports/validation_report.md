# Data Validation Report

## Row counts

| Table | Raw rows | Valid rows | Rejected rows |
|---|---:|---:|---:|
| users | 943 | 943 | 0 |
| items | 1682 | 1682 | 0 |
| interactions | 100000 | 100000 | 0 |

## Rule results

| Table | Check | Severity | Rows flagged |
|---|---|---|---:|
| users | missing_user_id | error | 0 |
| users | invalid_user_id_type | error | 0 |
| users | duplicate_user_id | error | 0 |
| users | invalid_age_type | error | 0 |
| users | invalid_age_range | error | 0 |
| users | missing_age | warning | 0 |
| items | missing_item_id | error | 0 |
| items | invalid_item_id_type | error | 0 |
| items | duplicate_item_id | error | 0 |
| items | invalid_release_year_type | error | 0 |
| items | invalid_release_year_range | error | 0 |
| items | missing_genres | warning | 0 |
| interactions | missing_user_id | error | 0 |
| interactions | missing_item_id | error | 0 |
| interactions | missing_rating | error | 0 |
| interactions | missing_timestamp | error | 0 |
| interactions | invalid_user_id_type | error | 0 |
| interactions | invalid_item_id_type | error | 0 |
| interactions | invalid_rating_type | error | 0 |
| interactions | invalid_rating_value | error | 0 |
| interactions | invalid_timestamp | error | 0 |
| interactions | future_timestamp | error | 0 |
| interactions | duplicate_record | error | 0 |
| interactions | orphan_user_id | error | 0 |
| interactions | orphan_item_id | error | 0 |
| interactions | repeated_user_item_pair | warning | 0 |

Rejected rows are written to `data/errors/<table>_errors.csv` with an `error_reasons` column.
Warnings do not reject rows; they are handled during preparation (imputation or de-duplication).
