# Dashboard data: Housing Price Model

Results from the pipeline bake-off in `housing-price-model.Rmd`. Values are the
rendered outputs of the analysis (cross-validated means with SDs, held-out test
scores, Little's MCAR test).

## Files

- `pipeline_cv_results.csv` — 10-fold cross-validated performance per pipeline/model.
  Columns: `pipeline` (P0_Baseline, P1_Imputed, P2_Indicators, P3_Normalized,
  P4_Full_Processing), `model` (OLS, Ridge, Lasso), `cv_rmse_mean`, `cv_rmse_sd`,
  `cv_mae_mean`, `cv_mae_sd`. RMSE/MAE are on the original dollar scale.
- `pipeline_test_results.csv` — performance on the held-out 20% test set.
  Columns: `pipeline`, `model`, `test_rmse`, `test_mae` (dollars).
- `mcar_test.json` — Little's MCAR test on the raw data: `statistic` (16.5),
  `df` (13), `p_value` (0.223), `missing_patterns` (3). p > 0.05, so missingness
  is consistent with completely at random.

Headline: the plain complete-case OLS baseline (P0) beat every imputation-based
pipeline on both CV and test RMSE/MAE.
