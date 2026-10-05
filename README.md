# Housing Price Model

Walkthrough with all plots: https://nepalanurag.github.io/housing-price-model/

I modeled house prices on a dataset with size, bedrooms, age, distance to the city, crime rate, and renovation status.

Data provenance note: `housing_data.csv` has no source citation and several values look generated rather than measured (fractional square footage, sub-day house ages), so I treat it as synthetic. The pipeline comparisons are still fair on this data, but the absolute error levels are not real-world numbers.

The interesting part was a pipeline bake-off testing common modeling choices: complete-case analysis vs median/mode imputation, adding missingness indicators, normalization for Ridge/Lasso, outlier Winsorization, and log-transforming the target. The plain complete-case OLS baseline (RMSE around 46k) beat every imputation-based pipeline. I confirmed with Little's MCAR test (p = 0.223) that the missingness was completely at random, which is exactly why dropping rows worked best. Winsorizing outliers hurt the most, because the extremes were genuinely informative luxury homes.

### Follow-up: back-transformation sensitivity analysis

As a follow-up robustness check I re-examined how the Pipeline 4 log-models are scored on the dollar scale. The original comparison back-transforms log-price predictions with a plain `exp()`, which by Jensen's inequality under-predicts the conditional mean and can inflate the log-pipeline's dollar RMSE. `analysis/smearing_sensitivity.R` re-runs all pipelines on the same seeded split and adds Duan's smearing-corrected back-transform (`exp(pred) * mean(exp(residuals))`, Duan 1983) plus the parametric `exp(sigma^2/2)` correction for the P4 models, then re-ranks everything on the test set. The naive-exp runs stay as originally reported; the corrected variants are reported alongside in `analysis/smearing_comparison.csv`. Run: `Rscript analysis/smearing_sensitivity.R` (from the repo root).

## Files

- `housing-price-model.Rmd` - the analysis
- `housing-price-model.html` / `housing-price-model.docx` - rendered versions
- `housing_data.csv` - the dataset
