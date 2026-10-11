# Housing Price Model

I modeled house prices on a dataset with size, bedrooms, age, distance to
the city, crime rate, and renovation status. The interesting part is a
pipeline bake-off: five preprocessing pipelines, each tested with OLS,
Ridge, and Lasso under 10-fold CV, to see which standard modeling choices
actually help.

**Full walkthrough with all plots:** https://nepalanurag.github.io/housing-price-model/

## Key results

The plain baseline won. Pipeline 0 OLS (complete-case analysis, no
imputation, no scaling) had the lowest CV RMSE (45,814.87, MAE 31,707.57)
and stayed best on the held-out test set (RMSE 63,202.49, MAE 36,907.90).
Every fancier pipeline did worse:

- **Imputation did not help.** Median/mode imputation (P1) pushed OLS CV
  RMSE from ~46k to ~51k. Dropping incomplete rows worked best, which
  makes sense given the missingness test: Little's MCAR test came back
  p = 0.223, so the missing values are completely at random, and dropping
  rows leaves a smaller but unbiased sample while imputation compresses
  variance and adds noise.
- **Missingness indicators added nothing.** P1 to P2 (imputation +
  indicators) pushed OLS RMSE from ~51k to ~53k. The indicators carried no
  signal about price, exactly what MCAR predicts.
- **Normalization did not save the regularized models.** It helped Lasso
  slightly relative to the imputation pipelines, but everything stayed
  worse than the baseline. Standardizing mutes the natural importance of
  large-scale features like square footage while the penalty bites harder.
- **Winsorizing hurt the most.** The full-preprocessing pipeline (P4) came
  in worst at ~58k RMSE. The extremes are genuinely informative luxury
  homes; capping them forced systematic under-prediction at the top end.
- **The log transform backfired on the dollar scale.** Back-transforming
  small log-scale errors into dollars blew them up (MAE ~39k). The price
  relationship in this data looks additive, not multiplicative.

### Follow-up: back-transformation sensitivity

Since the P4 log-models are scored back on the dollar scale with a plain
`exp()`, which by Jensen's inequality under-predicts the conditional mean,
I checked whether that naive back-transform was inflating the log
pipeline's RMSE (`analysis/smearing_sensitivity.R`, results in
`analysis/smearing_comparison.csv`). Adding Duan's smearing correction
(smearing factors 1.0258 Ridge, 1.0255 Lasso) and the parametric
`exp(sigma^2/2)` correction moved test RMSE slightly the wrong way
(Ridge 59,362 to 59,881; Lasso 62,106 to 63,129) and confirmed the
original ranking instead of overturning it. The log transform's poor
showing is not a back-transform artifact; Pipeline 4 stays last either
way.

## Data provenance

`housing_data.csv` has no source citation and several values look
generated rather than measured (fractional square footage, sub-day house
ages, prices to the cent with no heaping), so I treat it as synthetic. The
bake-off comparisons are fair on this data, but the absolute error levels
are not real-world numbers.

## How to run

The analysis is an R Markdown document (R, packages: tidyverse, caret,
glmnet, corrplot, gridExtra, naniar):

```r
rmarkdown::render("housing-price-model.Rmd")
```

The back-transform sensitivity check:

```bash
Rscript analysis/smearing_sensitivity.R   # from the repo root
```

## How the bake-off works

- **Split first, explore later.** The 80/20 train/test split happens
  before any exploration, so nothing from the test set leaks into the EDA
  or the preprocessing. Imputation and scaling statistics are computed on
  the training set only.
- **The five pipelines.** P0: complete-case OLS baseline. P1: median/mode
  imputation. P2: imputation plus missingness indicators. P3: imputation
  plus normalization for Ridge/Lasso. P4: full preprocessing, log target,
  Winsorization of outliers at training quantiles, rescale.
- **Models and scoring.** Each pipeline is fit with OLS, Ridge, and Lasso
  under 10-fold CV via caret, with the best CV row picked by RMSE, then
  scored on the unseen test set. Log-model predictions are back-transformed
  with `exp()` before scoring on the dollar scale.
- **Two mistakes avoided on purpose.** `price_per_sqft` correlates almost
  perfectly with price because it is computed from price and size, so it is
  dropped as target leakage. And imputation/scaling stats are never
  computed on the full dataset before the split, which would leak test
  information into training.

## Files

- `housing-price-model.Rmd` - the analysis
- `housing-price-model.html` / `housing-price-model.docx` - rendered versions
- `housing_data.csv` - the dataset
- `analysis/smearing_sensitivity.R` - back-transform sensitivity check
- `analysis/smearing_comparison.csv` - sensitivity results
- `session_info.txt` - R session info for reproducibility
