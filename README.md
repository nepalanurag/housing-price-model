# Housing Price Model

Walkthrough with all plots: https://nepalanurag.github.io/housing-price-model/

I modeled house prices on a dataset with size, bedrooms, age, distance to the city, crime rate, and renovation status.

The interesting part was a pipeline bake-off testing common modeling choices: complete-case analysis vs median/mode imputation, adding missingness indicators, normalization for Ridge/Lasso, outlier Winsorization, and log-transforming the target. The plain complete-case OLS baseline (RMSE around 46k) beat every imputation-based pipeline. I confirmed with Little's MCAR test (p = 0.223) that the missingness was completely at random, which is exactly why dropping rows worked best. Winsorizing outliers hurt the most, because the extremes were genuinely informative luxury homes.

## Files

- `housing-price-model.Rmd` - the analysis
- `housing-price-model.html` / `housing-price-model.docx` - rendered versions
- `housing_data.csv` - the dataset
