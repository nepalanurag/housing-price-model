# Back-transformation sensitivity analysis for housing-price-model.
#
# Follow-up robustness check on the Pipeline 4 log models. The original
# analysis back-transforms log-price predictions with a naive exp(), which by
# Jensen's inequality predicts E[log Y] mapped through exp instead of E[Y]:
# it systematically under-predicts on the dollar scale and inflates the
# log-pipeline's dollar RMSE. This script replicates the P4 preprocessing
# chain (housing-price-model.Rmd, Pipeline 4 section) and evaluates the fitted
# log models under three back-transformations:
#   naive     : exp(pred_log)
#   smearing  : exp(pred_log) * mean(exp(train residuals))   -- Duan (1983)
#   normal    : exp(pred_log) * exp(sigma^2 / 2)             -- parametric, lognormal
# It also re-runs the other pipelines on the same split so the corrected P4
# can be re-ranked against P0-P3 apples-to-apples.
#
# Replicates the Rmd chain with a fresh seeded 80/20 split (set.seed(42)); the
# Rmd itself did not seed, so absolute numbers differ slightly from the
# rendered tables. The comparison of interest (naive vs corrected, same data,
# same fits) is within-run.
#
# Run: Rscript analysis/smearing_sensitivity.R   (from the repo root)
# Writes: analysis/smearing_comparison.csv, analysis/smearing_summary.json
suppressMessages({
  library(caret)
  library(glmnet)
  library(dplyr)
})

SEED <- 42
args <- commandArgs(trailingOnly = FALSE)
file_arg <- sub("^--file=", "", args[grepl("^--file=", args)])
HERE <- if (length(file_arg)) dirname(normalizePath(file_arg)) else getwd()
ROOT <- dirname(HERE)

raw <- read.csv(file.path(ROOT, "housing_data.csv"))
raw$renovated <- as.factor(raw$renovated)

set.seed(SEED)
split_indices <- createDataPartition(raw$price, p = 0.8, list = FALSE)
housing_train <- raw[split_indices, ]
housing_test  <- raw[-split_indices, ]

impute_data <- function(target_df, reference_df) {
  res_df <- target_df
  for (nm in names(res_df)) {
    if (is.numeric(res_df[[nm]])) {
      res_df[[nm]][is.na(res_df[[nm]])] <- median(reference_df[[nm]], na.rm = TRUE)
    } else {
      mode_val <- names(sort(table(reference_df[[nm]]), decreasing = TRUE))[1]
      res_df[[nm]][is.na(res_df[[nm]])] <- mode_val
    }
  }
  res_df
}

cv_control <- trainControl(method = "cv", number = 10)
exp_summary <- function(data, lev = NULL, model = NULL) {
  obs <- exp(data$obs); pred <- exp(data$pred)
  c(RMSE = RMSE(pred, obs), Rsquared = R2(pred, obs), MAE = MAE(pred, obs))
}
cv_control_log <- trainControl(method = "cv", number = 10, summaryFunction = exp_summary)

test_perf <- function(preds, obs, label, model_type, backtransform) {
  data.frame(Pipeline = label, Model = model_type, Backtransform = backtransform,
             Test_RMSE = round(RMSE(preds, obs), 2),
             Test_MAE = round(MAE(preds, obs), 2))
}

results <- list()
i <- 1

# --- P0: complete-case baseline ---
train_clean <- housing_train %>% dplyr::select(-price_per_sqft)
test_clean  <- housing_test %>% dplyr::select(-price_per_sqft)
train_p0 <- na.omit(train_clean); test_p0 <- na.omit(test_clean)
set.seed(SEED)
f <- train(price ~ ., data = train_p0, method = "lm", trControl = cv_control)
results[[i <- i + 1]] <- test_perf(predict(f, test_p0), test_p0$price, "P0_Baseline", "OLS", "none")
set.seed(SEED)
f <- train(price ~ ., data = train_p0, method = "glmnet",
           tuneGrid = expand.grid(alpha = 0, lambda = seq(0.1, 10, length = 5)),
           trControl = cv_control)
results[[i <- i + 1]] <- test_perf(predict(f, test_p0), test_p0$price, "P0_Baseline", "Ridge", "none")

# --- P1: imputation ---
train_p1 <- impute_data(train_clean, train_clean)
test_p1  <- impute_data(test_clean, train_clean)
set.seed(SEED)
f <- train(price ~ ., data = train_p1, method = "lm", trControl = cv_control)
results[[i <- i + 1]] <- test_perf(predict(f, test_p1), test_p1$price, "P1_Imputed", "OLS", "none")
set.seed(SEED)
f <- train(price ~ ., data = train_p1, method = "glmnet",
           tuneGrid = expand.grid(alpha = 0, lambda = seq(0.1, 10, length = 5)),
           trControl = cv_control)
results[[i <- i + 1]] <- test_perf(predict(f, test_p1), test_p1$price, "P1_Imputed", "Ridge", "none")

# --- P2: imputation + missingness indicators ---
add_indicators <- function(df) mutate(df, across(everything(), is.na, .names = "na_{.col}"))
train_p2_raw <- add_indicators(train_clean)
test_p2_raw  <- add_indicators(test_clean)
train_p2_imp <- impute_data(train_p2_raw, train_p2_raw)
test_p2_imp  <- impute_data(test_p2_raw, train_p2_raw)
nzv_cols <- nearZeroVar(train_p2_imp)
train_p2 <- if (length(nzv_cols) > 0) train_p2_imp[, -nzv_cols] else train_p2_imp
test_p2  <- if (length(nzv_cols) > 0) test_p2_imp[, -nzv_cols] else test_p2_imp
set.seed(SEED)
f <- train(price ~ ., data = train_p2, method = "lm", trControl = cv_control)
results[[i <- i + 1]] <- test_perf(predict(f, test_p2), test_p2$price, "P2_Indicators", "OLS", "none")
set.seed(SEED)
f <- train(price ~ ., data = train_p2, method = "glmnet",
           tuneGrid = expand.grid(alpha = 0, lambda = seq(0.1, 10, length = 5)),
           trControl = cv_control)
results[[i <- i + 1]] <- test_perf(predict(f, test_p2), test_p2$price, "P2_Indicators", "Ridge", "none")

# --- P3: standardized ---
num_cols <- names(train_p2)[sapply(train_p2, is.numeric) & names(train_p2) != "price"]
train_means <- colMeans(train_p2[, num_cols]); train_sds <- apply(train_p2[, num_cols], 2, sd)
train_p3 <- train_p2; test_p3 <- test_p2
train_p3[num_cols] <- scale(train_p2[num_cols], center = train_means, scale = train_sds)
test_p3[num_cols]  <- scale(test_p2[num_cols], center = train_means, scale = train_sds)
set.seed(SEED)
f3r <- train(price ~ ., data = train_p3, method = "glmnet",
             tuneGrid = expand.grid(alpha = 0, lambda = seq(0.1, 10, length = 5)),
             trControl = cv_control)
results[[i <- i + 1]] <- test_perf(predict(f3r, test_p3), test_p3$price, "P3_Normalized", "Ridge", "none")
set.seed(SEED)
f3l <- train(price ~ ., data = train_p3, method = "glmnet",
             tuneGrid = expand.grid(alpha = 1, lambda = seq(0.1, 10, length = 5)),
             trControl = cv_control)
results[[i <- i + 1]] <- test_perf(predict(f3l, test_p3), test_p3$price, "P3_Normalized", "Lasso", "none")

# --- P4: log transform + winsorize + re-standardize ---
train_p4 <- train_p2 %>% mutate(price = log(price), crime_rate = log(crime_rate + 1))
test_p4  <- test_p2  %>% mutate(crime_rate = log(crime_rate + 1))
for (nm in num_cols) {
  qnts <- quantile(train_p4[[nm]], probs = c(.05, .95), na.rm = TRUE)
  train_p4[[nm]] <- pmin(pmax(train_p4[[nm]], qnts[1]), qnts[2])
  test_p4[[nm]]  <- pmin(pmax(test_p4[[nm]], qnts[1]), qnts[2])
}
train_p4_means <- colMeans(train_p4[, num_cols]); train_p4_sds <- apply(train_p4[, num_cols], 2, sd)
train_p4[num_cols] <- scale(train_p4[num_cols], center = train_p4_means, scale = train_p4_sds)
test_p4[num_cols]  <- scale(test_p4[num_cols], center = train_p4_means, scale = train_p4_sds)

set.seed(SEED)
f4r <- train(price ~ ., data = train_p4, method = "glmnet",
             tuneGrid = expand.grid(alpha = 0, lambda = seq(0.001, 0.1, length = 5)),
             trControl = cv_control_log)
set.seed(SEED)
f4l <- train(price ~ ., data = train_p4, method = "glmnet",
             tuneGrid = expand.grid(alpha = 1, lambda = seq(0.001, 0.1, length = 5)),
             trControl = cv_control_log)

obs_test <- test_p2$price  # test_p2$price was never logged

for (nm in c("Ridge", "Lasso")) {
  fit <- if (nm == "Ridge") f4r else f4l
  pred_log <- as.numeric(predict(fit, newdata = test_p4))
  # naive: exp() only (as in the original analysis)
  results[[i <- i + 1]] <- test_perf(exp(pred_log), obs_test,
                                     "P4_Full_Processing", nm, "naive exp()")
  # Duan (1983) smearing: E[Y|x] = exp(x'beta) * E[exp(eps)], estimated on
  # the log-scale training residuals
  resid_log <- train_p4$price - as.numeric(predict(fit, newdata = train_p4))
  smear <- mean(exp(resid_log))
  results[[i <- i + 1]] <- test_perf(exp(pred_log) * smear, obs_test,
                                     "P4_Full_Processing", nm, "Duan smearing")
  # parametric lognormal correction for reference
  sigma2 <- mean(resid_log^2)
  results[[i <- i + 1]] <- test_perf(exp(pred_log) * exp(sigma2 / 2), obs_test,
                                     "P4_Full_Processing", nm, "exp(sigma2/2)")
  if (nm == "Ridge") smear_ridge <- smear else smear_lasso <- smear
}

final <- bind_rows(results)
final <- final[order(final$Test_RMSE), ]
rownames(final) <- NULL
write.csv(final, file.path(HERE, "smearing_comparison.csv"), row.names = FALSE)

summary_out <- list(
  note = "Same data and fits for all three P4 back-transforms; only the back-transform varies.",
  smearing_factor_ridge = smear_ridge,
  smearing_factor_lasso = smear_lasso,
  implied_mean_bias_naive_ridge = 1 / smear_ridge,
  implied_mean_bias_naive_lasso = 1 / smear_lasso,
  ranked_test_rmse = lapply(seq_len(nrow(final)), function(r) as.list(final[r, ]))
)
jsonlite::write_json(summary_out, file.path(HERE, "smearing_summary.json"),
                     auto_unbox = TRUE, digits = 6, pretty = TRUE)
print(final, row.names = FALSE)
cat("\nSmearing factor (ridge):", round(smear_ridge, 4),
    " (lasso):", round(smear_lasso, 4), "\n")
