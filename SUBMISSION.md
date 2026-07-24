# Build an ML Pipeline for Short-Term Rental Prices in NYC

An end-to-end ML pipeline built with MLflow and Weights & Biases that cleans, tests, splits, and trains a Random Forest model to predict Airbnb rental prices in New York City, then verifies and re-runs the pipeline on new data.

## Submission Links

- **W&B Project:** https://wandb.ai/emalet07-western-governors-university/nyc_airbnb
- **GitHub Repository:** https://github.com/Emyroyale/Project-Build-an-ML-Pipeline-Starter
- **Release used for grading:** [1.0.1](https://github.com/Emyroyale/Project-Build-an-ML-Pipeline-Starter/releases/tag/1.0.1)

## Pipeline Overview

The pipeline runs the following steps, each logged as an artifact in W&B:

1. **download** – fetches the raw sample data
2. **basic_cleaning** – removes price outliers and (as of v1.0.1) filters out data points outside NYC's geographic boundaries
3. **data_check** – runs automated tests (column names, price range, row count, geo boundaries, neighborhood distribution) against a tagged reference dataset
4. **data_split** – splits cleaned data into train/validation and test sets
5. **train_random_forest** – trains a Random Forest regression model on the processed features
6. **test_regression_model** – evaluates the production model against the held-out test set

## Model Results

| Metric | Validation | Test |
|---|---|---|
| MAE | 34.13 | 33.85 |
| R² | 0.552 | 0.564 |

Test performance is comparable to (slightly better than) validation, indicating no overfitting.

Best hyperparameters (tagged `prod` in W&B):
- `n_estimators`: 100
- `max_depth`: 15

## Release History

- **v1.0.0** – Initial complete pipeline (EDA through model selection and test verification)
- **v1.0.1** – Added longitude/latitude filtering in `basic_cleaning` to handle out-of-bounds data points found when running the pipeline on `sample2.csv`; re-trained model achieved MAE 32.42 / R² 0.580 on the new data

## Possible Future Improvements

- Explore additional feature engineering on the `name` column beyond TF-IDF (e.g. sentiment or keyword flags)
- Try gradient-boosted models (XGBoost/LightGBM) alongside Random Forest for comparison
- Expand the hyperparameter search space and use a smarter search strategy (e.g. Bayesian optimization) instead of grid search
- Add data drift monitoring so `test_similar_neigh_distrib` failures trigger alerts rather than only pipeline failures
