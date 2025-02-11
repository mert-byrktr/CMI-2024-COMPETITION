# Kaggle Competition: Predicting Severity Impairment Index (SII)

## Overview

This repository contains my approach to a Kaggle competition where the goal was to predict a participant's **Severity Impairment Index (SII)** based on physical activity and internet usage behavior data. The dataset originates from the **Healthy Brain Network (HBN)** study, which aims to find biological markers for diagnosing and treating mental health and learning disorders.

The competition data consists of:
- **Parquet files** containing accelerometer (actigraphy) series.
- **CSV files** with tabular data, including fitness assessments and internet usage behavior.

One major challenge is handling missing data, as many measures are absent for most participants. Additionally, the test set is hidden, requiring careful model validation.

## Approach

### 1. Data Preprocessing & Feature Engineering

- **Handling NaN values:** Applied various imputation strategies (SimpleImputer, KNNImputer).
- **Feature selection:** Dropped low-importance columns while retaining informative features.
- **Target recalculation:** Adjusted `sii` using **PCIAT scores** to correct inconsistencies.
- **Time Series Processing:**
  - Explored actigraphy features.
  - Considered interpolation for missing values.
  - Used external feature engineering references:
    - [Antonina’s EDA](https://www.kaggle.com/code/antoninadolgorukova/cmi-piu-actigraphy-data-eda/notebook)
    - [AmbrosM’s EDA](https://www.kaggle.com/code/ambrosm/piu-eda-which-makes-sense)

### 2. Modeling Strategies

#### **Base Models Tested:**
- **LightGBM** (LGBMRegressor)
- **XGBoost** (XGBRegressor)
- **CatBoost** (CatBoostRegressor)
- **Random Forest, Gradient Boosting** (Ensembles)

#### **Optimization Techniques:**
- **Quadratic Weighted Kappa (QWK) optimization:**
  - Implemented threshold tuning using [QWK metric optimization](https://www.kaggle.com/code/carlolepelaars/efficientnetb5-with-keras-aptos-2019#Metric-(Quadratic-Weighted-Kappa)-) and Gunees Notebook](https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use/discussion/551533)
  - Since we are dealing with threshold optimization for the weighted QWK model, it was important to find a good starting point for the **Nelder-Mead optimizer**. Different heuristics were evaluated for that.
  - Instead of initializing thresholds statically as `[0.5, 1.5, 2.5]` like in public notebooks, I dynamically initialized them using the following approach in my training pipeline:
    ```python
    oof_mask = df['prediction'].notna()
    oof_initial_thresholds = df.loc[oof_mask].groupby('target')['prediction'].mean().iloc[1:].values.tolist()
    oof_optimized_thresholds = minimize(
        metrics.rounding_optimization_function,
        x0=oof_initial_thresholds,
        args=(df.loc[oof_mask, target], df.loc[oof_mask, 'prediction']),
        method='Nelder-Mead'
    ).x
    ```
  - This helps to initialize thresholds dynamically while experimenting and leads to better convergence.
- **Cross-validation:**
  - Used **StratifiedKFold (N=5)** for stable performance evaluation.
  - Reduced standard deviation between folds.
- **Seed Selection:**
  - Seeds were selected after applying a **t-test** and choosing **p-values lower than 0.05** to ensure statistically significant improvements.
  - Performed multiple runs with different random seeds to enhance generalization.

### 3. Model Ensembling

- **Voting Regressor:** Combined predictions from multiple models for better robustness.
- **LGBM + CatBoost blending:**
  - **85% LightGBM + 15% CatBoost** gave the best results (QWK ~ 0.470).

### 4. Final Submission Strategy

- Predictions were **threshold-rounded** for classification.
- Used **mode-based ensembling** to finalize outputs.

## Challenges & Learnings

- **Target inconsistencies:** The survey-based target values were unreliable due to subjective responses. To address this, the target was recalculated using **Antonina's technique** to improve consistency and reduce noise in training. This technique involved:
  - **Recalculating the total PCIAT score** by summing the responses to all individual questions, ensuring that missing values were accounted for by estimating their impact.
  - **Assigning target labels** based on predefined score thresholds, ensuring consistency across different participant responses.
  - **Adjusting the thresholds dynamically** by considering possible maximum scores if missing values were replaced by their expected values.
  - This recalibration process led to a more stable and interpretable target variable, reducing noise caused by unreliable self-reported survey responses.

- **Target inconsistencies:** The survey-based target values were unreliable due to subjective responses. To address this, the target was recalculated using **Antonina's technique** to improve consistency and reduce noise in training.

- **Handling missing data:** Many features had >50% missing values, requiring careful imputation.
- **Actigraphy data complexities:** Time series features were difficult to interpret and required extensive preprocessing.

## Results

- Achieved significant improvement in SII predictions using QWK optimization.
- Best model blend: **85% LGBM + 15% CatBoost**.
- Final submission outperformed baseline models.

## Conclusion

Competing against top **Kaggle data scientists**, including the **NVIDIA team**, was a fantastic experience. The competition provided deep insights into **time series modeling, missing data handling, and ensemble learning techniques**.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Developed by [Your Name]

## Contact

For discussions or improvements, feel free to reach out!

