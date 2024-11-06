# COVID-19-ML-ANALYSIS
Predicting Antibody-Inducing Activity in COVID-19 B-cell Epitopes using Machine Learning Models


## Project Overview

This project aims to predict antibody-inducing activity of B-cell epitopes in COVID-19 using various machine learning models. The results could significantly inform vaccine development by identifying peptide sequences with high antibody valency.

## Dataset
The data was obtained from [IEDB](https://www.iedb.org/) and [UniProt](https://www.uniprot.org/) and is publicly available on [Kaggle](https://www.kaggle.com/). It includes:
- **Rows**: 40,807
- **Variables**: 14 (11 continuous features, 3 categorical features removed for model input)
- **Target**: Binary variable (1 = antibody-inducing, 0 = non-inducing)

## Business Problem
The goal is to develop a machine learning model to accurately predict antibody-inducing activity in peptides, aiding in vaccine research.

---

## Project Steps

### 1. Data Preparation and Exploration
   - **Feature Engineering**: Added peptide length as a feature.
   - **Normality Tests**: Determined most variables were non-normally distributed.
   - **Outlier Analysis**: Left significant outliers to maintain data integrity.
   - **Correlation Analysis**: Weak correlation for most features except for strong correlation between `start_position` and `end_position`.

### 2. Modeling Approaches
   - **Models Used**: Logistic Regression, SVM (Radial Kernel), and Random Forest
   - **Evaluation Metrics**: Accuracy, Precision, Recall, F1 Score, and AUC

### 3. Model Tuning and Selection
   - **Hyperparameter Tuning**:
     - Logistic Regression: Stepwise selection (AIC) and tuning with cross-validation.
     - SVM: Grid search for `alpha` and `lambda` parameters.
     - Random Forest: Class weight adjustments due to class imbalance.
   - **Best Model**: Random Forest showed the highest accuracy, recall, and precision, making it the optimal choice for vaccine development needs.

### 4. Ensemble Modeling
   - Created an ensemble of Logistic Regression, SVM, and Random Forest, but Random Forest performed best as a standalone model.

### 5. Bagging Random Forest for Final Prediction
   - Applied bagging to reduce overfitting, improving robustness.

---

## Key Results
- **Best Performing Model**: Random Forest with 87% accuracy and high precision, crucial for vaccine development.
- **Insights**: The model effectively identifies antibody-inducing peptides with minimal false positives.

---

## Lessons and Future Work

- **Key Findings**:
   - Data transformation and hyperparameter tuning are essential for improving model performance by ensuring that each feature is appropriately scaled and that model parameters are optimized to best capture patterns within the data.
   - Class imbalance affects accuracy, so precision and recall are emphasized. In the context of vaccine development, precision is particularly important as it measures the proportion of correctly identified antibody-inducing peptides among those predicted to be positive. High precision minimizes false positives, which is crucial in vaccine research, as incorrectly identifying non-inducing peptides as inducing could lead to ineffective or even harmful vaccine candidates.

- **Future Directions**:
   - Address class imbalance directly to improve model robustness. Techniques such as SMOTE (Synthetic Minority Oversampling Technique), ADASYN (Adaptive Synthetic Sampling), or using alternative metrics (e.g., F2 score) could help the model learn from the minority class more effectively, which might improve both recall and the overall model's ability to identify true antibody-inducing peptides.
   - Further explore advanced feature engineering, such as additional physicochemical properties or sequence-based features, to strengthen predictive accuracy.
   - Experiment with other machine learning algorithms, such as gradient boosting, or ensemble methods that specifically handle imbalanced data.
   - Continue to refine hyperparameter tuning through cross-validation, and evaluate the model on new, unseen data for generalizability.

---

## Data Source
The datasets are available on Kaggle:
- [COVID-19 Dataset](https://www.kaggle.com/link-to-covid-dataset)
- [B-cell Epitope Dataset](https://www.kaggle.com/link-to-bcell-dataset)
- [SARS Dataset](https://www.kaggle.com/link-to-sars-dataset)

## How to Run
1. **Install Requirements**: `R` libraries used include `caret`, `randomForest`, `e1071`, and `dplyr`.
3. **Predict**: Use the ensemble or Random Forest model on new, unlabeled data.

---
