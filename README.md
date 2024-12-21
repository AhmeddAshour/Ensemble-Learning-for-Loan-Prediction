# Ensemble Learning for Loan Prediction

## Objective
The objective of this project is to explore and compare ensemble learning techniques for predicting loan status based on various features. The project involves data analysis, preprocessing, and implementing multiple ensemble learning methods to evaluate their performance.

## Dataset Description
- **Dataset Source**: [UCI Credit Card Default Dataset](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
- **Number of Records**: 30,000
- **Number of Features**: 23
- **Target Variable**: Default Payment Next Month (Binary Classification)

## Data Exploration and Cleaning

1. **Missing Values**:
   - No missing values were found in the dataset.
  
2. **Outlier Detection**:
   - Outliers in numerical columns (LIMIT_BAL, AGE) were detected and managed using the Interquartile Range (IQR) method.
  
3. **Exploratory Data Analysis (EDA)**:
   - Conducted exploratory data analysis with visualizations:
     - Histograms and box plots to understand distributions of LIMIT_BAL and AGE.
     - Correlation heatmaps to identify relationships between payment and billing amounts.
     - Pair plots for feature relationships such as PAY_AMT1 through PAY_AMT6.

## Data Preprocessing

1. **Feature Engineering**:
   - Selected significant features such as payment history (PAY_0 to PAY_6), bill amounts (BILL_AMT1 to BILL_AMT6), and payment amounts (PAY_AMT1 to PAY_AMT6) based on correlation analysis.

2. **Normalization/Standardization**:
   - Applied standardization to numeric features (LIMIT_BAL, AGE, BILL_AMT1-6) to improve model performance.

3. **Data Splitting**:
   - Divided the dataset into training and testing sets with an 80/20 split.

## Ensemble Methods Used

### Basic Techniques

1. **Max Voting**:
   - Combined predictions from multiple classifiers (Logistic Regression, SVM, Decision Tree).
  
2. **Averaging**:
   - Averaged predictions across classifiers.

3. **Weighted Averaging**:
   - Assigned weights to predictions based on individual model performance.

### Advanced Techniques

1. **Bagging**:
   - Reduced variance using BaggingClassifier.

2. **Boosting**:
   - Implemented AdaBoost for iterative bias reduction.

3. **Stacking**:
   - Combined base models with a meta-model for final predictions.

## Bagging and Boosting Algorithms

1. **BaggingClassifier**:
   - Evaluated on accuracy, precision, recall, and F1 score.

2. **Random Forest**:
   - Built and evaluated a Random Forest Classifier with hyperparameter tuning.

3. **AdaBoost**:
   - Visualized performance improvements across iterations.

4. **Gradient Boosting**:
   - Tuned hyperparameters to enhance prediction accuracy.

5. **XGBoost**:
   - Identified significant features using XGBoost’s feature importance.

## Performance Metrics

- Accuracy
- Precision
- Recall
- F1 Score

## Performance Summary

| Technique           | Accuracy | Precision | Recall | F1 Score |
|---------------------|----------|-----------|--------|----------|
| BaggingClassifier    | 0.81     | 0.63      | 0.37   | 0.46     |
| RandomForest        | 0.82     | 0.64      | 0.37   | 0.47     |
| AdaBoost            | 0.82     | 0.68      | 0.32   | 0.43     |
| GradientBoosting    | 0.82     | 0.67      | 0.35   | 0.46     |
| XGBoost             | 0.82     | 0.64      | 0.38   | 0.47     |

## Best Model

- **Model**: Random Forest and XGBoost (tie)
- **Reason for Selection**: Both models achieved the highest F1 Score (0.47), balancing precision and recall effectively. XGBoost provided additional insights via feature importance, enhancing interpretability.

## Key Insights

- Ensemble methods outperform individual classifiers in most cases.
- Boosting techniques (AdaBoost, XGBoost) provided superior accuracy due to iterative improvements.
- Feature importance analysis highlighted key predictors such as payment history (PAY_0 to PAY_6) and recent bill amounts (BILL_AMT1 to BILL_AMT3).

## Conclusion

This project demonstrated the effectiveness of ensemble learning techniques in improving classification performance. Future improvements could involve:

1. Experimenting with additional advanced models (LightGBM, CatBoost).
2. Incorporating additional features to enrich the dataset.
3. Optimizing hyperparameters further for each model.

