# Ensemble Learning for Loan Prediction

## Objective
The objective of this project is to explore and compare multiple ensemble learning techniques to predict the loan status of a person based on various parameters. I used default_of_credit_card_clients.xls dataset, perform data analysis, feature extraction, and apply various ensemble learning techniques to evaluate and compare their predictive performance.

## Dataset Selection
- **Dataset Requirements**:
  - used default_of_credit_card_clients.xls dataset.
  - Ensured the dataset includes sufficient features (independent variables) and a target variable (loan status) to enable classification.
- **Data Cleaning**:
  - Handled missing values, outliers, and irrelevant features.
- **Exploratory Data Analysis (EDA)**:
  - Visualize key insights from the data.

## Data Preprocessing
- **Feature Engineering and Extraction**:
  - Perform feature engineering to optimize model performance.
- **Normalization/Standardization**:
  - Normalize or standardize the data if required.
- **Data Splitting**:
  - Split the dataset into training and testing sets (use an 80/20 or 70/30 split).

## Ensemble Techniques

### Basic Ensemble Techniques
- **Max Voting**: Implement Max Voting using multiple classifiers and compare their predictions.
- **Averaging**: Implement the Averaging method where predictions from multiple classifiers are averaged.
- **Weighted Average**: Implement Weighted Averaging where classifiers are assigned weights based on their performance.

### Advanced Ensemble Techniques
- **Bagging**: Used Bagging techniques to improve model performance.
- **Boosting**: Used Boosting techniques to reduce bias and variance.
- **Stacking**: Implemented Stacking using multiple base models and a meta-model.

### Algorithms Based on Bagging and Boosting
1. **Bagging Meta-Estimator**: Implemented BaggingClassifier for classification. Train the model and evaluate its performance.
2. **Random Forest**: Trained a Random Forest Classifier and evaluate its accuracy, precision, recall, and F1 score.
3. **AdaBoost**: Implemented AdaBoost with decision trees as base models. Track and report its iterative improvement.
4. **Gradient Boosting (GBM)**: Used the Gradient Boosting Classifier to predict the loan status. Tune hyperparameters for better performance.
5. **XGBoost**: Trained an XGBoost Classifier. Use its built-in feature importance functionality to identify significant features.

## Analysis and Comparison
- **Performance Metrics**:
  - Compared the performance of each ensemble method in terms of accuracy, precision, recall, and F1 score.
- **Summary Table**:
  - Created a table showing the performance of all ensemble methods applied.
- **Best Technique**:
  - Identified which technique performed the best and provide reasoning for the observed performance.

## Implementation
- Writed a clear, well-commented code for each ensemble method.
- Used Python libraries such as:
  - NumPy
  - Pandas
  - Scikit-learn
  - Matplotlib
  - Seaborn

## Tools and Libraries
- **Python 3.x**
- **Libraries**:
  - NumPy
  - Pandas
  - Matplotlib
  - Seaborn
  - Scikit-learn
  - XGBoost
