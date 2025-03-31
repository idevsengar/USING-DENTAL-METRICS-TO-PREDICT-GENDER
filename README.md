# USING-DENTAL-METRICS-TO-PREDICT-GENDER


# Predicting Gender Using Dental Metrics

## Objective

The objective of this project is to analyze a dataset containing various dental measurements and build a machine learning model to predict the gender (Male/Female) of a person based on these dental features.

## Dataset Overview

- The dataset consists of 1,100 records.
- It includes various dental measurements, such as intercanine distance, canine width, and canine index values.
- The target variable is "Gender," which we will convert into numerical values for model building.

## Steps Followed

### 1. Problem Definition

We aim to predict the gender (Male/Female) of a person based on dental measurements. This involves building and evaluating machine learning models using various dental features.

### 2. Data Preprocessing

**Data Cleaning**:
- Checked for missing values and handled them by dropping columns with missing data.
- Dropped irrelevant columns like `'Sample ID'` and `'Sl No'`.

**Encoding Target Variable**:
- Encoded the target variable `'Gender'` using `LabelEncoder()`, where "Male" was encoded as 1 and "Female" as 0.

**Feature Scaling**:
- Standardized the features using `StandardScaler()` to ensure they were on the same scale.

### 3. Exploratory Data Analysis (EDA)

- Plotted histograms for feature distributions to understand the data better.
- Generated a correlation heatmap to examine relationships between features.

### 4. Model Building

We trained the following machine learning models to predict gender:

- **Logistic Regression**
- **Decision Tree Classifier**
- **Random Forest Classifier**
- **XGBoost Classifier**

### 5. Model Evaluation

Models were evaluated using:

- **Accuracy**: To measure how well each model performed on the test data.
- **Confusion Matrix**: To visualize true positives, false positives, true negatives, and false negatives.
- **Classification Report**: To assess precision, recall, and F1-score.
- **ROC Curve & AUC**: To visualize the trade-off between true positive rate and false positive rate.

### 6. Final Model Selection

The **Random Forest Classifier** performed the best, with an accuracy of 91%, making it the chosen model for predicting gender.

### 7. Conclusion

- **Best Model**: Random Forest Classifier with an accuracy of 91%.
- The Random Forest model is suitable for real-world applications in predicting gender based on dental metrics.

## Requirements

To run the code and reproduce the results, you will need the following Python libraries:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost

You can install these libraries using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost

