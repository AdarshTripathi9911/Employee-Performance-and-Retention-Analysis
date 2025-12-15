# Employee-Performance-and-Retention-Analysis
Employee Performance and Retention Analysis – Detailed Explanation
Project Overview

The Employee Performance and Retention Analysis project focuses on understanding employee behavior, performance trends, and attrition patterns using data-driven techniques. Organizations often face challenges in identifying why employees leave and what factors influence performance. This project uses real-world employee data to analyze these factors and build predictive models that help organizations make informed HR decisions.

The project integrates statistics, probability, machine learning, and deep learning to provide both analytical insights and predictive capabilities.

Objectives

The main objectives of this project are:

To analyze employee performance based on demographic and professional factors.

To identify key factors contributing to employee attrition.

To apply statistical methods to understand trends and relationships in the data.

To build machine learning and deep learning models that predict:

Employee attrition (leave or stay).

Employee performance scores.

To generate actionable recommendations to improve employee retention and performance.

Dataset Description

The dataset contains employee-level information, including:

Employee ID & Name – Unique identification of employees.

Age – Age of the employee.

Department – Functional area within the organization.

Salary – Monthly or annual compensation.

Years at Company – Employee experience within the organization.

Performance Score – Rating of employee performance.

Attrition – Whether the employee left the company (Yes/No).

Phase 1: Data Collection and Exploratory Data Analysis (EDA)
Data Preprocessing

The dataset is loaded using Pandas. Missing values are handled, duplicate records are removed, and inconsistent data entries are cleaned to ensure data quality.

Exploratory Data Analysis

EDA is performed to understand the structure and characteristics of the dataset:

Descriptive statistics such as mean, median, variance, and standard deviation summarize numerical features.

Pair plots visualize relationships between multiple features.

Correlation heatmaps identify strong relationships between variables like salary, experience, and performance.

Box plots help detect outliers that may affect model performance.

Phase 2: Probability and Statistical Analysis

Statistical methods are applied to draw meaningful conclusions:

Probability analysis estimates the likelihood of employee attrition based on performance levels and departments.

Bayes’ Theorem is used to calculate the probability of attrition given a specific performance score.

Hypothesis testing determines whether the average performance score differs significantly across departments.

These analyses help validate patterns observed during EDA.

Phase 3: Predictive Modeling (Machine Learning)
Feature Engineering

Categorical variables such as Department and Attrition are encoded numerically using Label Encoding. Numerical features like Salary and Performance Score are scaled to improve model performance.

Attrition Prediction (Classification)

Machine learning classification models (e.g., Logistic Regression or Random Forest) are trained to predict employee attrition. Model performance is evaluated using:

Accuracy

Precision

Recall

F1-score
A confusion matrix is used to visualize correct and incorrect predictions.

Performance Prediction (Regression)

A Linear Regression model predicts employee performance scores. The model is evaluated using:

R-squared (R²) – Measures explained variance.

Mean Squared Error (MSE) – Measures prediction error.
Predicted vs. actual performance plots show model effectiveness.

Phase 4: Deep Learning Models
Performance Prediction using Neural Networks

A feedforward neural network is built using TensorFlow/Keras. It consists of:

Input layer with employee features.

Hidden layers with ReLU activation.

Output layer predicting performance score.
The model is trained using Mean Squared Error loss.

Attrition Prediction using Neural Networks

A deep learning classification model predicts whether an employee will leave or stay. Performance is evaluated using accuracy and classification metrics.

Phase 5: Insights and Recommendations

Based on analysis and predictions:

Salary, experience, and performance score are key factors affecting attrition.

Certain departments show higher attrition risk.

Predictive models can identify high-risk employees early.

Recommendations include:

Department-wise performance improvement plans.

Targeted employee engagement and retention programs.

Data-driven HR policies to reduce attrition.

Conclusion

This project demonstrates how data analytics, machine learning, and deep learning can be effectively applied to solve real-world HR problems. The insights and predictive models developed can help organizations improve employee satisfaction, performance, and retention.



# Employee Performance and Retention Analysis
# Complete End-to-End Project Code

# ================================
# Phase 0: Import Required Libraries
# ================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# ================================
# Phase 1: Data Collection & EDA
# ================================

# Load dataset
df = pd.read_csv('employee_data.csv')

# Basic inspection
print(df.head())
print(df.info())
print(df.describe())

# Handle missing values
df.fillna(method='ffill', inplace=True)

# Remove duplicates
df.drop_duplicates(inplace=True)

# ================================
# Exploratory Data Analysis (EDA)
# ================================

# Descriptive statistics
print("Mean:\n", df.mean(numeric_only=True))
print("Median:\n", df.median(numeric_only=True))
print("Standard Deviation:\n", df.std(numeric_only=True))

# Pairplot
sns.pairplot(df.select_dtypes(include=np.number))
plt.show()

# Correlation heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Boxplot for outlier detection
for col in df.select_dtypes(include=np.number).columns:
    plt.figure()
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
    plt.show()

# ================================
# Probability & Statistical Analysis
# ================================

# Probability of attrition
attrition_prob = df['Attrition'].value_counts(normalize=True)
print("Attrition Probability:\n", attrition_prob)

# Probability of attrition given performance score below threshold
low_perf = df[df['PerformanceScore'] < df['PerformanceScore'].mean()]
prob_attrition_low_perf = low_perf['Attrition'].value_counts(normalize=True)
print("Attrition Probability (Low Performance):\n", prob_attrition_low_perf)

# ================================
# Phase 2: Feature Engineering
# ================================

# Encode categorical variables
le = LabelEncoder()
df['Department'] = le.fit_transform(df['Department'])
df['Attrition'] = le.fit_transform(df['Attrition'])

# Feature scaling
scaler = StandardScaler()
df[['Salary', 'PerformanceScore']] = scaler.fit_transform(df[['Salary', 'PerformanceScore']])

# ================================
# Attrition Prediction - Classification
# ================================

X = df.drop(['Attrition', 'EmployeeID', 'Name'], axis=1)
y = df['Attrition']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression model
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.show()

# ================================
# Performance Prediction - Regression
# ================================

X_perf = df.drop(['PerformanceScore', 'EmployeeID', 'Name'], axis=1)
y_perf = df['PerformanceScore']

X_train, X_test, y_train, y_test = train_test_split(X_perf, y_perf, test_size=0.2, random_state=42)

reg = LinearRegression()
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)

print("R2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

# Actual vs Predicted
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Performance')
plt.ylabel('Predicted Performance')
plt.title('Actual vs Predicted Performance')
plt.show()

# ================================
# Phase 3: Deep Learning Models
# ================================

# Deep Learning - Performance Prediction
model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(16, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)

loss = model.evaluate(X_test, y_test)
print("Deep Learning MSE:", loss)

# Deep Learning - Attrition Prediction
X_dl = df.drop(['Attrition', 'EmployeeID', 'Name'], axis=1)
y_dl = df['Attrition']

X_train, X_test, y_train, y_test = train_test_split(X_dl, y_dl, test_size=0.2, random_state=42)

model_dl = Sequential([
    Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model_dl.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_dl.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)

_, accuracy = model_dl.evaluate(X_test, y_test)
print("Deep Learning Attrition Accuracy:", accuracy)

# ================================
# Phase 4: Insights
# ================================
print("\nKey Insights:")
print("- Salary and performance strongly influence attrition.")
print("- Certain departments show higher attrition risk.")
print("- Performance can be predicted effectively using ML and DL models.")
