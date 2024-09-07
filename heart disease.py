import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the dataset
df = pd.read_csv('heart.csv')

# Create the output directory if it doesn't exist
output_dir = 'output_images'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 1. Data Cleaning

# Check for missing values
missing_values = df.isnull().sum()
print("Missing values in each column:\n", missing_values)

# Handle missing values if any (for demonstration, assuming no missing values)

# Check for outliers using IQR method (Interquartile Range)
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

# Filtering out the outliers
df_outliers_removed = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

print(f"Original dataset size: {df.shape}")
print(f"Dataset size after removing outliers: {df_outliers_removed.shape}")

# 2. Exploratory Data Analysis (EDA)

# Distribution of the target variable
plt.figure()
sns.countplot(x='target', data=df)
plt.title('Distribution of Target Variable (Heart Disease)')
plt.savefig(os.path.join(output_dir, 'target_distribution.png'))
plt.close()

# Correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
plt.close()

# Pairplot for the dataset
sns.pairplot(df, hue='target')
plt.savefig(os.path.join(output_dir, 'pairplot.png'))
plt.close()

# 3. Question Formulation and Analysis

# Question 1: What is the distribution of age in the dataset?
plt.figure()
sns.histplot(df['age'], kde=True)
plt.title('Age Distribution')
plt.savefig(os.path.join(output_dir, 'age_distribution.png'))
plt.close()

# Question 2: Is there a relationship between age and maximum heart rate?
plt.figure()
sns.scatterplot(x='age', y='thalach', data=df)
plt.title('Age vs. Maximum Heart Rate')
plt.savefig(os.path.join(output_dir, 'age_vs_max_heart_rate.png'))
plt.close()

# Question 3: Does chest pain type correlate with the likelihood of heart disease?
plt.figure()
sns.countplot(x='cp', hue='target', data=df)
plt.title('Chest Pain Type vs. Heart Disease')
plt.savefig(os.path.join(output_dir, 'cp_vs_heart_disease.png'))
plt.close()

# Question 4: How does cholesterol level vary with age?
plt.figure()
sns.scatterplot(x='age', y='chol', data=df)
plt.title('Age vs. Cholesterol Levels')
plt.savefig(os.path.join(output_dir, 'age_vs_cholesterol.png'))
plt.close()

# Question 5: What is the distribution of resting blood pressure?
plt.figure()
sns.histplot(df['trestbps'], kde=True)
plt.title('Resting Blood Pressure Distribution')
plt.savefig(os.path.join(output_dir, 'resting_blood_pressure_distribution.png'))
plt.close()

# Question 6: Is fasting blood sugar a significant indicator of heart disease?
plt.figure()
sns.countplot(x='fbs', hue='target', data=df)
plt.title('Fasting Blood Sugar vs. Heart Disease')
plt.savefig(os.path.join(output_dir, 'fbs_vs_heart_disease.png'))
plt.close()

# Question 7: How does exercise-induced angina relate to heart disease?
plt.figure()
sns.countplot(x='exang', hue='target', data=df)
plt.title('Exercise-Induced Angina vs. Heart Disease')
plt.savefig(os.path.join(output_dir, 'exang_vs_heart_disease.png'))
plt.close()

# Question 8: What is the distribution of ST depression induced by exercise?
plt.figure()
sns.histplot(df['oldpeak'], kde=True)
plt.title('ST Depression (Oldpeak) Distribution')
plt.savefig(os.path.join(output_dir, 'oldpeak_distribution.png'))
plt.close()

# Question 9: How does the number of major vessels relate to heart disease?
plt.figure()
sns.countplot(x='ca', hue='target', data=df)
plt.title('Number of Major Vessels (ca) vs. Heart Disease')
plt.savefig(os.path.join(output_dir, 'ca_vs_heart_disease.png'))
plt.close()

# Question 10: How does thalassemia type relate to heart disease?
plt.figure()
sns.countplot(x='thal', hue='target', data=df)
plt.title('Thalassemia Type vs. Heart Disease')
plt.savefig(os.path.join(output_dir, 'thal_vs_heart_disease.png'))
plt.close()

# 4. Data Visualization

# Plotting the distribution of key features for heart disease presence/absence
features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

for feature in features:
    plt.figure(figsize=(10, 5))
    sns.histplot(df[feature], kde=True, hue=df['target'])
    plt.title(f'Distribution of {feature} by Heart Disease Status')
    plt.savefig(os.path.join(output_dir, f'{feature}_distribution_by_heart_disease.png'))
    plt.close()

# Visualizing the effect of slope on heart disease
plt.figure()
sns.countplot(x='slope', hue='target', data=df)
plt.title('Slope of Peak Exercise ST Segment vs. Heart Disease')
plt.savefig(os.path.join(output_dir, 'slope_vs_heart_disease.png'))
plt.close()
