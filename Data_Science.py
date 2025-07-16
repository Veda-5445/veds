import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Create a dummy dataset
data = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank', None],
    'Age': [25, np.nan, 35, 40, 29, 30, 22],
    'Salary': [50000, 60000, np.nan, 80000, 70000, 60000, 55000],
    'Department': ['HR', 'Finance', 'IT', 'HR', None, 'Finance', 'IT']
})

print("Raw Data Preview:")
print(data)

# Check for missing values
print("\nMissing Values (per column):")
null_counts = data.isnull().sum()
print(null_counts)

# Calculate total and average missing values
total_null = null_counts.sum()
average_null = null_counts.mean()
print("\nTotal Missing Values in Dataset:", total_null)
print("Average Missing Values per Column:", average_null)

# Handle missing values
imputer_num = SimpleImputer(strategy="mean")
data_numeric = data.select_dtypes(include=[np.number])
data[data_numeric.columns] = imputer_num.fit_transform(data_numeric)

imputer_cat = SimpleImputer(strategy="most_frequent")
data_categorical = data.select_dtypes(include=[object])
data[data_categorical.columns] = imputer_cat.fit_transform(data_categorical)

print("\nMissing values handled.")
print(data)

# Remove duplicate rows
before_duplicates = data.shape[0]
data = data.drop_duplicates()
after_duplicates = data.shape[0]
print("Removed duplicate rows:", before_duplicates - after_duplicates)

# Encode categorical variables
le = LabelEncoder()
for col in data.select_dtypes(include=[object]).columns:
    data[col] = le.fit_transform(data[col])

print("\nCategorical variables encoded.")
print(data)

# Scale numeric features
scaler = StandardScaler()
data[data_numeric.columns] = scaler.fit_transform(data[data_numeric.columns])

print("\nNumeric features scaled.")
print(data)

# Save cleaned data
data.to_csv("cleaned_dataset.csv", index=False)
print("Cleaned data saved as 'cleaned_dataset.csv'")
