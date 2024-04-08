import pandas as pd

# File path
csv_path = "C:/Users/Om/OneDrive/Desktop/heart_attack_prediction_dataset.csv"

# Read CSV file
csv_read = pd.read_csv(csv_path)

# Display first few rows
print("\n", csv_read.head())
print("--------------------------------------------------------------")

# Sort data by age
sort_data = csv_read.sort_values(by='Age', ascending=True)
print(sort_data.head())

# Accessing column 'Age'
column = sort_data['Age']
print("--------------------------------------------------------------")
print(column.head())
print("-----------------------------------------------------------------")

# Descriptive statistics
print("Describe \n", sort_data.describe())
print("-----------------------------------------------------------------")

# Column names
print("Attributes of table : ", sort_data.columns)

# Data types of each column
print("Data Type of each column:\n", sort_data.dtypes)
print("-----------------------------------------------------------------")

# Accessing unique values and counts
for col in sort_data.columns:
    print("Unique values for", col, "are:", sort_data[col].unique())
    print("Number of unique values for", col, "is:", sort_data[col].nunique())

# Convert 'Age' column to float
sort_data['Age'] = sort_data['Age'].astype(float)

# Identifying missing values
missing_values = csv_data.isnull().sum()

# Filling in the missing values
csv_data_filled = csv_data.fillna(10)

