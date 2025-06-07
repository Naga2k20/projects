import pandas as pd

# Load the dataset
df = pd.read_csv("enhanced_cleaned_dataset_medium.csv")

# Display basic statistics
print("Dataset Info:")
print(df.info())

print("Summary Statistics:")
print(df.describe())

# Check for missing values
print("Missing Values:")
print(df.isnull().sum())
