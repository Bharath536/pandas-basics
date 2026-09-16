import pandas as pd

# Load CSV dataset
df = pd.read_csv("students.csv")

# First 5 rows
print("HEAD:")
print(df.head())

# Last 5 rows
print("\nTAIL:")
print(df.tail())

# Shape of dataset
print("\nSHAPE:")
print(df.shape)

# Column names
print("\nCOLUMNS:")
print(df.columns)

# Data types
print("\nDATA TYPES:")
print(df.dtypes)

# Count values in city column
print("\nCITY VALUE COUNTS:")
print(df["city"].value_counts())

# Filter students with marks above 80
print("\nSTUDENTS WITH MARKS ABOVE 80:")
print(df[df["marks"] > 80])