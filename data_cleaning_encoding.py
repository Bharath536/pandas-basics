import pandas as pd
from sklearn.preprocessing import LabelEncoder


# =========================================================
# PART 1: DATA CLEANING
# =========================================================

print("========== PART 1: DATA CLEANING ==========\n")

data = {
    "Name": ["Bharath", "Rahul", "Priya", "Anu", "Kiran"],
    "Age": [22, None, 24, 23, None],
    "Salary": [30000, 35000, None, 40000, 32000],
    "City": ["Hyderabad", "Mumbai", "Chennai", None, "Hyderabad"]
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)

print("\nMissing Values:")
print(df.isnull().sum())


# =========================================================
# DROPNA()
# =========================================================

df_dropna = df.dropna()

print("\nDataset after dropna():")
print(df_dropna)


# =========================================================
# FILLNA()
# =========================================================

df_fillna = df.copy()

df_fillna["Age"] = df_fillna["Age"].fillna(
    df_fillna["Age"].mean()
)

df_fillna["Salary"] = df_fillna["Salary"].fillna(
    df_fillna["Salary"].mean()
)

df_fillna["City"] = df_fillna["City"].fillna("Unknown")

print("\nDataset after fillna():")
print(df_fillna)


# =========================================================
# PART 2: CATEGORICAL ENCODING
# =========================================================

print("\n========== PART 2: CATEGORICAL ENCODING ==========\n")

data2 = {
    "Name": ["Bharath", "Rahul", "Priya", "Anu", "Kiran"],
    "Education": [
        "High School",
        "Bachelors",
        "Masters",
        "PhD",
        "Bachelors"
    ],
    "City": [
        "Hyderabad",
        "Mumbai",
        "Chennai",
        "Hyderabad",
        "Mumbai"
    ],
    "Gender": [
        "Male",
        "Male",
        "Female",
        "Female",
        "Male"
    ]
}

df2 = pd.DataFrame(data2)

print("Original Categorical Dataset:")
print(df2)


# =========================================================
# ORDINAL ENCODING
# =========================================================

education_order = {
    "High School": 0,
    "Bachelors": 1,
    "Masters": 2,
    "PhD": 3
}

df2["Education"] = df2["Education"].map(education_order)

print("\nAfter Label Encoding Education:")
print(df2)


# =========================================================
# ONE-HOT ENCODING
# =========================================================

df_encoded = pd.get_dummies(
    df2,
    columns=["City", "Gender"],
    drop_first=True,
    dtype=int
)

print("\nAfter One-Hot Encoding:")
print(df_encoded)


# =========================================================
# REMOVE NAME COLUMN
# =========================================================

df_encoded = df_encoded.drop(columns=["Name"])


# =========================================================
# VERIFY NO STRING COLUMNS REMAIN
# =========================================================

print("\nData Types:")
print(df_encoded.dtypes)

string_columns = df_encoded.select_dtypes(include=["str"]).columns

print("\nRemaining String Columns:")
print(list(string_columns))

if len(string_columns) == 0:
    print("\nSUCCESS: No string columns remain.")
else:
    print("\nWARNING: String columns still exist.")


# =========================================================
# SAVE FINAL DATASET
# =========================================================

df_encoded.to_csv(
    "cleaned_encoded_dataset.csv",
    index=False
)

print("\nFinal dataset saved as cleaned_encoded_dataset.csv")