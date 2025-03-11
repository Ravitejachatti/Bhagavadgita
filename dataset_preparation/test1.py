import pandas as pd

# Load the dataset and manually set column names
df = pd.read_csv("../dataset/dataset1.0.csv", header=None, names=["Speech", "Shloka", "Speaker"])

# Drop the 'Speaker' column if it's not needed
df = df[["Speech", "Shloka"]]  

# Print to verify
print("Column names:", df.columns)
print(df.head())

# Save the corrected dataset
df.to_csv("corrected_dataset.csv", index=False)