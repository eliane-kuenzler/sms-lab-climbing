import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
df = pd.read_csv('../data/dataframes/labels_and_coordinates.csv')

# Display basic information about the DataFrame
print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")
print(f"Column names: {df.columns.tolist()}")
print(df.describe())

# Checking the first few rows of the DataFrame
print(df.head())

# Plot the count of different labels
plt.figure(figsize=(10, 6))
df['label'].value_counts().plot(kind='bar', edgecolor='k', alpha=0.7)
plt.title('Count of Different Labels')
plt.xlabel('Label')
plt.ylabel('Count')
plt.show()