import pandas as pd
from pathlib import Path
import time

"""
The global coordinates and the labels of each frame are put together into one single dataframe. 

Input: 
- global coordinates as parquet files
- all labels with metadata

Output: 
- csv file with all the data merged together: 
frame | label | boulder | camera | participant | repetition | time(s) | all coordinates
"""


# Function to print the elapsed time
def print_elapsed_time(start_time, message):
    elapsed_time = time.time() - start_time
    print(f"{message} - Elapsed time: {elapsed_time:.2f} seconds")


# Load the allLabels.csv file into a DataFrame
start_time = time.time()
all_labels_df = pd.read_csv('data/all_annotations/allLabels.csv')
print_elapsed_time(start_time, "Loaded allLabels.csv")

# Directory containing the coordinate parquet files
coordinates_dir = Path('output/video_coordinates')

# Initialize an empty list to store DataFrames
dfs = []

# Iterate over each row in the all_labels DataFrame
for index, row in all_labels_df.iterrows():
    boulder = row['boulder']
    camera = row['camera']
    participant = row['participant']
    repetition = row['repetition']
    frame_number = row['frame']

    # Split the participant name into first and last name
    first_name, last_name = participant.split()

    # Construct the filename of the corresponding Parquet file
    filename = f'{boulder}_{camera}_{first_name}_{last_name}_{repetition}_coordinates_global.parquet'
    file_path = coordinates_dir / filename

    if not file_path.exists():
        print(f"Warning: File {filename} not found. Skipping.")
        continue

    # Load the Parquet file into a DataFrame
    start_time = time.time()
    coordinates_df = pd.read_parquet(file_path)
    print_elapsed_time(start_time, f"Loaded {filename}")

    # Add frame number to the coordinates DataFrame
    coordinates_df.insert(0, 'frame_number', range(len(coordinates_df)))

    # Filter the coordinates DataFrame to only include the relevant frame
    frame_coordinates_df = coordinates_df[coordinates_df['frame_number'] == frame_number]

    if frame_coordinates_df.empty:
        # prints out if the coordinates are missing for the frame number in the allLabels.csv
        print(f"Warning: Frame {frame_number} not found in {filename}. Skipping.")
        continue

    # Merge the all_labels row with the filtered coordinates DataFrame
    merged_df = pd.merge(pd.DataFrame([row]), frame_coordinates_df, how='inner', left_on='frame', right_on='frame_number')

    # Append the merged DataFrame to the list
    dfs.append(merged_df)

# Concatenate all DataFrames into a single DataFrame
start_time = time.time()
labels_and_coordinates_df = pd.concat(dfs, ignore_index=True)
print_elapsed_time(start_time, "Concatenated all DataFrames")

# Define the output path
output_path = Path('output')

# Save the final DataFrame to a CSV file
start_time = time.time()
output_file_path = output_path / 'labels_and_coordinates.csv'
labels_and_coordinates_df.to_csv(output_file_path, index=False)
print_elapsed_time(start_time, "Saved labels_and_coordinates.csv")
