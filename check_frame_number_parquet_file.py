import pandas as pd

# Load the coordinate file into a DataFrame
coordinates_df = pd.read_parquet('/output/video_coordinates/W1_Cam24_Ai_Mori_V2_coordinates_global.parquet')

# Get the number of frames
num_frames = len(coordinates_df)

print("Number of frames:", num_frames)