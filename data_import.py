import pandas as pd
from pathlib import Path
import json
from zipfile import ZipFile

"""
The exported annotations from CVAT are put together to one single dataset. 

Input: 
- json files with all the labels per frame generated in CVAT

Output:
- dataset with all the labels together: 
frame | label | boulder | camera | participant | repetition
- binary dataset

IMPORTANT: the naming of the label files and the mp4 videos must be the same!
The label data and the pose detection data from the mp4 videos will be merged  later.
"""


# Define label mapping
label_map = {
    0: 'swing_phase', 1: 'dyno', 2: 'toe_hook', 3: 'heel_hook', 4: 'start_position',
    12: 'shoulder_pull', 14: 'egyptian', 15: 'skate', 17: 'ninja_kick', 20: 'top',
    21: 'before_start_position', 22: 'hand_matching', 23: 'foot_swap', 25: 'falling',
    26: 'no_movement_of_interest', 27: 'shoulder_press', 28: 'foot_matching', 9: 'flagging'
}

# File path to folder with all videos and labels
allLabelsDir = Path("data/all_annotations")

# Iterate through each zip file in the directory
for zipFile in allLabelsDir.glob("*.zip"):
    # Set the extraction path to be a folder with the same name as the zip file, minus '.zip'
    extraction_path = zipFile.with_suffix('')

    # Extract the zip file if the target directory does not already exist
    if not extraction_path.exists():
        with ZipFile(zipFile, 'r') as zip_ref:
            zip_ref.extractall(extraction_path)

        # Remove the zip file after successful extraction
        zipFile.unlink()
        print(f"Extracted and removed {zipFile.name}")
    else:
        print(f"Skipping {zipFile.name}, already extracted.")

# Load all JSON files
# Naming: W4_Cam22_Helene_Janicot_V2
# Naming: bouldernumber_camera_firstname_lastname_repetition
allLabelsDir = Path("data/all_annotations")
allLabelsFiles = [path for path in allLabelsDir.rglob("*.json")]
allLabelsPD = []

# Process each JSON file for metadata
for labelPath in allLabelsFiles:
    with open(labelPath) as file:
        labelsDict = json.load(file)
    labels = []
    string_list = str(labelPath.parent.parent.name).split("_")
    boulder = string_list[0]
    camera = string_list[1]
    participant = string_list[2] + " " + string_list[3]
    repetition = string_list[4]
    print(boulder, camera, participant, repetition, labelPath)

    for item in labelsDict["items"]:
        if item["annotations"]:  # Check if annotations list is not empty
            frame_id = item["attr"]["frame"]
            label_id = item["annotations"][0]["label_id"]
            label_name = label_map.get(label_id, "Unknown Label")  # Use 'get' to handle missing label_ids gracefully
            labels.append({
                "frame": frame_id,
                "label": label_name
            })
        else:
            print(f"No annotations found for frame {item['attr']['frame']} in {labelPath}")

    labelsPD = pd.DataFrame(labels)
    labelsPD["boulder"] = boulder
    labelsPD["camera"] = camera
    labelsPD["participant"] = participant
    labelsPD["repetition"] = repetition
    allLabelsPD.append(labelsPD)

# Concatenate all label data
allLabelsPD = pd.concat(allLabelsPD, ignore_index=True)
allLabelsPD.to_csv(allLabelsDir / "allLabels.csv", index=False)

# Initialize binary DataFrame with zeros for each label type
binary_df = pd.DataFrame(0, index=allLabelsPD.index, columns=label_map.values())

# Add metadata columns to the binary DataFrame
binary_df['boulder'] = allLabelsPD['boulder']
binary_df['camera'] = allLabelsPD['camera']
binary_df['participant'] = allLabelsPD['participant']
binary_df['repetition'] = allLabelsPD['repetition']

# Set 1 in binary_df for each row and label column where the label occurs
for index, row in allLabelsPD.iterrows():
    if row['label'] in binary_df.columns:
        binary_df.at[index, row['label']] = 1

# Print and save the binary DataFrame
print(binary_df.head())
binary_df.to_csv(allLabelsDir / "binaryLabels.csv", index=False)
