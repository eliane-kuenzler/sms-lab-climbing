import cv2
from pathlib import Path
import numpy as np
from tqdm import tqdm
import pandas as pd

"""
generates a mp4 video file with the coordinates from mediapipe. 
Enables to check how accurate the mediapipe pose estimation algorithm is.
"""


input_video_path = Path('data/cvat_videos/W4_Cam22_Oriane_Bertone_V6.mp4')
input_landmarks_path = Path('output/video_coordinates/video_coordinates_local.parquet')
output_overlay_path = Path('output/video_coordinates/W4_Cam22_Oriane_Bertone_V6.mp4')

landmarks = pd.read_parquet(str(input_landmarks_path))

process_capture = cv2.VideoCapture(str(input_video_path))

ret, frame = process_capture.read()
fps = process_capture.get(cv2.CAP_PROP_FPS)
height, width, channels = frame.shape
out = cv2.VideoWriter(str(output_overlay_path),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (int(width), int(height)),
)
max_frame_number = min(int(process_capture.get(cv2.CAP_PROP_FRAME_COUNT)), len(landmarks))


for current_frame_number in tqdm(range(max_frame_number)):
    current_landmarks = landmarks.iloc[current_frame_number]

    landmarks_x = current_landmarks[current_landmarks.index.str.contains('_x')]*int(width)
    landmarks_y = current_landmarks[current_landmarks.index.str.contains('_y')]*int(height)

    for point in [*zip(landmarks_x, landmarks_y)]:
        x, y = point
        if not np.isnan(x) and not np.isnan(y):
            cv2.circle(frame, (int(x), int(y)), 5, (255, 0, 0), -1)  # -1 fills the circle

    out.write(frame)
    ret, frame = process_capture.read()

process_capture.release()
out.release()