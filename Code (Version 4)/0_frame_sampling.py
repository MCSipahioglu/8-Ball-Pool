import cv2
import os
import shutil

from variables import video_path, sampling_period, frames_folder


output_folder = frames_folder

# Clear the existing contents of the output folder
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
os.makedirs(output_folder)

# Open the video file
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Calculate the frame index step based on sapling rate
frame_index_step = int(fps * sampling_period)

frame_count = 0

while True:
    # Read the next frame
    ret, frame = cap.read()

    if not ret:
        break  # Break the loop if the video ends

    frame_count += 1

    # Save frame if it's time to sample based on sapling rate
    if frame_count % frame_index_step == 0:
        frame_filename = os.path.join(output_folder, f"{frame_count // frame_index_step}.jpg")
        cv2.imwrite(frame_filename, frame)

# Release the video capture object
cap.release()