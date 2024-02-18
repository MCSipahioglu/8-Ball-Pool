import cv2
import os
import re

from variables import img_width, img_height, sampling_period, processed_frames_folder, frame_rate



frame_files = [f for f in os.listdir(processed_frames_folder) if f.endswith('.jpg')]
frame_files.sort(key=lambda x: int(re.search(r'\d+', x).group()))   # Frame files should be in 1,2,3,4 order not 1, 10, 100, 11 order!
frame_size = (2*img_width, img_height)


# Initialize video writer
video_writer = cv2.VideoWriter('video_processed.mp4', cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, frame_size)

# Iterate through each frame and add it to the video
for frame_file in frame_files:
    frame_path = os.path.join(processed_frames_folder, frame_file)
    frame = cv2.imread(frame_path)
    if frame is not None:
        # Resize frame if necessary
        if frame.shape[:2] != frame_size:
            frame = cv2.resize(frame, frame_size)
        # Write the frame to the video
        for _ in range(int(frame_rate * sampling_period)):
            video_writer.write(frame)

# Release video writer
video_writer.release()

print("Video created successfully!")
