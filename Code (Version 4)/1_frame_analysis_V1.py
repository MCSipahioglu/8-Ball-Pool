# Libraries
import cv2
import numpy as np
import os
import shutil

# External code
import f
from variables import frames_folder, processed_frames_folder, img_width, img_height, gui_scale


##########################################################################################


output_folder=processed_frames_folder

# Clear the existing contents of the output folder
if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
os.makedirs(output_folder)


# Get the list of all frame files in the folder
frame_files = [f for f in os.listdir(frames_folder) if f.endswith('.jpg')]
console_timer=len(frame_files)
console_time=1

for frame_file in frame_files:
    print(f"{console_time}/{console_timer}")
    console_time+=1

    # Construct the full path to the frame file
    frame_path = os.path.join(frames_folder, frame_file)

    # Read the frame
    frame = cv2.imread(frame_path)

    # Get the table only, rectified.
    img_table=f.get_table(frame,img_width,img_height)


    # Remove the tablecloth, only the balls and other objects should be left.
    img_reference=f.remove_tablecloth(img_table)


    # Detect the contours.
    contours=f.detect_contours(img_reference)


    # Detect balls with class and state.
    balls=f.detect_balls(img_reference,contours)

    # Draw each classified frame
    for ball in balls:
        ball_class,ball_state,ball_center,ball_contour,ball_luminance,ball_color_ratio=ball
    
        cv2.drawContours(img_reference, [ball_contour], 0, (0, 255, 0), int(1.5*gui_scale))
        cv2.putText(img_reference, ball_class, (ball_center[0], ball_center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5*gui_scale, (0, 0, 255), int(1.5*gui_scale))

    # Save the frame (replace 'saved_frame.jpg' with your desired filename)
    output_path = os.path.join(output_folder, frame_file)
    cv2.imwrite(output_path, cv2.resize(img_reference, (int(img_width / gui_scale), int(img_height / gui_scale))))















cv2.waitKey(0)
cv2.destroyAllWindows()











