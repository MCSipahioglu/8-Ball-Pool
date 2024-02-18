# Libraries
import cv2
import numpy as np
import os
import shutil
import re

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
frame_files.sort(key=lambda x: int(re.search(r'\d+', x).group()))   # Frame files should be in 1,2,3,4 order not 1, 10, 100, 11 order!
console_timer=len(frame_files)
console_time=1


# Process the first frame file separately--------------------------------------------
print(f"{console_time}/{console_timer}")
console_time+=1

if frame_files:
    first_frame_file = frame_files[0]

    # Construct the full path to the frame file
    first_frame_path = os.path.join(frames_folder, first_frame_file)

    # Read the frame
    first_frame = cv2.imread(first_frame_path)

before_frame=first_frame
before_img_table=f.get_table(before_frame,img_width,img_height)     # Get the table only, rectified
before_img_reference=f.remove_tablecloth(before_img_table)          # Remove the tablecloth, only the balls and other objects should be left.
before_contours=f.detect_contours(before_img_reference)             # Detect the contours.
before_balls=f.detect_balls(before_img_reference,before_contours)   # Detect balls with class and state.

# Save the processed image.
for ball in before_balls:
    ball_class,ball_state,ball_center,ball_contour,ball_image,ball_luminance,ball_color_ratio=ball
    cv2.drawContours(before_img_reference, [ball_contour], 0, (0, 255, 0), int(1.5*gui_scale))
    cv2.putText(before_img_reference, ball_class, (ball_center[0], ball_center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5*gui_scale, (0, 0, 255), int(1.5*gui_scale))

output_path = os.path.join(output_folder, first_frame_file)
cv2.imwrite(output_path, cv2.resize(before_img_reference, (int(img_width / gui_scale), int(img_height / gui_scale))))

# ---------------------------------------------------------------------------------



for frame_file in frame_files[1:]:
    print(f"{console_time}/{console_timer}")
    console_time+=1

    # Construct the full path to the frame file
    after_frame_path = os.path.join(frames_folder, frame_file)
    after_frame = cv2.imread(after_frame_path)


    after_img_table=f.get_table(after_frame,img_width,img_height)       # Get the table only, rectified
    after_img_reference=f.remove_tablecloth(after_img_table)            # Remove the tablecloth, only the balls and other objects should be left.


    after_img_subtracted=after_img_reference
    after_balls=[]

    for before_ball in before_balls:

        before_ball_state=before_ball[1]
        before_ball_center=before_ball[2]
        before_ball_contour=before_ball[3]

        movement_state=f.did_ball_move(before_ball,after_img_subtracted)

        if movement_state==0:      #Ball didn't move. If so reextract the image in this contour directly.
            # If the ball didnt move, the position based variables will not change.
            after_ball_state=before_ball_state
            after_ball_center=before_ball_center
            after_ball_contour=before_ball_contour

            # The image however might have changed slightly due to the lighting or camera. Recalculate the image variables inside the same contour.
            after_ball_luminance=f.calculate_luminance(after_ball_contour, after_img_subtracted)
            after_ball_class=f.classify_ball_luminance(after_ball_luminance)
            after_ball_image=f.image_inside_contour(after_ball_contour,after_img_subtracted)

            # The stationary ball is added to the balls that are in the after_frame.
            after_ball= [after_ball_class, after_ball_state, after_ball_center, after_ball_contour, after_ball_image, after_ball_luminance]
            after_balls.append(after_ball)

            # Remove the stationary ball from the after_frame. We want to only redetect the balls that have moved.
            after_img_subtracted= cv2.subtract(after_img_subtracted, after_ball_image)

        elif movement_state==2:         #Ball didn't move but ball is obstructed. (Behind a human or something.)
            after_ball=before_ball      #Carry on the detection from the frame before. (Know that there is a ball behind a person)
            after_balls.append(after_ball)
            # We do not remove the stationary ball from the after_frame since the ball is obstructed in the frame.




            


    
    # Detect the remaining balls with class and state.
    after_contours=f.detect_contours(after_img_subtracted)
    after_moved_balls=f.detect_balls(after_img_subtracted,after_contours)
    if len(after_moved_balls)!=0:                                           # If any balls are still detected.
        after_balls=f.resolve_add_new_balls(after_balls,after_moved_balls)  # Add the redetected (moved) balls
        after_balls=f.resolve_ball_impossibilities(after_balls)



    # Save the processed image.
    for ball in after_balls:
        ball_class,ball_state,ball_center,ball_contour,ball_image,ball_luminance,ball_color_ratio=ball
        cv2.drawContours(after_img_reference, [ball_contour], 0, (0, 255, 0), int(1.5*gui_scale))
        cv2.putText(after_img_reference, ball_class, (ball_center[0], ball_center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5*gui_scale, (0, 0, 255), int(1.5*gui_scale))

    output_path = os.path.join(output_folder, frame_file)
    cv2.imwrite(output_path, cv2.resize(after_img_reference, (int(img_width / gui_scale), int(img_height / gui_scale))))



    # Change after_balls to before_balls for the next loop
    before_balls=[]
    before_balls=after_balls









cv2.waitKey(0)
cv2.destroyAllWindows()












































