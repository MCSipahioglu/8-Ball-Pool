# Libraries
import cv2
import numpy as np
import os
import shutil
import re
import time

# External code
import f
import debug
from variables import frames_folder, processed_frames_folder, img_width, img_height, gui_scale

start_time = time.time()
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


# Initialize the Master Balls
initial_ball_state=-1       # Not Yet Detected

master_balls = [[-99] * 7 for _ in range(16)]     # Master balls will hold all ball data by tracking them. At 0 it will hold the cue ball data. From index 1-7 solid, index 8-> 8 ball, index 9-15 Striped Balls.
for i in range(len(master_balls)):        # At indexes 1-7 and 9-15 it will hold the solid or striped balls' data in no particular order since we don't classify the balls by specific color or number.
    
    master_balls[i][1]=initial_ball_state
    
    if i==0:                # Cue Ball
        master_balls[i][0]='C'
    elif 1 <= i <= 7:       # Solid Balls
        master_balls[i][0]='O'
    elif i==8:              # 8 Ball
        master_balls[i][0]='8'
    elif 9 <= i <= 15:      # Striped Balls
        master_balls[i][0]='||'



start_frame=94
#end_frame=148
console_time=start_frame

for frame_file in frame_files[start_frame-1:]:
    print(f"{console_time}/{console_timer}")
    console_time+=1

    # Construct the full path to the frame file
    frame_path = os.path.join(frames_folder, frame_file)
    frame = cv2.imread(frame_path)

    master_balls=f.update_master_balls(master_balls,frame)

    output_path = os.path.join(output_folder, frame_file)
    debug.gui_master_balls(0,frame,gui_scale,master_balls,1,output_path)








##########################################################################################
end_time = time.time()
execution_time_seconds = end_time - start_time

hours, remainder = divmod(execution_time_seconds, 3600)
minutes, seconds = divmod(remainder, 60)

print("Execution time: {} hours, {} minutes, {} seconds".format(int(hours), int(minutes), int(seconds)))






