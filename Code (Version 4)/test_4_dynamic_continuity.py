# Libraries
import cv2
import numpy as np
import os
import shutil
import re

# External code
import f
from variables import img_width, img_height, gui_scale, frames_folder
import debug

##########################################################################################
dbg=1
##########################################################################################

# Read the frames
#before_frame = cv2.imread('./frames/105.jpg')
#after_frame = cv2.imread('./frames/106.jpg')


# Read the frames
#before_frame = cv2.imread('./frames/4251.jpg')
#after_frame = cv2.imread('./frames/4252.jpg')

frame_files = [f for f in os.listdir(frames_folder) if f.endswith('.jpg')]
frame_files.sort(key=lambda x: int(re.search(r'\d+', x).group()))   # Frame files should be in 1,2,3,4 order not 1, 10, 100, 11 order!


#102-105 good start for testing. Everything moving. Then try bigger intervals. (73 is first frame with any meaningful movement)
start_frame=93  
end_frame=147


#start_frame=1403
#end_frame=1418

#start_frame=2170
#end_frame=2188

initial_ball_state=-1       # Not Yet Detected

master_balls = [[-99] * 7 for _ in range(16)]     # Master balls will hold all ball data by tracking them. At 0 it will hold the cue ball data. From index 1-7 solid, index 8-> 8 ball, index 9-15 Striped Balls.
for i, ball in enumerate(master_balls):         # At indexes 1-7 and 9-15 it will hold the solid or striped balls' data in no particular order since we don't classify the balls by specific color or number.
    ball[1]=initial_ball_state
    if i==0:                # Cue Ball
        ball[0]='C'
    elif 1 <= i <= 7:       # Solid Balls
        ball[0]='O'
    elif i==8:              # 8 Ball
        ball[0]='8'
    elif 9 <= i <= 15:      # Striped Balls
        ball[0]='||'






for frame_file in frame_files[start_frame-1:end_frame]:


    # Construct the full path to the frame file
    frame_path = os.path.join(frames_folder, frame_file)
    frame = cv2.imread(frame_path)

    master_balls=f.update_master_balls(master_balls,frame)


    if dbg==1: print(f'\nFrame{frame_file}')

    #debug.gui_master_balls(0,frame,gui_scale,master_balls,1,output_path)
    debug.gui_master_balls(frame_file,frame,gui_scale,master_balls,0,0)








cv2.waitKey(0)
cv2.destroyAllWindows()


