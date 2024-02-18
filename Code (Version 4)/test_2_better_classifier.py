# Libraries
import cv2
import numpy as np


# External code
import f
import debug
from variables import img_width, img_height, gui_scale



##########################################################################################
dbg=1 # 1 for debug mode on.
##########################################################################################


# Read the image (Case studies)
frame1 = cv2.imread('./test_frames/48.jpg')                              # Green striped ball in lowest sector classified as solid - Fixed (Tune color_ratio thresholds)
frame2 = cv2.imread('./test_frames/test_frame.jpg')                               # Yellow and orange solids classified as striped - Fixed (Created detect_balls_better method, two step classification)
frame3 = cv2.imread('./test_frames/A1.jpg')   # Ball in solo region (Very close to hole) - Fixed (mask out most of the holes in the beginning with luminance mask)
frame4 = cv2.imread('./test_frames/91.jpg')              # Solid red ball in lowest sector classified as striped - Fixed (Use luminance classification in sectors 27, 28)
frame5= cv2.imread('./test_frames/14.jpg')               # Hole on right top detected as ball (8 Ball) - Fixed (Luminance classification threshold tuned)
frame6= cv2.imread('./test_frames/461.jpg')              # 8 Ball vs Solid Ball Incorrect Classification but correct in the frames before and after - (If a stationary ball is classified differently this time check classification history to give most probable correct classification)
frame7= cv2.imread('./test_frames/476.jpg')              # When 8 ball is gone last solid ball is classified as 8 ball. Luminance actually lower. ^^^ Stationary contunity correction should solve it.
frame8 = cv2.imread('./frames/4252.jpg')
frame9 = cv2.imread('./frames/105.jpg')


frames=[frame1,frame2,frame3,frame4,frame5,frame6,frame7]

frame = frame9   # !!! Comment this line out if you want to analyze multiple frames as defined above. !!!

# If a single frame is defined analyze it only, if frame is commented out analyze multiple frames in "frames"
try:     
    # Get the table only, rectified.
    img_table=f.get_table(frame,img_width,img_height)
    if dbg==1: debug.gui_image('1.Rectified Image',img_table,gui_scale)

    # Remove the tablecloth, only the balls and other objects should be left.
    img_reference=f.remove_tablecloth(img_table)
    if dbg==1: debug.gui_image('2.Masked Image',img_reference,gui_scale)

    # Detect the contours.
    contours=f.detect_contours(img_reference)
    if dbg==1: debug.gui_image_with_contours('3.Detected Contours',img_reference,gui_scale,contours)


    # Detect balls with class and state.
    balls=f.detect_balls_better(img_reference,contours)
    if dbg==1: debug.gui_image_with_ball_classes('4.Detected Balls',img_reference,gui_scale,balls)
    if dbg==1: debug.gui_image_with_ball_states('5.Image with State Sectors',img_reference,gui_scale,balls)

except:
    i=1
    for frame in frames:
        print(f"FRAME{i}")
        
        img_table=f.get_table(frame,img_width,img_height)
        img_reference=f.remove_tablecloth(img_table)
        contours=f.detect_contours(img_reference)
        balls=f.detect_balls_better(img_reference,contours)
        if dbg==1: debug.gui_image_with_ball_classes(f'Frame{i} . Detected Balls',img_reference,gui_scale,balls)
        i+=1

        print("\n")







cv2.waitKey(0)
cv2.destroyAllWindows()















