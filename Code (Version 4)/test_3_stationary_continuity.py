# Libraries
import cv2
import numpy as np
import os
import shutil

# External code
import f
from variables import img_width, img_height, gui_scale
import debug

##########################################################################################
dbg=1
##########################################################################################


# Read the frames
before_frame = cv2.imread('./test_frames/A1.jpg')
after_frame = cv2.imread('./test_frames/A2.jpg')

# Read the frames
#before_frame = cv2.imread('./test_frames/B1.jpg')
#after_frame = cv2.imread('./test_frames/B2.jpg')


before_img_table=f.get_table(before_frame,img_width,img_height)     # Get the table only, rectified
before_img_reference=f.remove_tablecloth(before_img_table)          # Remove the tablecloth, only the balls and other objects should be left.
before_contours=f.detect_contours(before_img_reference)             # Detect the contours.
before_balls=f.detect_balls_better(before_img_reference,before_contours)   # Detect balls with class and state.
if dbg==1: debug.gui_image_with_ball_classes('0.Before: Detected Balls',before_img_reference,gui_scale,before_balls)

after_img_table=f.get_table(after_frame,img_width,img_height)       # Get the table only, rectified
if dbg==1: debug.gui_image('1.After: Just the Table',after_img_table,gui_scale)
after_img_reference=f.remove_tablecloth(after_img_table)            # Remove the tablecloth, only the balls and other objects should be left.
if dbg==1: debug.gui_image('2.After: Not Subtracted',after_img_reference,gui_scale)

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
        after_ball_image=f.image_inside_contour(after_ball_contour,after_img_subtracted)
        after_ball_luminance=f.calculate_luminance(after_ball_contour, after_img_subtracted)
        after_ball_class=f.classify_ball_luminance(after_ball_luminance)

        if (after_ball_class == "||" or after_ball_class == "O") and (after_ball_state!=27 and after_ball_state!=28 and after_ball_state!=15 and after_ball_state!=16):             # Cue and 8 Balls are classified really well using just luminance. Also sectors 27 and 28 are better classified with luminance
            after_ball_color_ratio = f.calculate_color_ratio(after_ball_image)
            after_ball_class =f.classify_ball_color_ratio(after_ball_color_ratio)
        else:
            after_ball_color_ratio=f.calculate_color_ratio(after_ball_image)


        # The stationary ball is added to the balls that are in the after_frame.
        after_ball= [after_ball_class, after_ball_state, after_ball_center, after_ball_contour, after_ball_image, after_ball_luminance, after_ball_color_ratio]
        after_balls.append(after_ball)

        # Remove the stationary ball from the after_frame. We want to only redetect the balls that have moved.
        after_img_subtracted= cv2.subtract(after_img_subtracted, after_ball_image)

    elif movement_state==2:         #Ball didn't move but ball is obstructed. (Behind a human or something.)
        after_ball=before_ball      #Carry on the detection from the frame before. (Know that there is a ball behind a person)
        after_balls.append(after_ball)
        # We do not remove the stationary ball from the after_frame since the ball is obstructed in the frame.



if dbg==1: debug.gui_image('3.After: Subtracted',after_img_subtracted,gui_scale)


after_contours=f.detect_contours(after_img_subtracted)
if dbg==1: debug.gui_image_with_contours('4.After: Detected Contours',after_img_subtracted,gui_scale,after_contours)


# Detect the remaining balls with class and state.
after_moved_balls=f.detect_balls_better(after_img_subtracted,after_contours)
if len(after_moved_balls)!=0:       # If any balls are now detected.
    if dbg==1: debug.gui_image_with_ball_classes('5.After: Newly Detected Balls',after_img_subtracted,gui_scale,after_moved_balls)

    # Add the redetected (moved) balls
    after_balls=f.resolve_add_new_balls(after_balls,after_moved_balls)  # Add the redetected (moved) balls
    after_balls=f.resolve_ball_impossibilities(after_balls)

if dbg==1: debug.gui_image_with_ball_classes('6.After: All Balls',after_img_reference,gui_scale,after_balls)
if dbg==1: debug.gui_image_with_ball_states('7.After: Ball States',after_img_reference,gui_scale,after_balls) 




















cv2.waitKey(0)
cv2.destroyAllWindows()


