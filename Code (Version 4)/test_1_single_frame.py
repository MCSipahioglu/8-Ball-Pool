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



# Read the image
frame = cv2.imread('./test_frames/test_frame.jpg')

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
balls=f.detect_balls(img_reference,contours)
if dbg==1: debug.gui_image_with_ball_classes('4.Detected Balls',img_reference,gui_scale,balls)
if dbg==1: debug.gui_image_with_ball_states('5.Image with State Sectors',img_reference,gui_scale,balls)   # Also the states are found during classification as well.



cv2.waitKey(0)
cv2.destroyAllWindows()















