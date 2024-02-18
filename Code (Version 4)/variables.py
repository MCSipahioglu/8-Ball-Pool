import numpy as np

#################################    FILE VARIABLES    ##################################
video_path = "video.mp4"    # Replace with your video file name
frames_folder = "frames"    # Replace with the existing output folder name
processed_frames_folder = "frames_processed"
frame_rate=30               # For output video

##############################    PERFORMANCE VARIABLES    ###############################
# We want the final plot to be 360x720 so it can be viewed on the screen.
gui_width=360
img_width=1440              # Extracted Rectified Table Image Size. (Higher for better performance)
sampling_period = 0.1       # Define the sampling rate in seconds. Every sampling_period seconds a frame is sampled.  (Lower for better performance)
img_height=2*img_width      # Standard aspect ratio for billiards tables are 1:2
gui_scale=img_width/gui_width


##################################    GUI VARIABLES    ##################################
# Master Balls Colors BGR
cue_color = (255, 0, 255)  # Purple
eight_color = (0, 0, 255)  # Red
solid_colors = [(0, 255, 0) for i in range(7)]   # Green
striped_colors = [(255, 0, 0) for i in range(7)]    # Blue
#solid_colors = [(0, 128 + i*32, 0) for i in range(7)]   # Shades of green
#striped_colors = [(128 + i*32, 0, 0) for i in range(7)]    # Shades of blue
master_color_array = [cue_color] + solid_colors + [eight_color] + striped_colors



########################    MANUALLY INSERTED IMAGE VARIABLES    ########################

# For rectifying image
corner_bottom_left=np.float32([291,595])
corner_bottom_right=np.float32([963,595])
corner_top_left=np.float32([437,67])
corner_top_right=np.float32([834,67])
corners_original  = np.float32([corner_bottom_left,corner_bottom_right,corner_top_left,corner_top_right])

# For masking image
color_mask_bounds=np.array([[86, 102, 60],[100, 255, 255]]) # Lower and upper bounds for HSV values to be removed
luminance_mask_bound=40


# For eliminating contours
ball_radius_limits=(10,26)          # If width is 360, ball radius should be in this range.
ball_aspect_ratio_limits=(1/3,1.5)  #1/3 is 1 width 3 height. (Possible near the top of the image due to distortion)
ball_area_limits=(125,937.5)

# For classifying balls
ball_luminance_limits=[64,95,140,189]                              #in luminance of LAB. Divide between classes: Hole-8Ball-Solid-Striped-Cue

color_ratio_mask_limit_white_minimum_lightness=220
color_ratio_mask_limit_color_minimum_saturation=40

color_ratio_mask_limits_white=np.array([[0, color_ratio_mask_limit_white_minimum_lightness, 0],[179, 255, 255]])  #in HLS for calculating the color ratio (Differentiating whiteish or colored pixels.)
color_ratio_mask_limits_color=np.array([[0, 0, color_ratio_mask_limit_color_minimum_saturation],[179, color_ratio_mask_limit_white_minimum_lightness-1, 255]])
color_ratio_mask_limits = np.concatenate((color_ratio_mask_limits_white, color_ratio_mask_limits_color))

ball_color_limits=[0.6,14.7,130]                                        #in color ratio.  Divide between classes: Cue-Striped-Solid-8Ball

# For finding ball state
hole_centers = [
    (0, 0),
    (img_width, 0),
    (0, img_height // 2),
    (img_width, img_height // 2),
    (0, img_height),
    (img_width, img_height)
]

solo_radius = 20        # Radius for the circular regions around the holes


# For checking the difference between two frames:
pixel_difference_thresholds=(10,125)

# For continuity correction.
matching_radius=100     #Upper limit for matching to the closest ball.

#Recalculate the radii for the gui_scale. (Original variables are tuned for an image of width 360)
ball_radius_limits=tuple(element * gui_scale for element in ball_radius_limits)
ball_area_limits=tuple(element * gui_scale * gui_scale for element in ball_area_limits)
solo_radius=int(gui_scale*solo_radius)
pixel_difference_thresholds=tuple(element * gui_scale * gui_scale for element in pixel_difference_thresholds)
matching_radius=matching_radius*gui_scale






########################    DEBUGGING VARIABLES    ########################
v4_dbg_flag=0               # Turns on or off update notifications in update_master_balls. (Version 4 Only)