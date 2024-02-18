import cv2
import numpy as np

import debug
from variables import img_width, img_height, gui_scale                                      # For printing the images on screen
from variables import color_mask_bounds, luminance_mask_bound, corners_original             # For preprocessing image before contour detection
from variables import ball_radius_limits, ball_aspect_ratio_limits, ball_area_limits        # For eliminating contours
from variables import ball_luminance_limits, color_ratio_mask_limits, ball_color_limits     # For classifying balls
from variables import solo_radius, hole_centers                                             # For classifying position (State)
from variables import pixel_difference_thresholds                                           # For detecting stationary balls (Stationary Continuity)
from variables import matching_radius                                                       # For matching moving balls (Dynamic Continuity)
from variables import v4_dbg_flag                                                           # For turning debugging on and off. (Affects test_4 and frame_analysis_V4)


def get_table(image,width,height):       # Get the frame, cut the table out, rectify and return it.
    corners_rectified = np.float32([ [0,0],[width,0],[0,height],[width,height] ])

    H_rect = cv2.getPerspectiveTransform(corners_original,corners_rectified) # getting perspective by 4 points of each image
    image_rectified = cv2.warpPerspective(image, H_rect, (width,height)) # warps perpective to new image
    image_rectified = cv2.rotate(image_rectified, cv2.ROTATE_180)
    return image_rectified



def remove_tablecloth(img_table):
 
    img_table_hsv = cv2.cvtColor(img_table, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for HSV values to be removed
    color_lower_bound = color_mask_bounds[0]
    color_upper_bound = color_mask_bounds[1]

    # Create mask for the specified HSV ranges
    mask = cv2.inRange(img_table_hsv, color_lower_bound, color_upper_bound)

    # Invert the mask (if you want to keep the regions outside the specified HSV range)
    inverted_mask = cv2.bitwise_not(mask)

    # Apply the mask to the original image
    img_masked = cv2.bitwise_and(img_table, img_table, mask=inverted_mask)


    # Convert the masked image to the LAB color space
    img_masked_lab = cv2.cvtColor(img_masked, cv2.COLOR_BGR2LAB)

    # Extract the L channel from the LAB image
    l_channel = img_masked_lab[:, :, 0]

    # Create a mask to identify pixels with luminance values below 50
    low_luminance_mask = cv2.inRange(l_channel, 0, luminance_mask_bound)

    # Invert the mask (if you want to keep the regions outside the specified HSV range)
    inverted_luminance_mask = cv2.bitwise_not(low_luminance_mask)

    # Apply the low luminance mask to the original image
    img_final = cv2.bitwise_and(img_masked, img_masked, mask=inverted_luminance_mask)

    return img_final



def detect_contours(image):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)

    # Find contours
    contours, _ = cv2.findContours(blurred, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)

    return contours



##########################################################################################



def center_of_contour(contour):
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return cx, cy
    else:
        return None



def image_inside_contour(contour, image):
    # Create a mask using the contour
    mask = np.zeros_like(image)
    cv2.drawContours(mask, [contour], 0, (255, 255, 255), thickness=cv2.FILLED)

    # Extract pixel values within the contour using the mask
    extracted_image = cv2.bitwise_and(image, mask)
    return extracted_image



def image_outside_contour(contour,image):

    # Create a mask using the contour
    mask = np.zeros_like(image[:,:,0])
    cv2.drawContours(mask, [contour], 0, (255), -1) # Draw the contour filled with white color (255)

    # Invert the mask
    mask_inv = cv2.bitwise_not(mask)

    # Set the pixels inside the contour to zero in the original image
    image_inside_contour_removed = cv2.bitwise_and(image, image, mask=mask_inv)
    return image_inside_contour_removed



##########################################################################################



def find_state(x, y, img_width, img_height):

    for i, hole_center in enumerate(hole_centers, start=1):
        distance_squared = (x - hole_center[0])**2 + (y - hole_center[1])**2
        if distance_squared <= solo_radius**2:
            return 10+i  # Return the index of the circular section (1 to 6)
    
    # If not in a circular region, determine the octant

    if x < img_width // 2:
        if y < img_height // 4:
            return 21     # Top Left
        elif img_height // 4 <= y < 2*img_height // 4:
            return 23     # Middle-Top Left
        elif 2*img_height // 4 <= y < 3*img_height // 4:
            return 25     # Middle-Bottom Left
        else:
            return 27     # Bottom Left
    else:
        if y < img_height // 4:
            return 22     # Top Right
        elif img_height // 4 <= y < 2*img_height // 4:
            return 24     # Middle-Top Right
        elif 2*img_height // 4 <= y < 3*img_height // 4:
            return 26     # Middle-Bottom Right
        else:
            return 28     # Bottom Right



def calculate_luminance(ball_contour,image):

    extracted_image=image_inside_contour(ball_contour, image)

    # Convert to LAB color space
    lab_image = cv2.cvtColor(extracted_image, cv2.COLOR_BGR2LAB)

    # Convert the pixels to a list of 3D points
    pixels = lab_image.reshape((-1, 3)).astype(np.float32)
    non_black_pixels = pixels[pixels[:, 0] != 0] # Extract pixels where the first element is not zero

    # Calculate the mean luminance within the contour
    luminance = np.mean(non_black_pixels[:, 0])  # Access the L channel
    return luminance



def classify_ball_luminance(ball_luminance):
    # Classify the ball based on luminance
    if ball_luminance < ball_luminance_limits[0]:
        ball_class= "Hole"
    elif ball_luminance > ball_luminance_limits[3]:
        ball_class= "C"
    elif ball_luminance < ball_luminance_limits[1]: 
        ball_class= "8"
    elif ball_luminance > ball_luminance_limits[2]:
        ball_class= "||"
    else:
        ball_class="O"

    return ball_class



def calculate_color_ratio(ball_image):
    hls_image = cv2.cvtColor(ball_image, cv2.COLOR_BGR2HLS)

    # Threshold the region to get binary masks for white and colored balls
    white_mask = cv2.inRange(hls_image, color_ratio_mask_limits[0], color_ratio_mask_limits[1])
    colored_mask = cv2.inRange(hls_image, color_ratio_mask_limits[2], color_ratio_mask_limits[3])

    num_white_pixels = cv2.countNonZero(white_mask)
    num_colored_pixels = cv2.countNonZero(colored_mask)
    if num_white_pixels:
        color_ratio=num_colored_pixels/num_white_pixels
    else:
        color_ratio=100

    return color_ratio



def classify_ball_color_ratio(color_ratio):

    if color_ratio>ball_color_limits[2]:
        return '8'
    elif color_ratio<ball_color_limits[0]:
        return 'C'
    elif color_ratio>ball_color_limits[1]:
        return 'O'
    else:
        return '||'



##########################################################################################



def calculate_ball_counts(balls):
    
    ball_counts=[0,0,0,0]

    for ball in balls:
        ball_class=ball[0]

        if ball_class=='C':
            ball_counts[0]+=1
        elif ball_class=='O':
            ball_counts[1]+=1
        elif ball_class=='8':
            ball_counts[2]+=1
        elif ball_class=='||':
            ball_counts[3]+=1
    
    return ball_counts



def resolve_ball_impossibilities(balls):
    if len(balls) > 0:      # Balls may not have been found at each frame.

        #There can only be one cue (Lightest) and one 8 Ball (Darkest)
        balls_classes = [ball[0] for ball in balls]
        count_C = balls_classes.count("C")
        count_8 = balls_classes.count("8")

        balls_lumini = [ball[5] for ball in balls]
        min_luminance=min(balls_lumini)
        max_luminance=max(balls_lumini)
        
        if count_C>1:                   # If more than one cue ball is detected reclassify the ones without the highest luminance as striped balls.
            for ball in balls:
                if ball[0]=="C" and ball[5]<max_luminance:
                    ball[0]="||"

        if count_8>1:                   # If more than one 8 ball is detected reclassify the ones without the lowest luminance as solid balls.
            for ball in balls:
                if ball[0]=="8" and ball[5]>min_luminance:
                    ball[0]="O"
        
        #If 16 balls were found the counts must be 1 7 1 7. (Can resolve only if there is 1 extra solid or striped ball)
        ball_counts=calculate_ball_counts(balls)
        if sum(ball_counts) == 16 and ball_counts != [1,7,1,7]:
            if ball_counts[1]==8:    #There is a (wrong) extra solid ball
                #Out of the 3 lowest color_ratio solid balls, the highest luminance should actually be striped

                balls_sol = [ball for ball in balls if ball[0] == 'O']                      # Filter balls with class 'O'
                balls_sol_sorted = sorted(balls_sol, key=lambda ball: ball[6])              # Sort the filtered balls based on color ratio
                balls_lowest_color_ratio = balls_sol_sorted[:3]                             # Select the first three balls with the lowest color ratio
                balls_lowest_color_ratio_sorted_by_luminance = sorted(balls_lowest_color_ratio, key=lambda ball: ball[5], reverse=True)
                ball_highest_luminance = balls_lowest_color_ratio_sorted_by_luminance[0]    # Select the ball with the highest luminance
                ball_highest_luminance[0] = '||'                                            # Change its class to '||'

                # Iterate over the original balls array to find and replace the corrected ball
                for i, ball in enumerate(balls):
                    if ball == ball_highest_luminance:
                        balls[i] = ball_highest_luminance
                        break  # Exit loop once the correction is made
                
                

            elif ball_counts[3]==8:  #There is a (wrong) extra striped ball
                #Out of the 3 highest color_ratio striped balls, the lowest luminance should actually be solid

                
                balls_str = [ball for ball in balls if ball[0] == '||']                     # Filter balls with class '||'
                balls_str_sorted = sorted(balls_str, key=lambda ball: ball[6], reverse=True)# Sort the filtered balls based on color ratio in descending order
                balls_highest_color_ratio = balls_str_sorted[:3]                            # Select the first three balls with the highest color ratio
                balls_highest_color_ratio_sorted_by_luminance = sorted(balls_highest_color_ratio, key=lambda ball: ball[5])
                ball_lowest_luminance = balls_highest_color_ratio_sorted_by_luminance[0]    # Select the ball with the lowest luminance
                ball_lowest_luminance[0] = 'O'                                              # Change its class to 'O'

                # Iterate over the original balls array to find and replace the corrected ball
                for i, ball in enumerate(balls):
                    if ball == ball_lowest_luminance:
                        balls[i] = ball_lowest_luminance
                        break  # Exit loop once the correction is made
        
        return balls
    else:
        return balls



##########################################################################################



def detect_balls_better(image, contours):
    balls = []

    # Get the height and width of the image
    img_height, img_width, _ = image.shape

    for contour in contours:
        
        # Find the minimum enclosing rectangle
        _, _, rect_width, rect_height = cv2.boundingRect(contour)
        rect_aspect_ratio = float(rect_width) / rect_height

        # Find the minimum enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)
        ball_state = find_state(center[0], center[1], img_width, img_height)    # Ball state is a measure of position

        # Find the contour area
        area = abs(cv2.contourArea(contour))

        # Eliminate each contour preemptively by their size and aspect ratio
        if ball_radius_limits[0] <= radius <= ball_radius_limits[1] and ball_aspect_ratio_limits[0] < rect_aspect_ratio < ball_aspect_ratio_limits[1] and ball_area_limits[0] < area < ball_area_limits[1]:
            
            ball_luminance=calculate_luminance(contour, image)
            ball_class=classify_ball_luminance(ball_luminance)


            if ball_class != "Hole":
                ball_image=image_inside_contour(contour, image)

                if (ball_class == "||" or ball_class == "O") and (ball_state!=27 and ball_state!=28 and ball_state!=15 and ball_state!=16):    # Cue and 8 Balls are classified really well using just luminance. Also sectors 27 and 28 are better classified with luminance
                    ball_color_ratio = calculate_color_ratio(ball_image)                                     # But classify striped or solids better.
                    ball_class =classify_ball_color_ratio(ball_color_ratio)
                else:
                    ball_color_ratio=calculate_color_ratio(ball_image)

                # Create a single entry for the ball
                ball = [ball_class, ball_state, center, contour, ball_image, ball_luminance, ball_color_ratio]

                # Append the entry to the list
                balls.append(ball)

    balls=resolve_ball_impossibilities(balls)

    return balls



##########################################################################################


# Version 4 did_ball_move not version 3
def did_ball_move(before_ball,after_img_reference):        # Is this ball in the same place in this frame?

    before_ball_contour=before_ball[3]
    
    # The Image where the before_ball was.
    new_ball_image=image_inside_contour(before_ball_contour, after_img_reference)
    _, ball_thresholded = cv2.threshold(cv2.cvtColor(new_ball_image, cv2.COLOR_BGR2GRAY), 50, 255, cv2.THRESH_BINARY)
    ball_nonzero_pixels = cv2.countNonZero(ball_thresholded) 
  

    # The difference between where the ball was 
    before_ball_image=before_ball[4]
    ball_image_difference = cv2.absdiff(before_ball_image, new_ball_image)

    # If there are less non-zero pixels than the threshold (images nearly identical) then the ball didnt move, otherwise the ball moved!
    # Threshold the grayscale image to find pixels with luminance higher than 50
    _, ball_thresholded_difference = cv2.threshold(cv2.cvtColor(ball_image_difference, cv2.COLOR_BGR2GRAY), 50, 255, cv2.THRESH_BINARY)   #Eliminate all very-dark pixels from the difference. (Sometimes the only difference is shadows)
    ball_difference_nonzero_pixels = cv2.countNonZero(ball_thresholded_difference) 
    if ball_difference_nonzero_pixels < pixel_difference_thresholds[1] and ball_nonzero_pixels >= pixel_difference_thresholds[0]:        #Images nearly identical: Ball didn't move.
        
        #ball_difference_nonzero_pixels checks if the image didn't change much. (No movement)
        #ball_nonzero_pixels limit eliminates if some error causes a black image to be held as the 8 ball. (Deletes completely black images.)
        return 0
    else:           # Otherwise ball moved.
        return 1



def calculate_master_ball_counts(master_balls):

    detected_ball_counts=[0,0,0,0]

    for master_ball in master_balls:
        master_ball_state=master_ball[1]
        if master_ball_state>0:
            master_ball_class=master_ball[0]

            if master_ball_class=='C':
                detected_ball_counts[0]+=1
            elif master_ball_class=='O':
                detected_ball_counts[1]+=1
            elif master_ball_class=='8':
                detected_ball_counts[2]+=1
            elif master_ball_class=='||':
                detected_ball_counts[3]+=1

    return detected_ball_counts



def update_master_balls(master_balls,new_frame):

    new_img_table=get_table(new_frame,img_width,img_height)                 # Get the table only, rectified
    new_img_reference=remove_tablecloth(new_img_table)                      # Remove the tablecloth, only the balls and other objects should be left.
    new_img_subtracted=new_img_reference

    master_ball_matched_flags=[]                                            # Flag 0 if unmatched (needs to be matched), Flag 1 if matched, Flag -1 if ball not in game or not detected. (Doesn't need matching)
    for i, master_ball in enumerate(master_balls):                          # Initialize an array of matched_flags. The master balls in game should need matching.
        master_ball_state=master_ball[1]
        if master_ball_state>0:                                             # If a master_ball is in game, it needs to be matched with the balls on the table.
            master_ball_matched_flags.append(0)
        else:
            master_ball_matched_flags.append(-1)


    
        

    #TAKE EACH MASTER BALL THAT NEEDS MATCHING. FIND (MATCH IT TO A BALL ON THE TABLE) IT, UPDATE IT, REMOVE IT FROM FRAME. ANY BALLS STILL DETECTED AFTER THAT ARE NEW DETECTIONS, ADD THEM AS IS.
    #if a master ball is completely stationary -> just recalculate image data. Eliminate from new_balls.
    #if a master ball is not where it was -> find the new balls, find the closest new ball. This is where this master_ball went. Update all its data with the new data.
    #after these if there is still a ball it must be an undetected master_ball add it into the master balls.

    # MATCH STATIONARY BALLS
    for i, master_ball in enumerate(master_balls):                          # First do a slight update for the balls that didn't move.
        if master_ball_matched_flags[i]==0:                                 # If the ball needs matching
            master_ball_class=master_ball[0]
            master_ball_state=master_ball[1]
            master_ball_center=master_ball[2]
            master_ball_contour=master_ball[3]
            movement_state=did_ball_move(master_ball,new_img_subtracted)    #See if this ball moved in the new frame.


            if movement_state==0:                                           #If this ball didn't move it is matched to itself.

                master_ball_matched_flags[i]=1
                if v4_dbg_flag==1: print(f'Stationary Ball\t\t{i}')

                # Remove the stationary ball from the after_frame. We want to only redetect the balls that have moved.
                new_img_subtracted= image_outside_contour(master_ball_contour,new_img_subtracted)


    # Now that we have removed the stationary balls from the image. We can detect the moved balls.
    new_contours=detect_contours(new_img_subtracted)
    moved_balls=detect_balls_better(new_img_subtracted,new_contours)
    # These balls are either the moved versions of already detected master balls. (In which case they must be matched to their master ball and update the master ball)
    # Or these balls can also be completely new detections.



    # MATCH MOVED BALLS
    if len(moved_balls)!=0:                                                         # If any balls are now detected.
        moved_ball_matched_flags=np.zeros(len(moved_balls))

        for i, master_ball in enumerate(master_balls):                              # These are the master balls that need matching.
            if master_ball_matched_flags[i]==0:
                master_ball_class=master_ball[0]
                master_ball_state=master_ball[1]
                master_ball_center=master_ball[2]
                master_ball_contour=master_ball[3]

                for j, moved_ball in enumerate(moved_balls):
                    moved_ball_class = moved_ball[0]
                    moved_ball_center = moved_ball[2]


                    if cv2.pointPolygonTest(master_ball_contour, moved_ball_center, measureDist=False) >= 0:      # If the center of the new moved ball is inside the contour of an old ball.
                        if v4_dbg_flag==1: print(f'Slight Movement\t\t{i}')
                        moved_ball[0]=master_ball_class
                        master_balls[i] = moved_ball                                                                # (Case where a ball has slightly rolled over.)
                        master_ball_matched_flags[i]=1                                                              # Match these balls.
                        moved_ball_matched_flags[j]=1

                    elif moved_ball_class=='C' and master_ball_class=='C':                                          # If the cue ball is within the moved balls, match these balls (Assuming cue ball is always detectedd correctly) (This has lower priority than the slight rolling case.)
                        master_balls[i] = moved_ball
                        master_ball_matched_flags[i]=1
                        moved_ball_matched_flags[j]=1
                        if v4_dbg_flag==1:  print(f'Cue Ball\t\t{i}')
                    


        # At this point if there are still unmatched moved balls assign them to the closest master_ball. If there aren't any master_balls left that need matching these are new detections we will simply make them master balls after this loop.
        distance_between_unmatched_balls=[]
        for j, moved_ball in enumerate(moved_balls):
            if moved_ball_matched_flags[j]==0:
                moved_ball_class = moved_ball[0]
                moved_ball_center = moved_ball[2]
                for i, master_ball in enumerate(master_balls):                              # These are the master balls that need matching.
                    if master_ball_matched_flags[i]==0:                                     # Else find the distance to this unmatched master ball. We will match the moved ball to the closest unmatched master ball.
                        master_ball_class=master_ball[0]
                        master_ball_state=master_ball[1]
                        master_ball_center=master_ball[2]
                        master_ball_contour=master_ball[3]

                        distance_to_unmatched_master_ball=cv2.norm(master_ball_center,moved_ball_center)
                        distance_between_unmatched_balls.append((i,j,distance_to_unmatched_master_ball))

        if len(distance_between_unmatched_balls)!=0:
            distance_between_unmatched_balls.sort(key=lambda x: x[2])   # Sorting the distances between unmatched balls

            # Matching moved balls to master balls based on closest distances
            for master_ball_index, moved_ball_index, distance in distance_between_unmatched_balls:
                if master_ball_matched_flags[master_ball_index]==0 and moved_ball_matched_flags[moved_ball_index]==0 and distance < matching_radius:
                    if not( (moved_ball_class=='C' and master_ball_class=='O')  or (moved_ball_class=='O' and master_ball_class=='C')  or 
                            (moved_ball_class=='||' and master_ball_class=='8') or (moved_ball_class=='8' and master_ball_class=='||') or
                            (moved_ball_class=='C' and master_ball_class=='8')  or (moved_ball_class=='8' and master_ball_class=='C')):
                        
                        # Assign same class to master ball. (No History, No class correction.)
                        master_ball=master_balls[master_ball_index]
                        moved_ball=moved_balls[moved_ball_index]
                        moved_ball[0]=master_ball[0]

                        master_balls[master_ball_index] = moved_ball
                        master_ball_matched_flags[master_ball_index] = 1
                        moved_ball_matched_flags[moved_ball_index] = 1
                        if v4_dbg_flag==1: print(f'Assigned To Closest\t{master_ball_index}')






        # ADD IN NEWLY DETECTED BALLS INTO THE MASTER_BALLS
        for j, moved_ball in enumerate(moved_balls):        # If there are still unmatched moved_balls these must be newly detected master_balls. Make them master_balls
            if moved_ball_matched_flags[j]==0:
                moved_ball_class = moved_ball[0]
                moved_ball_state = moved_ball[1]
                
                index_of_matching_ball = None                                   # Find empty master slot and assign this ball to there.
                for i, master_ball in enumerate(master_balls):
                    master_ball_class=master_ball[0]
                    master_ball_state=master_ball[1]
                    master_ball_center=master_ball[2]
                    master_ball_contour=master_ball[3]

                    if master_ball_state == -1 and master_ball_class == moved_ball_class:  # Check if state is -1 (New Detection) or 0 (Was Classified as Potted, it was a false positive so now readd it) and class matches
                        master_balls[i]=moved_ball
                        master_ball_matched_flags[i]=1
                        moved_ball_matched_flags[j]=1
                        if v4_dbg_flag==1: print(f'Newly Detected Ball\t{i}')
                        break   # This moved_ball has been placed, go to the next moved_ball.
                    elif master_ball_state == 0 and master_ball_class == moved_ball_class and 10 <= moved_ball_state <= 19:  # Potted False Positive Recorrection. There must be a movedball in a solo region detected for a already potted ball. Has lesser priority than population
                        master_balls[i]=moved_ball
                        master_ball_matched_flags[i]=1
                        moved_ball_matched_flags[j]=1
                        if v4_dbg_flag==1: print(f'Potted Recorrection\t{i}')
                        break   # This moved_ball has been placed, go to the next moved_ball.

    # POTTED CHECK
    for i, master_ball in enumerate(master_balls):
        master_ball_state=master_ball[1]                             
        if master_ball_matched_flags[i]==0:      # At the end of all these matchings if there is still a master_ball waiting to be matched, that ball has left the table: i.e. POTTED (STATE 0) or LOST (STATE -1)             
            if 11<=master_ball_state<=16:       # If it left the table from a solo sector it was probably potted.
                master_ball[1]=0
                master_balls[i]=master_ball
                master_ball_matched_flags[i]=1
                if v4_dbg_flag==1: print(f'Potted Ball\t{i}')
            else:                               # If it is not found but it was not in a solo sector the last time it was seen. It is probably simply not detected.
                master_ball[1]=-1
                master_balls[i]=master_ball
                master_ball_matched_flags[i]=1
                if v4_dbg_flag==1: print(f'Lost Ball\t{i}')





    # IF CUE BALL IS LOST MAYBE CHECK IF IT WAS MISCLASSIFIED
    cue_ball=master_balls[0]
    cue_ball_state=cue_ball[1]

    if cue_ball_state<=9:                                                         # If the cue ball is not on the table but there is a saved master ball that could be the cue ball, make that master ball the cue ball.
        for i, master_ball in enumerate(master_balls):

            master_ball_luminance=master_ball[5]
            master_ball_reclass=classify_ball_luminance(master_ball_luminance)      # Re-calculate the class of the master ball, if it is supposed to be the cue ball make it.

            if master_ball_reclass=='C':
                master_balls[i][0]='||'                 #Free up the wrongly classified master ball. Cue ball could have only been wrongfully classified as '||' due to our limitations above.
                master_balls[i][1]=-1

                master_ball[0]='C'                      #Migrate the ball data to master 'C' ball.
                master_balls[0]=master_ball



    # Extremely specific correction for aesthetics: If a cue ball is hit very fast with the cue. And if the tip of the cue was mistakenly detected as a ball just as it hits the cue ball and the cue ball detection is interrupted. The mistakenly detected ball can be overwritten by the ghost of the cue ball. Basically this situation changes the class of a || master ball to C. Never supposed to happen.
    # If this explanation is complex don't worry about it. It basically makes sure that the classes of the Master Balls are always C, O*7, 8, ||*7 as they are never meant to be changed.
    # The program is robust enough that even without this section it can keep track of eerything correctly.
    for i, ball in enumerate(master_balls):         # At indexes 1-7 and 9-15 it will hold the solid or striped balls' data in no particular order since we don't classify the balls by specific color or number.
        if i==0:                # Cue Ball
            ball[0]='C'
        elif 1 <= i <= 7:       # Solid Balls
            ball[0]='O'
        elif i==8:              # 8 Ball
            ball[0]='8'
        elif 9 <= i <= 15:      # Striped Balls
            ball[0]='||'


    return master_balls





















##########################################################################################
# V3 DEPRECIATED!
##########################################################################################


def resolve_add_new_balls(balls,new_balls):
    for new_ball in new_balls:

        # Flag to indicate if the new ball has been placed
        placed = False

        new_ball_center=new_ball[2]
        
        for i, ball in enumerate(balls):
            # Check if the center of the new ball is inside the contour of an old ball. (Case where a ball has slightly rolled over.)
            ball_contour=ball[3]
            if cv2.pointPolygonTest(ball_contour, new_ball_center, measureDist=False) >= 0:
                # Replace the current ball in the array with the new ball
                balls[i] = new_ball
                placed = True
                break
        
        if not placed:
            # If new ball was not placed, append it to the array
            balls.append(new_ball)

    return balls






def detect_balls(image, contours):
    balls = []

    # Get the height and width of the image
    img_height, img_width, _ = image.shape

    for contour in contours:
        
        # Find the minimum enclosing rectangle
        _, _, rect_width, rect_height = cv2.boundingRect(contour)
        rect_aspect_ratio = float(rect_width) / rect_height

        # Find the minimum enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)

        # Classify each contour by size and brightness.
        if ball_radius_limits[0] <= radius <= ball_radius_limits[1] and ball_aspect_ratio_limits[0] < rect_aspect_ratio < ball_aspect_ratio_limits[1]:
            ball_luminance=calculate_luminance(contour, image)
            ball_class=classify_ball_luminance(ball_luminance)


            if ball_class != "Hole":
                ball_state = find_state(center[0], center[1], img_width, img_height)
                ball_image=image_inside_contour(contour, image)
                ball_color_ratio=calculate_color_ratio(ball_image)

                # Create a single entry for the ball
                ball = [ball_class, ball_state, center, contour, ball_image, ball_luminance, ball_color_ratio]

                # Append the entry to the list
                balls.append(ball)

    balls=resolve_ball_impossibilities(balls)

    return balls






        

        









