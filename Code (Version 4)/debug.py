import cv2
import numpy as np
from variables import img_height, img_width, master_color_array, hole_centers, solo_radius
import f

def gui_image(window_name,img_reference,gui_scale):
    image = img_reference.copy()
    img_height, img_width, _ = image.shape

    cv2.imshow(window_name, cv2.resize(image, (int(img_width / gui_scale), int(img_height / gui_scale))))


def gui_image_with_contours(window_name,img_reference,gui_scale,contours):
    image = img_reference.copy()
    img_height, img_width, _ = image.shape

    print(f"Number of contours: {len(contours)}")

    for contour in contours:
        cv2.drawContours(image, [contour], 0, (255, 0, 0), int(1.5*gui_scale))      # Draw the contour

        (x, y), radius = cv2.minEnclosingCircle(contour)                            # Draw the smallest encapsulating circle we will use for elimination 
        center = (int(x), int(y))
        radius = int(radius)
        cv2.circle(image, center, radius, (255, 255, 0), int(1.5 * gui_scale))      

    cv2.imshow(window_name, cv2.resize(image, (int(img_width / gui_scale), int(img_height / gui_scale))))



def gui_image_with_ball_classes(window_name,img_reference,gui_scale,balls):
    image = img_reference.copy()
    img_height, img_width, _ = image.shape

    print(f"Number of detected balls: {len(balls)}")
    for ball in balls:
        ball_class,ball_state,ball_center,ball_contour,ball_image,ball_luminance,ball_color_ratio=ball
        print(f"Class: {ball_class}\tState: {ball_state}\tCenter: {ball_center}\tLuminance: {ball_luminance}\tColor Ratio: {ball_color_ratio}")
    
        cv2.drawContours(image, [ball_contour], 0, (0, 255, 0), int(1.5*gui_scale))
        cv2.putText(image, ball_class, (ball_center[0], ball_center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5*gui_scale, (0, 0, 255), int(1.5*gui_scale))

    cv2.imshow(window_name, cv2.resize(image, (int(img_width / gui_scale), int(img_height / gui_scale))))


def gui_image_with_ball_states(window_name,img_reference,gui_scale,balls):
    image = img_reference.copy()
    img_height, img_width, _ = image.shape

    # Draw circles around the specified centers on the original image
    for hole_center in hole_centers:
        cv2.circle(image, hole_center, solo_radius, (0, 0, 255), int(1.5*gui_scale))

    # Draw the octant borders
    line1_start = (img_width // 2, 0)
    line1_end = (img_width // 2, img_height)

    line2_start = (0, img_height // 4)
    line2_end = (img_width, img_height // 4)

    line3_start = (0, 2*img_height // 4)
    line3_end = (img_width, 2*img_height // 4)

    line4_start = (0, 3*img_height // 4)
    line4_end = (img_width, 3*img_height // 4)


    # Draw all the lines on the image in red
    cv2.line(image, line1_start, line1_end, (0, 0, 255), int(1.5*gui_scale))  # Red color, thickness 2
    cv2.line(image, line2_start, line2_end, (0, 0, 255), int(1.5*gui_scale))  # Red color, thickness 2
    cv2.line(image, line3_start, line3_end, (0, 0, 255), int(1.5*gui_scale))  # Red color, thickness 2
    cv2.line(image, line4_start, line4_end, (0, 0, 255), int(1.5*gui_scale))  # Red color, thickness 2


    # Draw the balls with their states
    for ball in balls:
        ball_class,ball_state,ball_center,ball_contour,ball_image,ball_luminance,ball_color_ratio=ball
    
        cv2.drawContours(image, [ball_contour], 0, (0, 255, 0), 2)
        cv2.putText(image, str(ball_state), (ball_center[0], ball_center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5*gui_scale, (0, 0, 255), int(1.5*gui_scale))

    cv2.imshow(window_name, cv2.resize(image, (int(img_width / gui_scale), int(img_height / gui_scale))))




def gui_master_balls(window_name,frame,gui_scale,master_balls,save_flag,output_path):
    img_table=f.get_table(frame,img_width,img_height)     # Get the table only, rectified
    img_reference=f.remove_tablecloth(img_table)          # Remove the tablecloth, only the balls and other objects should be left.

    image = img_reference.copy()

    master_ball_counts=f.calculate_master_ball_counts(master_balls)
    ball_count=sum(master_ball_counts)
    
    text_lines = [
        "Ongoing Game...",
        f"{ball_count} Balls are in game.",
        f"{master_ball_counts[3]} (||)",
        f"{master_ball_counts[1]} (O)",
        f"{master_ball_counts[2]} (8)",
        f"{master_ball_counts[0]} Cue",
        "--------------------------",
        "Line 6",
        "Line 7",
        "Line 8",
        "Line 9",
        "Line 10",
        "Line 11",
        "Line 12",
        "Line 13",
        "Line 14",
        "Line 15",
        "Line 16",
        "Line 17",
        "Line 18",
        "Line 19",
        "Line 20",
        "Line 21"
    ]

    if master_ball_counts[3]==0 and master_ball_counts[2]==0:       # If no striped or 8 balls are left.
        text_lines[0]="STRIPED WINS!"
    elif master_ball_counts[1]==0 and master_ball_counts[2]==0:     # If no solid or 8 balls are left.
        text_lines[0]="SOLID WINS!"



    if save_flag!=1:print(f"Number of detected master balls: {ball_count}")

    for i, ball in enumerate(master_balls):
        ball_class,ball_state,ball_center,ball_contour,ball_image,ball_luminance,ball_color_ratio=ball

        if ball_state>0:        # Draw the balls that are on the table. (Detected, not potted)
            cv2.drawContours(image, [ball_contour], 0, master_color_array[i], int(1.5*gui_scale))
            if ball_class=='O':
                cv2.putText(image, f'{ball_class}{i}', (ball_center[0], ball_center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5*gui_scale, master_color_array[i], int(1.5*gui_scale))
            elif ball_class=='||':
                cv2.putText(image, f'{i}{ball_class}', (ball_center[0], ball_center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5*gui_scale, master_color_array[i], int(1.5*gui_scale))
            else:
                cv2.putText(image, f'{ball_class}', (ball_center[0], ball_center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5*gui_scale, master_color_array[i], int(1.5*gui_scale))

        text_lines[i+7]=f"{ball_class}    State: {ball_state}    Lum: {int(ball_luminance)}    CR: {int(ball_color_ratio)}"

        if save_flag!=1: print(f"Class: {ball_class}\tState: {ball_state}\tCenter: {ball_center}\tLuminance: {int(ball_luminance)}\tColor Ratio: {int(ball_color_ratio)}")

            



    # GAME STATE TEXT
    # Create a new blank image with the desired dimensions
    new_image = np.zeros((img_width*2, img_height, 3), dtype=np.uint8)

    # Add the imported image to the left side
    new_image[:img_height, :img_width] = image

    # Add text lines to the right side
    line_height = img_height // (len(text_lines)+2)
    for i, line in enumerate(text_lines):
        y = (i+1) * line_height + line_height // 2
        cv2.putText(new_image, line, (img_width + 50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5*gui_scale, (255, 255, 255), int(1.5*gui_scale))

    cv2.line(new_image, (img_width, 0), (img_width, img_height), (255, 255, 255), thickness=2)

    if save_flag!=1:
        cv2.imshow(window_name, cv2.resize(new_image, (int(2*img_width / gui_scale), int(img_height / gui_scale))))
    else:
        cv2.imwrite(output_path, new_image)




def print_master_balls(master_balls):
    print(f"Number of detected master balls: {sum(f.calculate_master_ball_counts(master_balls))}")
    for ball in master_balls:
        ball_class,ball_state,ball_center,ball_contour,ball_image,ball_luminance,ball_color_ratio=ball
        print(f"Class: {ball_class}\tState: {ball_state}\tCenter: {ball_center}\tLuminance: {ball_luminance}\tColor Ratio: {ball_color_ratio}")








