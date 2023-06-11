# See relevant sources in function description.
# #!/usr/bin/env python3
import sys
import pikepdf
from PIL import Image
import PIL.ImageOps    
from pdf2image import convert_from_path
import os
import cv2
import numpy as np
import math
from pytesseract import pytesseract
import imutils
import platform
from matplotlib import pyplot as plt
import random
from scipy import optimize

# Global params
A0_Width  = 841
A0_Height = 1189    

# Extract relevant PDF layers
# Source: https://gist.github.com/jangxx/bd9256009b6698f1550fb7034003f877.
# Made relevant changes.
def pdfLayers(pdf_name, pdf_out, desired_layers):
    """
    Extract relevant PDF layers  \n
    Args:
        pdf_name - the main pdf file, e.g: 'main_pdf.pdf' \n
        pdf_out - format to save the pdf files, e.g: 'Out_{num}.pdf' \n
        desired_layers - layers to extract from the main pdf file \n
 
    Returns:
        void, saves the desired layers as pdf files in pdf_out format.
    """
    # check if we even have some OCGs
    pdf = pikepdf.open(pdf_name)

    try:
        layers = pdf.Root.OCProperties.OCGs
    except (AttributeError, KeyError):
        print("Unable to locate layers in PDF.")
        sys.exit(1)

    page_count = len(pdf.pages)
    pdf.close()

    # (hopefully) all pdf operators which "display" anything. everything else is styling, which we need to preserve
    hidden_operators = ["S", "s", "f" "F", "f*", "B", "B*", "b", "b*", "n", "Do", "sh", "Tj", "TJ", "m", "l", "c", "v", "y", "h", "re"]

    extracted_groups = []
    cur_layer = 0

    for i in range(page_count):
        end_reached = False

        while not end_reached:
            commands = []
            extract_commands = True
            extracted_one = False

            pdf = pikepdf.open(pdf_name)
            page = pdf.pages[i]

            for j in range(len(pdf.pages)):
                if i < j:
                    del pdf.pages[1]
                elif i > j:
                    del pdf.pages[0]

            for operands, operator in pikepdf.parse_content_stream(page):
                if pikepdf.Name("/OC") in operands: # new OCG starts
                    ocg_name = operands[1]
                    if not ocg_name in extracted_groups and not extracted_one:
                        extracted_groups.append(ocg_name)
                        extract_commands = True
                        extracted_one = True
                    else:
                        extract_commands = False
            
                if str(operator) == "EMC": # OCG has ended
                    extract_commands = True
                    continue

                if extract_commands or (not extract_commands and str(operator) not in hidden_operators):
                    commands.append([ operands, operator ])

                    # if cur_layer == 6:
                    #     print("Operands {}, operator {}".format(operands, operator))

            if not extracted_one:
                end_reached = True
            else:
                page.Contents = pdf.make_stream(pikepdf.unparse_content_stream(commands))
                # pdf.save(sys.argv[2].format(num=cur_layer))
                if (cur_layer in desired_layers):
                    pdf.save(pdf_out.format(num=cur_layer))                
                    
                cur_layer += 1

def pdf2image(desired_layers, pdf_out, img_out):
    """
    Save each given pdf as an image  \n
    Args:
        desired_layers - pdf numbers to look for, e.g 'Out_0.pdf' \n        
        pdf_out - pdf file format to read, e.g: 'Out_{num}.pdf' \n
        img_out - format to save the image files, e.g: 'Out_{num}.png'
        
    Returns:
        one of the image's shape, img.shape, (y,x), (All the image's shapes are identical).
    """
    # path = os.path.realpath(os.path.dirname(__file__))
    path = os.getcwd()

    # Store Pdf with convert_from_path function
    for j in range(len(desired_layers)):
        pdf_path = path + '/' + pdf_out.format(num=desired_layers[j])
        images = convert_from_path(pdf_path, dpi=150)
        # image.save(str(desired_layers[j]) +'.jpg', 'JPEG')
        for i in range(len(images)):        
            # Save pages as images in the pdf
            images[i].save(path + '/' + img_out.format(num=desired_layers[j]), 'PNG')
    img = cv2.imread(img_out.format(num=desired_layers[0]))
    return img.shape




# Source: https://richardpricejones.medium.com/drawing-a-rectangle-with-a-angle-using-opencv-c9284eae3380
# Made slight adjustments to color
def draw_angled_rec(x0, y0, width, height, angle, img, color):

    _angle = angle * math.pi / 180.0
    b = math.cos(_angle) * 0.5
    a = math.sin(_angle) * 0.5
    pt0 = (int(x0 - a * height - b * width),
           int(y0 + b * height - a * width))
    pt1 = (int(x0 + a * height - b * width),
           int(y0 - b * height - a * width))
    pt2 = (int(2 * x0 - pt0[0]), int(2 * y0 - pt0[1]))
    pt3 = (int(2 * x0 - pt1[0]), int(2 * y0 - pt1[1]))

    if color == 'green':
        cv2.line(img, pt0, pt1, (0, 255, 0), 5)
        cv2.line(img, pt1, pt2, (0, 255, 0), 5)
        cv2.line(img, pt2, pt3, (0, 255, 0), 5)
        cv2.line(img, pt3, pt0, (0, 255, 0), 5)
    else:
        cv2.line(img, pt0, pt1, (0, 0, 255), 5)
        cv2.line(img, pt1, pt2, (0, 0, 255), 5)
        cv2.line(img, pt2, pt3, (0, 0, 255), 5)
        cv2.line(img, pt3, pt0, (0, 0, 255), 5)


# Source: Roald's response in https://stackoverflow.com/questions/11627362/how-to-straighten-a-rotated-rectangle-area-of-an-image-using-opencv-in-python/48553593#48553593
# Made slight changes
def crop_image(cnt, image, type, ptrn_num, ptrn_imgs):
    rect = cv2.minAreaRect(cnt)
    shape = (image.shape[1], image.shape[0])  # cv2.warpAffine expects shape in (length, height)
    center, size, theta = rect
    width, height = tuple(map(int, size))
    center = tuple(map(int, center))    

    # Allow only for angles of rotation lower than 90 degrees.
    # To simplify handling.
    if theta == 90:
        if width > height:
            alpha = 90
        else:
            alpha = 0
        width, height = height, width
    elif theta == 0:
        if width < height:
            alpha = 90        
            width, height = height, width
        else:
            alpha = 0
    else:                    # theta != 0 and theta != 90:
        alpha = 90 - theta
        if width < height:
            width, height = height, width
        elif type == 'ptrn_save':
            alpha += 90

    if type == 'ptrn_save':
        if alpha < 1:
            alpha = 0
        elif alpha > 89:
            if (abs(alpha - 90) < 1 and (alpha != 90)):
                alpha = 90
        return alpha
    
    x_old = int(center[0])
    y_old = int(center[1])
    # x_old = int(center[0])
    # y_old = int(center[1])
    distance = math.sqrt(x_old**2+y_old**2)
    image = imutils.rotate_bound(image, angle = alpha)
    
    alpha_rad = math.radians(alpha)

    y_temp = y_old*math.cos(alpha_rad) + x_old*math.sin(alpha_rad)
    x_offset = math.sin(alpha_rad) * max(shape)
    phi = math.acos(y_temp/distance)
    beta = math.atan2(x_old, y_old)
    if beta > alpha_rad:
        x_temp = x_offset + math.sin(phi)*distance
    else:
        x_temp = x_offset - math.sin(phi)*distance
    
    y_new = math.floor(y_temp)
    x_new = math.floor(x_temp)

    if type == 'pattern':
        new_height = int(height / 2)
        new_width = int(width / 2)
    elif type == 'direction':
        if width < height:
            new_height = height // 2
            new_width = math.floor(2 * width)           # To make sure the text is encompassed
        else:
            new_width = width // 2
            new_height = math.floor(2 * height)         # To make sure the text is encompassed

    if (y_new - new_height) < 0:
        y_new = new_height
    elif (y_new + new_height) > image.shape[0]:
        y_new = image.shape[0] - new_height
    
    if (x_new - new_width) < 0:
        x_new = new_width
    elif (x_new + new_width) > image.shape[1]:
        x_new = image.shape[1] - new_width
    
    image = image[y_new - new_height : y_new + new_height, x_new - new_width: x_new + new_width]
    cv2.imwrite('img_test.png',image)

    return image


# Find the pattern contours
def find_pattern_contours(image, resized):
    """
    Finds the pattern contours  \n
    Args:
        image - pattern contour image format to read \n        
        resized - bool, whether the image was resized to A0 or not, determines kernel size.
        
    Returns:
        The detected pattern contours
    """
    counter = 0
    img = cv2.imread(image)
        
    if resized:
        kernel_size = 3
    else:
        kernel_size = 7

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # The first parameter is the original image,
    # kernel is the matrix with which the image is
    # convolved and third parameter is the number
    # of iterations, which will determine how much
    # you want to erode/dilate a given image.
    img = cv2.erode(img, kernel, iterations=6)
    img = cv2.dilate(img, kernel, iterations=3)
    ## For debugging
    # image_copy = img.copy()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(img_gray, 240, 255, cv2.THRESH_BINARY)
    # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
    contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)                                            
    good_contours = []
    j = 0
    for cnt in contours:
        if counter == 0:                    # First contour encompasses entire image
            counter += 1
            continue

        good_contours.append(cnt)
        heir = hierarchy[0][counter][3]     # [next, previous, first child, parent].
                                            # See source: https://stackoverflow.com/questions/11782147/python-opencv-contour-tree-hierarchy-structure
        if heir != 0:                       # If heir is 0, means it's the outer most contour, which is what I'm interested in.
            good_contours.pop(j)
        else:             
            j += 1
        counter += 1
    # For debugging
    # cv2.drawContours(image=image_copy, contours=good_contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    # cv2.imwrite('image_copy.png',image_copy)
    return good_contours 


# Find grainline contours
def find_potential_direction_contours(image, ptrn_cntrs):
    """
    Finds direction contours and the pattern contours which contain direction contours inside them.  \n
    Args:
        image - direction contour image format to read \n        
        ptrn_cntrs - The detected pattern contours inside which to look for the direction contours
        
    Returns:
        potential_contours - Potential direction contours \n
        potential_contours_ptrn_index - direction contour pattern index, e.g 0 means relating to pattern in index 0. \n
        ptrn_cntrs_new - The pattern contours which contain direction contours inside them. \n
    """
    img = cv2.imread(image)
    ptrn_cnt_counter = 0
    potential_contours = []
    potential_contours_ptrn_index = []
    ptrn_cntrs_new = []
    dir_ptrn_flag = 0
    for ptrn_cnt in ptrn_cntrs:
        img_tmp = img.copy()
        img_cropped = crop_image(ptrn_cnt, img_tmp, 'pattern', 0, 0)    
        img_gray = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)
        # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
        contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)            
        slender_rat = 3
        min_width = 10
        max_width = 120
        min_len = 150
        first = 1
        image_copy = img_cropped.copy()
        cv2.imwrite('image_copy.png',image_copy) 
        # cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        for cnt in contours:
            if first == 1:        # First contour encompasses entire image
                first = 0
                continue
            i = 0
            rect = cv2.minAreaRect(cnt)
            x = rect[0][0]
            y = rect[0][1]
            w = rect[1][0]
            h = rect[1][1]
            theta = rect[2]   
            draw_angled_rec(x, y, w, h, theta, image_copy, 'green')
            cv2.imwrite('image_copy.png',image_copy)  
            if w == 0 or h == 0:
                continue
            if w/h > slender_rat or h/w > slender_rat:
                if w/h > slender_rat:
                    if (w < min_len or h < min_width or h > max_width):
                        continue
                    else:                    
                        dir_ptrn_flag = 1
                        img_cropped = draw_angled_rec(x, y, w, h, theta, image_copy, 'red')
                        cv2.imwrite('image_copy.png',image_copy)  
                        potential_contours.append(cnt)
                        potential_contours_ptrn_index.append(ptrn_cnt_counter)
                elif h/w > slender_rat:
                    if (h < min_len or w < min_width or w > max_width):
                        continue
                    else:
                        dir_ptrn_flag = 1
                        img_cropped = draw_angled_rec(x, y, w, h, theta, image_copy, 'red')
                        cv2.imwrite('image_copy.png',image_copy) 
                        potential_contours.append(cnt)
                        potential_contours_ptrn_index.append(ptrn_cnt_counter)
        
        # No direction contour found
        if dir_ptrn_flag == 0:
            continue

        ptrn_cntrs_new.append(ptrn_cnt)
        dir_ptrn_flag = 0            
        ptrn_cnt_counter += 1

    return potential_contours, potential_contours_ptrn_index, ptrn_cntrs_new


def save_patterns(ptrn_image, pattern_contours, dir_cnt, dir_ptrn_cnt, ptrn_imgs):
    """
    Crops and saves the pattern contours images before being folded \n
    Args:
        ptrn_image - Main pattern image format to open
        pattern_contours - the pattern contours
        dir_cnt - the direction contours
        dir_ptrn_cnt - the direction contour pattern indices
        ptrn_imgs - the pattern image format to save the images, e.g 'pattern_{num}.png'.
        
    Returns:
        rot_ang - A list of the rotation angles needed for each pattern to get the first (Arbitrary) direction contour horizontal.
    """
    img0 = cv2.imread(ptrn_image)
    rot_ang = []
    for i in range(len(pattern_contours)):
        img1 = crop_image(pattern_contours[i], img0, 'pattern', 0, 0)
        cv2.imwrite('img_test.png',img1)
        cnt = dir_cnt[dir_ptrn_cnt.index(i)]   #Find the first relevent direction contour
        angle = crop_image(cnt, img1, 'ptrn_save', i, ptrn_imgs)
        rot_ang.append(angle)
        img = img1.copy()
        
        kernel_size = int(img0.shape[0]*img0.shape[1] * 0.0000002 + 0.5)
        if kernel_size < 1:
            kernel_size = 3
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        img = cv2.erode(img, kernel, iterations=2)
        img = cv2.dilate(img, kernel, iterations=2)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(img_gray, 230, 255, cv2.THRESH_BINARY)        

        x_min = img.shape[1]
        x_max = 0
        y_min = img.shape[0]
        y_max = 0
        contours, _ = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE) 
        
        first = 0
        for cnt in contours:
            if first == 0:
                first = 1
                continue
            cnt_np = np.asarray(cnt, dtype = object)
            max_val = cnt_np.max(axis=0, keepdims=False)
            min_val = cnt_np.min(axis=0, keepdims=False)
            debug = max_val[0][0]
            for j in range(len(cnt)):
                if max_val[0][0] > x_max:
                    x_max = int(max_val[0][0])
                if max_val[0][1] > y_max:
                    y_max = int(max_val[0][1])
                if min_val[0][0] < x_min:
                    x_min = int(min_val[0][0])
                if min_val[0][1] < y_min:
                    y_min = int(min_val[0][1])
        
        # px_buffer = int(1.5 + img0.shape[1] / A0_Width) * 30
        px_buffer = 0

        if img.shape[0] - y_max < px_buffer:
            y_max = img.shape[0]
        else:
            y_max += px_buffer

        if img.shape[1] - x_max < px_buffer:
            x_max = img.shape[1]
        else:
            x_max += px_buffer

        if y_min < px_buffer:
            y_min = 0
        else:
            y_min -= px_buffer

        if x_min < px_buffer:
            x_min = 0
        else:
            x_min -= px_buffer        

        img_cropped = img1[y_min : y_max, x_min: x_max]
        cv2.imwrite(ptrn_imgs.format(num=i),img_cropped) 
    return rot_ang
    



def find_text(image, pattern_contours, dir_cnt, dir_ptrn_cnt):
    """
    Detects relevant text: fold, grainline, main fabric, lining, # of copies \n
    Args:
        image - direction layer image format
        pattern_contours - the pattern contours
        dir_cnt - the direction contours
        dir_ptrn_cnt - the direction contour pattern indices
        
    Returns:
        copies_list - list of copies per pattern \n
        lining_list -  list of whether the pattern should be on lining per pattern \n
        main_fabric_list -  list of whether the pattern should be on main fabric per pattern \n
        fold_list -  list of fold directions per pattern \n
        dir_cnt - updated direction contour list per pattern. If grainline appears, put it first. \n
    """
    if platform.system() == 'Windows':
        pytesseract.tesseract_cmd=r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    else:
        pytesseract.tesseract_cmd=r'/usr/bin/tesseract'    
    img0 = cv2.imread(image)
    ang_inc = 90
    copies_list = []
    lining_list = []
    main_fabric_list = []
    fold_list = []
    ptrn_counter = 0
    dir_cnt_np = np.asarray(dir_cnt, dtype = object)
    dir_ptrn_cnt_np = np.asarray(dir_ptrn_cnt, dtype = object)
    for ptrn in pattern_contours:
        rect = cv2.minAreaRect(dir_cnt[dir_ptrn_cnt.index(ptrn_counter)])   #Find the first relevent direction contour
        theta = rect[2]
        copies = 1
        lining = 0
        main_fabric = 1
        fold = []
        cropped_img = crop_image(ptrn, img0, 'pattern', 0, 0)    
        for cnt in dir_cnt_np[np.where(dir_ptrn_cnt_np == ptrn_counter)]:
            dir_cropped_img = crop_image(cnt, cropped_img, 'direction', 0, 0)
            for i in range (int(360/ang_inc) - 1):                                      # Rotating 360 deg in 90 deg inc to find all text orientations.
                img = imutils.rotate_bound(dir_cropped_img, angle = (i * ang_inc))      # rotate_bound rotation is clockwise for positive values.
                cv2.imwrite('img_test.png',img)
                text = (pytesseract.image_to_string(img)).lower()
                ## For debugging
                # print(text[:-1])
                if 'fold' in text:
                    fold.append(cnt)                
                    break
                if 'grain' in text:
                    dir_cnt[dir_ptrn_cnt.index(ptrn_counter)] = cnt                
                    break

        if len(fold) == 0:                         
            fold_list.append(0)
        else:
            fold_list.append(fold)                                   

        rotated_img = imutils.rotate_bound(cropped_img, angle = (90 - theta))
        for i in range (int(360/ang_inc) - 1):  
            img = imutils.rotate_bound(rotated_img, angle = (i * ang_inc))
            cv2.imwrite('img_test.png',img)
            text = (pytesseract.image_to_string(img)).lower()            
            ## For debugging
            # print(text[:-1])                                    #print the text line by line
            if 'cut two' in text or 'cut 2' in text:
                copies = 2
            if 'lining' in text:
                lining = 1
                if 'main fabric' not in text:
                    main_fabric = 0
        copies_list.append(copies)
        lining_list.append(lining)
        main_fabric_list.append(main_fabric)
        ptrn_counter += 1
        
    return copies_list, lining_list, main_fabric_list, fold_list, dir_cnt


def fold_patterns(fold_list, pattern_img, rot_ang, size):
    """
    Folds the pattern images based on the given fold list \n
    copies images to larger blank images for easier later manipulation \n
    and saves back to pattern_img format \n
    Args:
        fold_list - list of fold direction contours per pattern \n
        pattern_img - pattern images format \n
        rot_ang - rotation angles needed to make sure the grainlines are horizontal \n
        size - size of the original images to know how much to resize to fit to an A0 size
        
    Returns:
        void
    """
    resize_y = A0_Height / size[0]
    resize_x = A0_Width / size[1]

    for i in range(len(fold_list)):
        if fold_list[i] != 0:            
            flip_code = -1
            for j in range(len(fold_list[i])):
                ptrn_img = cv2.imread(pattern_img.format(num = i))
                img = ptrn_img.copy()
                if len(fold_list[i]) > 1:
                    rect = cv2.minAreaRect(fold_list[i][j])
                else:
                    rect = cv2.minAreaRect(fold_list[i][0])
                x = rect[0][0]
                y = rect[0][1]
                w = rect[1][0]
                h = rect[1][1]
                theta = rect[2] 
                if theta == 0:
                    if w > h:
                        if flip_code == 0:      # Already folded along that side.
                            break
                        else:
                            flip_code = 0
                        if y < (img.shape[0] // 2):
                            flip_side = 'up'
                        else:
                            flip_side = 'down'
                    else:
                        if flip_code == 1:
                            break
                        else:
                            flip_code = 1
                        if x < (img.shape[1] // 2):
                            flip_side = 'left'
                        else:
                            flip_side = 'right'
                else: #theta == 90
                    if w > h:
                        if flip_code == 1:
                            break
                        else:
                            flip_code = 1
                        if x < (img.shape[1] // 2):
                            flip_side = 'left'
                        else:
                            flip_side = 'right'
                    else:
                        if flip_code == 0:
                            break
                        else:
                            flip_code = 0
                        if y < (img.shape[0] // 2):
                            flip_side = 'up'
                        else:
                            flip_side = 'down'
                    
                img_flipped = cv2.flip(img, flip_code)
                if flip_code == 0:
                    if flip_side == 'up':
                        stitched_img = cv2.vconcat([img_flipped, img])
                    else:
                        stitched_img = cv2.vconcat([img, img_flipped])
                else:
                    if flip_side == 'left':
                        stitched_img = cv2.hconcat([img_flipped, img])
                    else:
                        stitched_img = cv2.hconcat([img, img_flipped])

                cv2.imwrite(pattern_img.format(num = i), stitched_img)
        # Rotating the pattern images to make sure all the grainlines are the same for all patterns (Where applicable).
        ptrn_img = cv2.imread(pattern_img.format(num = i))
        blank = np.zeros((ptrn_img.shape[0] * 3,ptrn_img.shape[1] * 3, 3), dtype=np.uint8)
        blank[:] = 255
        blank[ptrn_img.shape[0]:ptrn_img.shape[0]*2, ptrn_img.shape[1]:ptrn_img.shape[1]*2] = ptrn_img
        cv2.imwrite(pattern_img.format(num = i), blank)
        ptrn_img = cv2.imread(pattern_img.format(num = i))
        ptrn_img = imutils.rotate_bound(ptrn_img, angle = rot_ang[i])
        cv2.imwrite(pattern_img.format(num = i), ptrn_img)        
        kernel = np.ones((7, 7), np.uint8)
        ptrn_img = cv2.erode(ptrn_img, kernel, iterations=1)
        cv2.imwrite(pattern_img.format(num = i), ptrn_img)
        ptrn_img = cv2.imread(pattern_img.format(num = i))
        ptrn_img = cv2.resize(ptrn_img,(0, 0),fx=resize_x, fy=resize_y, interpolation = cv2.INTER_LINEAR)
        cv2.imwrite(pattern_img.format(num = i), ptrn_img)



def gen_array(ptrn_imgs, ptrn_num, inv):
    """
    returns an array for the desired pattern with the following values: \n
    X inside and on the pattern contour
    Y outside the pattern contour
    Args:
        ptrn_img - The desired pattern image to generate an array from
        inv - Whether to rotate the array by 180 or not
        
    Returns:
        2D array, int, origin (0,0) top left corner, positive Y axis is downwards, positive X axis is to the right.
    """    
    img0 = cv2.imread(ptrn_imgs.format(num=ptrn_num))
    img = img0.copy()
    cv2.imwrite('image_copy.png',img)
    cntr = find_pattern_contours('image_copy.png', True)
    cv2.drawContours(image=img, contours=cntr, contourIdx=-1, color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
    cv2.imwrite('image_copy.png',img)
    cntr_np = np.asarray(cntr, dtype = object)
    max_val = cntr_np.max(axis=1, keepdims=False)
    min_val = cntr_np.min(axis=1, keepdims=False)
    x_max = int(max_val[0][0][0])
    y_max = int(max_val[0][0][1])
    x_min = int(min_val[0][0][0])
    y_min = int(min_val[0][0][1])
    blank = np.zeros(((y_max - y_min+1),(x_max-x_min+1), 3), dtype=np.uint8)        
    blank[:] = 255
    blank[0:y_max-y_min + 1, 0:x_max-x_min + 1] = img[y_min:y_max+1, x_min:x_max+1]
    for i in range((cntr[0].shape[0])):
        cntr[0][i][0][0] -= x_min
        cntr[0][i][0][1] -= y_min
    epsilon = 0.001
    aprox_cnt = cv2.approxPolyDP(cntr[0], epsilon, True)
    shape = (blank.shape[1],blank.shape[0])
    arr = np.zeros(shape, dtype = np.int16)
    #Determine whether each pixel is in or outside the contour and give the relevant value.
    for i in range (arr.shape[0]):
        for j in range (arr.shape[1]):
            ptInCntr = cv2.pointPolygonTest(aprox_cnt, (i,j), False)
            if (ptInCntr >= 0): #Inside or on contour
                arr.itemset((i,j), 0)
            else:   #Outisde contour
                arr.itemset((i,j), 1)
    if inv:
        arr = np.rot90(arr, 2)
    ## For debugging
    # plt.imshow(arr.T, interpolation='none')
    # plt.waitforbuttonpress()   
    # arr = np.rot90(arr, 2)
    # plt.imshow(arr.T, interpolation='none')
    # plt.waitforbuttonpress()   
    return arr.T

def init_main_arr(Fabric_width, num_of_ptrns, ptrn_imgs):
    """
    Returns an initialized main fabric array. \n
    leftmost column values are 1, rightmost column values are 2 \n
    The values increase linearly based on column #. \n
    The # of columns are a conservative estimate based on sum of pattern length. \n
    Args:
        Fabric_width - Fabric width in mm (pixels) \n
        num_of_ptrns - Number of patterns
        ptrn_imgs - the pattern image format to save the images, e.g 'pattern_{num}.png'.
        
    Returns:
        2D array, int, origin (0,0) top left corner, positive Y axis is downwards, positive X axis is to the right.
    """    
    len = 0
    for i in range(num_of_ptrns):
        arr = gen_array(ptrn_imgs, i, False)
        len += arr.shape[0]
    shape = (Fabric_width, len)
    main_array = np.zeros(shape)
    for i in range(Fabric_width):
        for j in range(len):
            main_array[i,j] = 2 + 0.5*i/Fabric_width - math.sqrt((j+1)/len)
    plt.imshow(main_array, interpolation='none')
    plt.waitforbuttonpress() 
    return main_array


def opt_place(main_array, num_of_ptrns, ptrn_imgs):
    """
    ???
    Args:
        main_array - Initialized main fabric array \n
        num_of_ptrns - Number of patterns \n
        ptrn_imgs - the pattern image format to save the images, e.g 'pattern_{num}.png'. \n
        
    Returns:
        void
    """   
    init_main_arr_sum = main_array.sum()
    min = 1
    index_min = []
    for i in range(num_of_ptrns):
        min = 1
        for j in range(num_of_ptrns):
            if j in index_min:
                continue
            main_arr_copy = main_array.copy()
            arr = gen_array(ptrn_imgs, j, False)
            x = random.randint(0, main_array.shape[0] - arr.shape[0])
            y = random.randint(0, main_array.shape[1] - arr.shape[1])        
            pos = [x,y]
            opts = {'disp': False, 'maxiter': 60, 'fatol': 1e-8}
            ## Maniuplate x and y independantly:
            # resx = optimize.minimize(cost_func, x, args=(main_array, init_main_arr_sum, arr, 1, y), method='Nelder-Mead', options=opts)
            # x = int(resx.x)
            # resy = optimize.minimize(cost_func, y, args=(main_array, init_main_arr_sum, arr, 0, x), method='Nelder-Mead', options=opts)
            # y = int(resy.x)            
            # print(resy.fun)
            # if min > resy.fun:
            #     min = resy.fun
            #     x_min = int(resx.x)
            #     y_min = int(resy.x)
            #     arr_min = arr.copy()
            #     index_min_val = j
            ## Maniuplate x and y simultaneously:
            res = optimize.minimize(cost_func, pos, args=(main_array, init_main_arr_sum, arr, 2, 0), method='Nelder-Mead', options=opts)
            cost = res.fun
            x = int(res.x[0])
            y = int(res.x[1])            
            print(cost)
            if min > res.fun:
                min = res.fun
                x_min = int(res.x[0])
                y_min = int(res.x[1])
                arr_min = arr.copy()
                index_min_val = j
            main_arr_copy[x:x+arr.shape[0], y:y+arr.shape[1]] = arr
            plt.imshow(main_arr_copy, interpolation='none')
            plt.waitforbuttonpress() 
            # plt.imshow(main_arr_copy, interpolation='none')
            # plt.waitforbuttonpress() 
        index_min.append(index_min_val)
        main_array[x_min:x_min+arr_min.shape[0], y_min:y_min+arr_min.shape[1]] = arr_min
        plt.imshow(main_array, interpolation='none')
        plt.waitforbuttonpress() 

    # TODO: 
    # 1. Add the same outer for loop
    # 2. Prevent pattern collisions
        # Consider combining existing patterns to 1 contour and checking if random point is inside contour or not.


def cost_func(pos1, main_array, init_main_arr_sum, arr, x_flag, pos2):
    if (x_flag == 1):
        x_pos = int(pos1)
        y_pos = int(pos2)
        if x_pos < 0:
            cost = math.sqrt(1 - x_pos)
            return cost
        if x_pos > (main_array.shape[0] - arr.shape[0]):
            cost = math.sqrt(x_pos -(main_array.shape[0] - arr.shape[0]))
            return cost
    elif (x_flag == 0):
        x_pos = int(pos2)
        y_pos = int(pos1)
        if y_pos < 0:
            cost = math.sqrt(1 - y_pos)
            return cost
        if y_pos > (main_array.shape[1] - arr.shape[1]):
            cost = math.sqrt(y_pos -(main_array.shape[1] - arr.shape[1]))
            return cost
    else: # x_flag = 2, Manipulate both x and y
        x_pos = int(pos1[0])
        y_pos = int(pos1[1])
        if x_pos < 0:
            cost = math.sqrt(1 - x_pos)
            return cost
        if x_pos > (main_array.shape[0] - arr.shape[0]):
            cost = math.sqrt(x_pos -(main_array.shape[0] - arr.shape[0]))
            return cost
        if y_pos < 0:
            cost = math.sqrt(1 - y_pos)
            return cost
        if y_pos > (main_array.shape[1] - arr.shape[1]):
            cost = math.sqrt(y_pos -(main_array.shape[1] - arr.shape[1]))
            return cost

    area_replaced = arr.size / main_array.size
    main_arr_copy = main_array.copy()
    main_arr_copy[x_pos:x_pos+arr.shape[0], y_pos:y_pos+arr.shape[1]] = arr
    cost = main_arr_copy.sum() * area_replaced / init_main_arr_sum;
    print(cost)
    return cost

