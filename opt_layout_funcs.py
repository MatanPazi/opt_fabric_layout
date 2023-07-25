# See relevant sources in function description.
#!/usr/bin/env python3
import sys
import pikepdf
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
import fitz



# Global params
A0_Width  = 841
A0_Height = 1189    
min_rot_ang = 10




# Extract relevant PDF layers
# Source: https://gist.github.com/jangxx/bd9256009b6698f1550fb7034003f877.
# Made relevant changes.
def pdfLayers(pdf_name, pdf_out, desired_layers):
    """
    Extract relevant PDF pages and layers and saves them as individual pdf files   \n
    Args:
        pdf_name - the main pdf file, e.g: 'main_pdf.pdf' \n
        pdf_out - format to save the pdf files, e.g: 'Out_{num}.pdf' \n
        desired_layers - layers to extract from the main pdf file \n
 
    Returns:
        number of pages in inputted pdf file.
    """
    # check if we even have some OCGs
    pdf = pikepdf.open(pdf_name)

    try:
        layers = pdf.Root.OCProperties.OCGs
    except (AttributeError, KeyError):
        print("Unable to locate layers in PDF.")
        sys.exit(1)

    # num_of_layers = len(layers)
    page_count = len(pdf.pages)
    pdf.close()

    # (hopefully) all pdf operators which "display" anything. everything else is styling, which we need to preserve
    hidden_operators = ["S", "s", "f" "F", "f*", "B", "B*", "b", "b*", "n", "Do", "sh", "Tj", "TJ", "m", "l", "c", "v", "y", "h", "re"]

    for i in range(page_count):
        end_reached = False
        extracted_groups = []
        cur_layer = 0

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
                    pdf.save(pdf_out.format(page_num = i, layer_num=cur_layer))                
                    
                cur_layer += 1
    return page_count

def pdf2image(desired_layers, pdf_out, img_out_init, img_out, page_count):
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
    # os.makedirs(output_folder, exist_ok=True)
    
    for k in range(page_count):
        # Store Pdf with convert_from_path function
        for j in range(len(desired_layers)):
            pdf_path = path + '/' + pdf_out.format(page_num = k, layer_num=desired_layers[j])

            # Open the PDF file
            pdf_file = fitz.open(pdf_path)

            # Iterate over each page of the PDF
            for i, page in enumerate(pdf_file):
                # Render the page as a pixmap
                pix = page.get_pixmap(dpi = 200)

                # Save the pixmap as PNG image
                pix.save(path + '/' + img_out_init.format(page_num = k, layer_num=desired_layers[j]), 'PNG')
    
    for i in range(len(desired_layers)):
        for k in range(page_count):        
            if k == 0:
                image = cv2.imread(img_out_init.format(page_num = k, layer_num=desired_layers[i]))
            else:
                img_temp = cv2.imread(img_out_init.format(page_num = k, layer_num=desired_layers[i]))
                image = cv2.vconcat([image, img_temp])
        cv2.imwrite(img_out.format(num = desired_layers[i]),image)



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
    elif color == 'red':
        cv2.line(img, pt0, pt1, (0, 0, 255), 5)
        cv2.line(img, pt1, pt2, (0, 0, 255), 5)
        cv2.line(img, pt2, pt3, (0, 0, 255), 5)
        cv2.line(img, pt3, pt0, (0, 0, 255), 5)
    else:
        cv2.line(img, pt0, pt1, (255, 0, 255), 5)
        cv2.line(img, pt1, pt2, (255, 0, 255), 5)
        cv2.line(img, pt2, pt3, (255, 0, 255), 5)
        cv2.line(img, pt3, pt0, (255, 0, 255), 5)

# Source: Roald's response in https://stackoverflow.com/questions/11627362/how-to-straighten-a-rotated-rectangle-area-of-an-image-using-opencv-in-python/48553593#48553593
# Made slight changes
def crop_image(cnt, image, type, ptrn_num, ptrn_imgs, ang_in, x_in, y_in, shape_in):
    if type == 'new_xy':
        center = (x_in, y_in)
        # shape = (image.shape[0], image.shape[1])  # Switching shape 1 and 0 since new_xy type assumes a 90 deg rotation.
        shape = (shape_in[1], shape_in[0])
    else:
        rect = cv2.minAreaRect(cnt)        
        shape = (image.shape[1], image.shape[0])  # cv2.warpAffine expects shape in (length, height)
        center, size, theta = rect
        width, height = tuple(map(int, size))
        center = tuple(map(int, center))    
        ## For debugging
        # image_copy = image.copy()
        # x = rect[0][0]
        # y = rect[0][1]
        # w = rect[1][0]
        # h = rect[1][1]
        # theta = rect[2]
        # draw_angled_rec(x, y, w, h, theta, image_copy, 'pink')
        # cv2.imwrite('img_compy.png',image_copy)

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
            if type == 'ang_rtrn' and width > height: # We want the rot ang required to have a horizontal grainline
                alpha += 90
            else:                                     # Otherwise, for cropping, we always need to switch width and height so height is on y axis.
                width, height = height, width   
            
    if type == 'ang_rtrn':
        # if alpha < min_rot_ang:
        #     alpha = 0
        # elif alpha > (90-min_rot_ang):
        #     if (abs(alpha - 90) < min_rot_ang and (alpha != 90)):
        #         alpha = 90
        return alpha
    elif type == 'new_xy':
        alpha = ang_in
    
    x_old = int(center[0])
    y_old = int(center[1])
    # x_old = int(center[0])
    # y_old = int(center[1])
    distance = math.sqrt(x_old**2+y_old**2)
    
    if type != 'new_xy':
        image = imutils.rotate_bound(image, angle = alpha)
        cv2.imwrite('img_test.png',image)

    alpha_rad = math.radians(alpha)

    y_temp = y_old*math.cos(alpha_rad) + x_old*math.sin(alpha_rad)
    x_offset = math.sin(alpha_rad) * shape[1]
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
    elif type == 'new_xy':
        return x_new, y_new

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
def find_pattern_contours(image, type):
    """
    Finds the pattern contours  \n
    Args:
        image - pattern contour image format to read \n        
        type - determines application need for this function.
                0 - initial pattern contour extraction
                1 - pattern contour extraction after resizing - different kernel size
                2 - pattern contour extraction before resizing - not saving patterns_overview img        
    Returns:
        The detected pattern contours
    """
    min_cnt_area = 400
    min_dist = 20
    epsilon = 1
    counter = 0
    img = cv2.imread(image)
        
    if type == 1:
        kernel_size = 3
    else:
        kernel_size = 7

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # The first parameter is the original image,
    # kernel is the matrix with which the image is
    # convolved and third parameter is the number
    # of iterations, which will determine how much
    # you want to erode/dilate a given image.
    img = cv2.erode(img, kernel, iterations=7)
    cv2.imwrite('img_test.png', img)
    img = cv2.dilate(img, kernel, iterations=2)
    cv2.imwrite('img_test.png', img)
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
        
        heir = hierarchy[0][counter][3]     # [next, previous, first child, parent]. 
                                            # See source: https://stackoverflow.com/questions/11782147/python-opencv-contour-tree-hierarchy-structure                        
        if heir == 0:                       # If heir is 0, means it's the outer most contour, which is what I'm interested in.            
            good_contours.append(cnt)
            aprox_main_cnt = cv2.approxPolyDP(cnt, epsilon, True)

        elif type == 1:
            append = 0
            aprox_cnt = cv2.approxPolyDP(cnt, epsilon, True)
            newArea = cv2.contourArea(cnt)

            if newArea > min_cnt_area:
                append = 1
                for i in range(len(aprox_main_cnt)):
                    point_x = aprox_main_cnt.item(2*i)
                    point_y = aprox_main_cnt.item(2*i+1)
                    dist = abs(cv2.pointPolygonTest(aprox_cnt, (point_x,point_y), True))
                    if dist < min_dist:
                        append = 0                        
                        break
            if append:
                good_contours.append(cnt)
        counter += 1
    ## For debugging
    image_copy = img.copy()
    cv2.drawContours(image=image_copy, contours=good_contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    if type == 0:                
        cv2.imwrite('patterns_overview.png',image_copy) 
    elif type == 1:
        cv2.imwrite('image_copy.png',image_copy) 
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
    # Need to blurr direction layer image due to intersecting lines on grainline arrows, causing contour detection to assume the arrow is divided.    
    img_blurred = img.copy()
    cv2.imwrite('image_copy.png',img_blurred)
    kernel_size = 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    img_blurred = cv2.erode(img_blurred, kernel, iterations=2)
    img_blurred = cv2.dilate(img_blurred, kernel, iterations=2)
    cv2.imwrite('image_copy.png',img_blurred)
    ptrn_cnt_counter = 0
    potential_contours = []
    potential_contours_ptrn_index = []
    ptrn_cntrs_new = []
    dir_ptrn_flag = 0
    for ptrn_cnt in ptrn_cntrs:
        img_tmp = img_blurred.copy()
        ## For debugging
        # img_debug = img.copy()
        # cv2.drawContours(image=img_debug, contours=ptrn_cnt, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        # cv2.imwrite('img_debug.png',img_debug) 
        img_cropped = crop_image(ptrn_cnt, img_tmp, 'pattern', 0, 0, 0, 0, 0, 0)    
        img_gray = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)
        # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
        contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)            
        slender_rat = 3
        min_width = 10
        max_width = 120
        min_len = 120
        first = 1
        image_copy = img_cropped.copy()
        cv2.imwrite('image_copy.png',image_copy) 
        # cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        # Used to add text on pattern
        rect_ptrn = cv2.minAreaRect(ptrn_cnt)
        x_ptrn = int(rect_ptrn[0][0])
        y_ptrn = int(rect_ptrn[0][1])

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
                        # Adding text on pattern image
                        img_ptrn = cv2.imread('patterns_overview.png')
                        image_tmp = img_ptrn.copy()
                        image_tmp = cv2.putText(img=image_tmp, text=str(ptrn_cnt_counter), org=(x_ptrn, y_ptrn), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=20, color=(0,0,0), thickness=70)
                        cv2.imwrite('patterns_overview.png',image_tmp)
                        potential_contours.append(cnt)
                        potential_contours_ptrn_index.append(ptrn_cnt_counter)
                elif h/w > slender_rat:
                    if (h < min_len or w < min_width or w > max_width):
                        continue
                    else:
                        dir_ptrn_flag = 1
                        img_cropped = draw_angled_rec(x, y, w, h, theta, image_copy, 'red')
                        cv2.imwrite('image_copy.png',image_copy) 
                        # Adding text on pattern image
                        img_ptrn = cv2.imread('patterns_overview.png')
                        image_tmp = img_ptrn.copy()
                        image_tmp = cv2.putText(img=image_tmp, text=str(ptrn_cnt_counter), org=(x_ptrn, y_ptrn), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=20, color=(0,0,0), thickness=70)
                        cv2.imwrite('patterns_overview.png',image_tmp)
                        potential_contours.append(cnt)
                        potential_contours_ptrn_index.append(ptrn_cnt_counter)
        
        # No direction contour found
        if dir_ptrn_flag == 0:
            continue

        ptrn_cntrs_new.append(ptrn_cnt)
        dir_ptrn_flag = 0            
        ptrn_cnt_counter += 1

    return potential_contours, potential_contours_ptrn_index, ptrn_cntrs_new


def save_patterns(ptrn_image, pattern_contours, dir_cnt, dir_ptrn_cnt, pattern_img):
    """
    Crops and saves the pattern contours images before being folded \n
    Args:
        ptrn_image - Main pattern image format to open
        pattern_contours - the pattern contours
        dir_cnt - the direction contours
        dir_ptrn_cnt - the direction contour pattern indices
        ptrn_imgs - the pattern image format to save the images, e.g 'pattern_{num}.png'.
        
    Returns:
        rot_ang_data - A list of the rotation angle data needed for each pattern to get the first (Arbitrary) direction contour horizontal.
    """
    img0 = cv2.imread(ptrn_image)
    cv2.imwrite('img_test.png',img0)
    rot_ang_data = []
    for i in range(len(pattern_contours)):
        img1 = crop_image(pattern_contours[i], img0, 'pattern', 0, 0, 0, 0, 0, 0)
        cv2.imwrite('img_test.png',img1)
        cnt = dir_cnt[dir_ptrn_cnt.index(i)]   #Find the first relevent direction contour
        ang = crop_image(cnt, img1, 'ang_rtrn', i, pattern_img, 0, 0, 0, 0)
        shape = img1.shape[0:2]
        rot_ang_data.append((ang, shape))

        cv2.imwrite(pattern_img.format(num = i), img1)
        ptrn_img = cv2.imread(pattern_img.format(num = i))
        blank = np.zeros((ptrn_img.shape[0] * 3,ptrn_img.shape[1] * 3, 3), dtype=np.uint8)
        blank[:] = 255
        blank[ptrn_img.shape[0]:ptrn_img.shape[0]*2, ptrn_img.shape[1]:ptrn_img.shape[1]*2] = ptrn_img
        cv2.imwrite(pattern_img.format(num = i), blank)  
        ptrn_img = cv2.imread(pattern_img.format(num = i))
        ptrn_img = imutils.rotate_bound(ptrn_img, angle = ang)      # Rotating here and not in fold_patterns() because we want to flip image on straight edge.
        cv2.imwrite(pattern_img.format(num = i), ptrn_img)
        img_temp = ptrn_img.copy()
        cv2.imwrite('img_temp.png',img_temp)
        ptrn_cntr = find_pattern_contours('img_temp.png', 2)
        # computing the bounding rectangle of the contour
        x, y, w, h = cv2.boundingRect(ptrn_cntr[0])
        # ptrn_img = cv2.rectangle(ptrn_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imwrite('img_test.png',ptrn_img)
        img_cropped_temp = ptrn_img[y : y+h, x: x+w]
        cv2.imwrite('img_test.png',img_cropped_temp)
        
        
        
        img = img_cropped_temp.copy()        
        kernel_size = 3
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        img = cv2.erode(img, kernel, iterations=1)
        cv2.imwrite('img_test.png',img)
        img = cv2.dilate(img, kernel, iterations=1)
        cv2.imwrite('img_test.png',img)
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

        img_cropped = img_cropped_temp[y_min : y_max, x_min: x_max]                   
        cv2.imwrite(pattern_img.format(num=i),img_cropped) 
    return rot_ang_data
    



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
    # All possible values of copies
    cut_num_txt = ['cut two', 'cut 2', 'cut2', 'cut four', 'cut 4', 'cut4']
    cut_2 = cut_num_txt[0:3]
    cut_4 = cut_num_txt[3:6]

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
        cropped_img = crop_image(ptrn, img0, 'pattern', 0, 0, 0, 0, 0, 0)    
        for cnt in dir_cnt_np[np.where(dir_ptrn_cnt_np == ptrn_counter)]:
            dir_cropped_img = crop_image(cnt, cropped_img, 'direction', 0, 0, 0, 0, 0, 0)
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
            # if # of copies indicated:
            for string in cut_2:
                if string in text:
                    copies = 2
                    continue
            for string in cut_4:
                if string in text:
                    copies = 4
                    continue
            if 'lining' in text:
                lining = 1
                if 'main fabric' not in text:
                    main_fabric = 0
        copies_list.append(copies)
        lining_list.append(lining)
        main_fabric_list.append(main_fabric)
        ptrn_counter += 1
        
    return copies_list, lining_list, main_fabric_list, fold_list, dir_cnt


def fold_patterns(fold_list, pattern_img, size, page_count, rot_ang):
    """
    Folds the pattern images based on the given fold list \n
    copies images to larger blank images for easier later manipulation \n
    and saves back to pattern_img format \n
    Args:
        fold_list - list of fold direction contours per pattern \n
        pattern_img - pattern images format \n
        rot_ang - rotation angle data needed to make sure the grainlines are horizontal \n
        size - size of the original images to know how much to resize to fit to an A0 size
        
    Returns:
        void
    """
    # Dependent on original image orientation:
    if size[0] > size[1]:       # Portrait
        resize_y = page_count * A0_Height / size[0]
        resize_x = A0_Width / size[1]
    else:                       # Landscape
        resize_y = page_count * A0_Width / size[0]
        resize_x = A0_Height / size[1]

    for i in range(len(fold_list)):
        if fold_list[i] != 0:            
            flip_code = -1
            invert = 0
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
                angle = rect[2] + rot_ang[i][0]
                shape = rot_ang[i][1]

                if rot_ang[i][0] > (180 - min_rot_ang):
                    invert = 1
                
                elif rot_ang[i][0] > (90 - min_rot_ang) and rot_ang[i][0] < (90 + min_rot_ang):
                    x, y = crop_image(0, ptrn_img, 'new_xy', 0, 0, 90, x, y, shape)                    
                    if rect[2] > (90 - min_rot_ang):
                        w,h = h,w
                elif rot_ang[i][0] > (90 + min_rot_ang) and rot_ang[i][0] < (180 - min_rot_ang):
                    x, y = crop_image(0, 0, 'new_xy', 0, 0, 90, x, y, shape)
                    shape_temp = (shape[1], shape[0])
                    x, y = crop_image(0, 0, 'new_xy', 0, 0, (rot_ang[i][0] - 90), x, y, shape_temp)

                if (angle < min_rot_ang) or (angle > (180 - min_rot_ang)) or invert:
                    if w > h:
                        if flip_code == 0:      # Already folded along that side.
                            break
                        else:
                            flip_code = 0
                        if y < (shape[0] // 2):
                            flip_side = 'up'
                        else:
                            flip_side = 'down'
                    else:
                        if flip_code == 1:
                            break
                        else:
                            flip_code = 1
                        if x < (shape[1] // 2):
                            flip_side = 'left'
                        else:
                            flip_side = 'right'
                else: #angle > (90 - min_rot_ang)
                    if w > h:
                        if flip_code == 1:
                            break
                        else:
                            flip_code = 1
                        if x < (shape[1] // 2):
                            flip_side = 'left'
                        else:
                            flip_side = 'right'
                    else:
                        if flip_code == 0:
                            break
                        else:
                            flip_code = 0
                        if y < (shape[0] // 2):
                            flip_side = 'up'
                        else:
                            flip_side = 'down'
                    
                img_flipped = cv2.flip(img, flip_code)
                if flip_code == 0:
                    if ((flip_side == 'up') and (invert == 0)) or ((flip_side == 'down') and (invert == 1)):
                        stitched_img = cv2.vconcat([img_flipped, img])
                    else:
                        stitched_img = cv2.vconcat([img, img_flipped])
                else:
                    if ((flip_side == 'left') and (invert == 0)) or ((flip_side == 'right') and (invert == 1)):
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
        kernel = np.ones((7, 7), np.uint8)
        ptrn_img = cv2.erode(ptrn_img, kernel, iterations=1)
        cv2.imwrite(pattern_img.format(num = i), ptrn_img)
        ptrn_img = cv2.imread(pattern_img.format(num = i))
        ptrn_img = cv2.resize(ptrn_img,(0, 0),fx=resize_x, fy=resize_y, interpolation = cv2.INTER_LINEAR)
        cv2.imwrite(pattern_img.format(num = i), ptrn_img)



def gen_array(ptrn_imgs, ptrn_num, inv, config):
    """
    returns an array for the desired pattern with the following values: \n
    X inside and on the pattern contour
    Y outside the pattern contour
    Args:
        ptrn_img - The desired pattern image to generate an array from
        ptrn_num - Number of patterns
        inv - Whether to rotate the array by 180 or not
        config: 0 - Bottom left optimization
                1 - Array to replace in the main array.
                2 - Used to determine which main array indices to sum
        
    Returns:
        2D array, int, origin (0,0) top left corner, positive Y axis is downwards, positive X axis is to the right.
    """    
    img0 = cv2.imread(ptrn_imgs.format(num=ptrn_num))
    img = img0.copy()
    if inv:
        img = cv2.flip(img, 1)      # Flip horizontally
    cv2.imwrite('image_copy.png',img)

    cntr = find_pattern_contours('image_copy.png', 1)
    cv2.drawContours(image=img, contours=cntr, contourIdx=-1, color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
    cv2.imwrite('image_copy.png',img)
    cntr_np_temp = []
    for i in range(len(cntr)):
        cntr_np_temp.append(cntr[i])
    
    # Using list comprehension. Converting 3D to 2D
    cntr_np = [elem for twod in cntr_np_temp for elem in twod]
    cntr_np = np.asarray(cntr_np, dtype = np.int32)

    max_val = cntr_np.max(axis=0, keepdims=False)
    x_max = int(max_val[0][0])
    y_max = int(max_val[0][1])

    min_val = cntr_np.min(axis=0, keepdims=False)     
    x_min = int(min_val[0][0])
    y_min = int(min_val[0][1])

    blank = np.zeros(((y_max - y_min+1),(x_max-x_min+1), 3), dtype=np.uint8)        
    blank[:] = 255
    blank[0:y_max-y_min + 1, 0:x_max-x_min + 1] = img[y_min:y_max+1, x_min:x_max+1]

    for i in range(len(cntr)):
        for j in range(cntr[i].shape[0]):
            cntr[i][j][0][0] -= x_min
            cntr[i][j][0][1] -= y_min
    epsilon = 1

    aprox_cntrs = []
    for i in range(len(cntr)):
        aprox_cnt_temp = cv2.approxPolyDP(cntr[i], epsilon, True)
        aprox_cntrs.append(aprox_cnt_temp)

    # Find contour center
    M = cv2.moments(cntr[0])
    cy = int(M['m01']/M['m00'])
    cx = int(M['m10']/M['m00'])
    ## For debugging 
    # for i in range(len(aprox_cnt)):
    #     x_cord = aprox_cnt.item(2*i)
    #     y_cord = aprox_cnt.item(2*i+1)
    #     loc = (x_cord,y_cord)
    #     cv2.circle(blank,(loc), 5, (0,0,255), -1)
    # cv2.imwrite('image_copy.png',blank)
    shape = (blank.shape[1],blank.shape[0])
    arr = np.zeros(shape)
    max_dist = 0
    first_cntr = 1

    #Determine whether each pixel is in or outside the contour and give the relevant value.
    if config == 0:
        for aprox_cnt in aprox_cntrs:
            for i in range (arr.shape[0]):
                for j in range (arr.shape[1]):                        
                    dist = cv2.pointPolygonTest(aprox_cnt, (i,j), True)
                    if first_cntr:                        
                        if (dist >= 0): #Inside or on contour
                            arr.itemset((i,j), 0.0)
                        else:
                            arr.itemset((i,j), 1.0)
                    else:
                        if (dist >= 0): #Inside or on inner contour
                            arr.itemset((i,j), 1.0)
            first_cntr = 0
    elif config == 1:
        for aprox_cnt in aprox_cntrs:
            for i in range (arr.shape[0]):
                for j in range (arr.shape[1]): 
                    dist = cv2.pointPolygonTest(aprox_cnt, (i,j), True)
                    if max_dist < dist:
                        max_dist = dist
        for aprox_cnt in aprox_cntrs:
            for i in range (arr.shape[0]):
                for j in range (arr.shape[1]): 
                    dist = cv2.pointPolygonTest(aprox_cnt, (i,j), True)
                    if first_cntr:
                        if (dist > 0): #Inside contour
                            arr.itemset((i,j), ((-dist / max_dist) - 50))
                        elif (dist == 0): #on contour
                            arr.itemset((i,j), 1.0)
                        else:   #Outisde contour
                            arr.itemset((i,j), 1.0)
                    else:
                        if (dist > 0): #Inside or on inner contour
                            arr.itemset((i,j), 1.0)               
            first_cntr = 0
    else:
        for aprox_cnt in aprox_cntrs:
            for i in range (arr.shape[0]):
                for j in range (arr.shape[1]):        
                    dist = cv2.pointPolygonTest(aprox_cnt, (i,j), True)
                    if first_cntr:
                        if (dist >= 0): #Inside or on contour
                            arr.itemset((i,j), 1.0)
                        else:
                            arr.itemset((i,j), 0.0)
                    else:
                        if (dist >= 0): #Inside or on inner contour
                            arr.itemset((i,j), 0.0)                        
            first_cntr = 0
    
    ## For Debugging
    # plt.imshow(arr, interpolation='none')
    # plt.waitforbuttonpress()

    if config == 0:
        return arr.T
    else:
        return arr.T, aprox_cntrs, cy, cx, max_dist

def init_main_arr(Fabric_width, num_of_ptrns, ptrn_imgs, config, aprox_cntrs, main_array, ptrn_list):
    """
    Returns an initialized main fabric array. \n
    leftmost column values are 1, rightmost column values are 2 \n
    The values increase linearly based on column #. \n
    The # of columns are a conservative estimate based on sum of pattern length. \n
    Args:
        Fabric_width - Fabric width in mm (pixels) \n
        num_of_ptrns - Number of patterns
        ptrn_imgs - the pattern image format to save the images, e.g 'pattern_{num}.png'.
        config - 0 means bottom left optimization, 1 means NFP (No Fit Polygon) optimization init placement, 2 means subsequent placements.
        
    Returns:
        2D array, int, origin (0,0) top left corner, positive Y axis is downwards, positive X axis is to the right.
    """    

    len = 0
    for i in range(num_of_ptrns):
        if i not in ptrn_list:
            continue
        arr = gen_array(ptrn_imgs, i, False, 0)
        len += arr.shape[1]
    shape = (Fabric_width, len)
    if config == 0:
        main_array = np.zeros(shape)
        for i in range(Fabric_width):
            for j in range(len):
                main_array[i,j] = 500*(2 + i/Fabric_width - 2*math.sqrt((j+1)/len))        
    else:
        max_dist = math.sqrt(Fabric_width**2 + len**2)
        if config == 1:    #First placement
            main_array = max_dist * np.ones(shape)
        first_cntr = 1
        for aprox_cnt in aprox_cntrs:
            for i in range(Fabric_width):
                for j in range(len):
                    if main_array[i,j] > 0: #Not inside another contour
                        dist = -1 * cv2.pointPolygonTest(aprox_cnt, (i,j), True) #Positive values for outisde the contour              
                        if first_cntr:                            
                            if dist > 0:    # Outside the current contour
                                temp = (max_dist - dist) / max_dist + (len-j)/len
                                if (temp > main_array[i,j]) or (config == 1):  # Min distance 
                                    main_array[i,j] = (max_dist - dist) / max_dist + (len-j)/len
                            else:
                                main_array[i,j] = 1
                        else:
                            if dist < 0:    # Inside the inner contour
                                temp = (max_dist - dist) / max_dist + (len-j)/len
                                if (temp > main_array[i,j]) or (config == 1):  # Min distance 
                                    main_array[i,j] = (max_dist - dist) / max_dist + (len-j)/len
            first_cntr = 0
         
    return main_array

def first_pattern_placement(main_array, num_of_ptrns, ptrn_imgs, ptrn_list):

    ## Locating first pattern in bottom left side (Choosing largest pattern for now) 
    max_area = 0    
    max_area_index = 0
    for i in range(num_of_ptrns):
        if i not in ptrn_list:
            continue
        arr = gen_array(ptrn_imgs, i, False, 0)
        if max_area < arr.size:
            max_area = arr.size
            max_area_arr = arr
            max_area_index = i    
    # opts = {'disp': False, 'maxiter': 50, 'fatol': 1e-10}
    # TODO: Look for min function value based on inverted or not
    cost_min = 1
    for invert in range(2):
        arr, _, _, _, _ = gen_array(ptrn_imgs, max_area_index, invert, 2)
        # TODO: Optimization isn't really needed at first placement. Can simply put this array at bottom left and choose the lowest cost (inv or not).
        main_arr_copy = main_array.copy()    
        y_pos = main_arr_copy.shape[0] - arr.shape[0]
        x_pos = 0
        main_arr_copy[y_pos:main_arr_copy.shape[0], x_pos:arr.shape[1]] = np.multiply(main_arr_copy[y_pos:main_arr_copy.shape[0], x_pos:arr.shape[1]], arr)
        init_sum = main_arr_copy[y_pos:main_arr_copy.shape[0], x_pos:arr.shape[1]].sum()
        cost = 1/init_sum
        if cost < cost_min:
            cost_min = cost
            inv = invert
    
    return y_pos, x_pos, max_area_index, inv

def opt_place(copies, ptrn_imgs, fabric_width, ptrn_list):
    """
    Run optimization
    Args:
        main_array - Initialized main fabric array \n
        num_of_ptrns - Number of patterns \n
        ptrn_imgs - the pattern image format to save the images, e.g 'pattern_{num}.png'. \n
        
    Returns:
        void
    """   
    num_of_ptrns = len(copies)
    num_of_copies = sum(copies)
    # First pattern placement
    main_array = init_main_arr(fabric_width, num_of_ptrns, ptrn_imgs, 0, 0, 0, ptrn_list)
    main_poly_ind = []
    main_poly_pts = []
    y,x,arr_index, inv = first_pattern_placement(main_array, num_of_ptrns, ptrn_imgs, ptrn_list)
    arr, aprox_cntrs, center_y, center_x, _ = gen_array(ptrn_imgs, arr_index, inv, 1)
    center_y += y
    center_x += x
    main_poly_ind.append(arr_index)

    for aprox_cnt in aprox_cntrs:
        for n in range(len(aprox_cnt)): # Changing to (y,x) from (x,y) format
            tempx = aprox_cnt[n][0][0]
            aprox_cnt[n][0][0] = aprox_cnt[n][0][1]
            aprox_cnt[n][0][1] = tempx

        for i in range(len(aprox_cnt)):
            aprox_cnt[i][0][0] += y
            aprox_cnt[i][0][1] += x

        main_poly_pts.append(aprox_cnt)
    
    # Preparing for subsequent pattern placements
    main_array = init_main_arr(fabric_width, num_of_ptrns, ptrn_imgs, 1, aprox_cntrs, 0, ptrn_list)
    main_array[y:y+arr.shape[0], x:x+arr.shape[1]] = np.multiply(main_array[y:y+arr.shape[0], x:x+arr.shape[1]], arr)
    ## For Debugging
    # plt.imshow(main_array, interpolation='none')
    # plt.waitforbuttonpress()
    # Adding pattern # to image.
    cv2.imwrite('opt_res.png',main_array)
    image = cv2.imread('opt_res.png', cv2.IMREAD_GRAYSCALE)
    image[y:y+arr.shape[0], x:x+arr.shape[1]] = abs(main_array[y:y+arr.shape[0], x:x+arr.shape[1]]) * 200
    image = cv2.putText(img=image, text=str(arr_index), org=(center_x, center_y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(255,255,0), thickness=3)
    cv2.imwrite('opt_res.png',image)

    opts = {'disp': False, 'maxiter': 50, 'fatol': 1e-9}
    for k in range(num_of_copies):
        print(k)
        main_array_init = main_array.copy()           
        min = 1
        index_min_val = 0
        ptrn_added = 0
        for i in range(num_of_ptrns):
            if ((i in main_poly_ind) and (main_poly_ind.count(i) == copies[i])) or (i not in ptrn_list):      # Taking number of copies into account.                
                continue
            ptrn_added = 1
            cost_min = 1
            for invert in range(2):
                arr, _, center_x, center_y, _ = gen_array(ptrn_imgs, i, invert, 2)            
                for p in range(len(main_poly_pts)):
                    for j in range(len(main_poly_pts[p])):
                        # main_array_copy = main_array.copy()
                        y = main_poly_pts[p][j][0][0] - center_y
                        x = main_poly_pts[p][j][0][1] - center_x                                            
                        # Prevent initializing outside of main_array
                        if y < 0:
                            y = 0
                        elif y > (main_array.shape[0] - arr.shape[0]):
                            y = main_array.shape[0] - arr.shape[0]
                        if x < 0:
                            x = 0
                        elif x > (main_array.shape[1] - arr.shape[1]):
                            x = main_array.shape[1] - arr.shape[1]

                        init_pos = [y,x]
                        ## Maniuplate x and y simultaneously:                
                        debug = 0
                        # if i == 1:
                        #     debug = 1
                        res = optimize.minimize(cost_func_NFP, init_pos, args=(main_array, arr, debug), method='Nelder-Mead', options=opts)
                        y = res.x[0]
                        x = res.x[1]
                        cost = res.fun
                        if cost_min > cost:
                            cost_min = cost
                            res_min = res
                            inv_temp = invert
                            center_y_temp = y + center_x
                            center_x_temp = x + center_y
                            ## For Debugging                            
                            # arr_min, _, _, _, _ = gen_array(ptrn_imgs, i, invert, 1)
                            # if x < 0 or y < 0:
                            #     continue
                            # if y+arr_min.shape[0] > main_array_copy.shape[0] or x+arr_min.shape[1] > main_array_copy.shape[1]:
                            #     continue
                            # main_array_copy[int(y):int(y)+arr_min.shape[0], int(x):int(x)+arr_min.shape[1]] = np.multiply(main_array_copy[int(y):int(y)+arr_min.shape[0], int(x):int(x)+arr_min.shape[1]], arr_min)                                                    
                            # plt.title(res_min.fun)
                            # plt.imshow(main_array_copy, interpolation='none')
                            # plt.waitforbuttonpress()

            if res_min.x[0] < 0:
                res_min.x[0] = 0
            if res_min.x[0] > (main_array.shape[0] - arr.shape[0]):
                res_min.x[0] = main_array.shape[0] - arr.shape[0]

            if res_min.x[1] < 0:
                res_min.x[1] = 0
            if res_min.x[1] > (main_array.shape[1] - arr.shape[1]):
                res_min.x[1] = main_array.shape[1] - arr.shape[1]

            y = int(res_min.x[0])            
            x = int(res_min.x[1])

            if min > res_min.fun:
                min = res_min.fun
                y_min = int(res_min.x[0])
                x_min = int(res_min.x[1])
                index_min_val = i   
                invert_min = inv_temp
                center_y_min = int(center_y_temp)
                center_x_min = int(center_x_temp)
        
        if (len(main_poly_ind) < num_of_copies) and (ptrn_added == 1):
            arr_min, aprox_cntrs_min, _, _, _ = gen_array(ptrn_imgs, index_min_val, invert_min, 1)
            main_poly_ind.append(index_min_val)
            for aprox_cnt_min in aprox_cntrs_min:
                for n in range(len(aprox_cnt_min)): # Changing to (y,x) from (x,y) format
                    tempx = aprox_cnt_min[n][0][0]
                    aprox_cnt_min[n][0][0] = aprox_cnt_min[n][0][1]
                    aprox_cnt_min[n][0][1] = tempx 
                
                # Need to handle offset and rotation(?)
                for m in range(len(aprox_cnt_min)):
                    aprox_cnt_min[m][0][0] += y_min
                    aprox_cnt_min[m][0][1] += x_min
                
                main_poly_pts.append(aprox_cnt_min)


            main_array = init_main_arr(fabric_width, num_of_ptrns, ptrn_imgs, 2, aprox_cntrs_min, main_array_init, ptrn_list)
            main_array[y_min:y_min+arr_min.shape[0], x_min:x_min+arr_min.shape[1]] = np.multiply(main_array[y_min:y_min+arr_min.shape[0], x_min:x_min+arr_min.shape[1]], arr_min)
            ## For Debugging
            # plt.imshow(main_array, interpolation='none')
            # plt.waitforbuttonpress()
            # Adding pattern # to image.
            image = cv2.imread('opt_res.png', cv2.IMREAD_GRAYSCALE)
            image[y_min:y_min+arr_min.shape[0], x_min:x_min+arr_min.shape[1]] = abs(main_array[y_min:y_min+arr_min.shape[0], x_min:x_min+arr_min.shape[1]]) * 200
            image = cv2.putText(img=image, text=str(index_min_val), org=(center_x_min, center_y_min), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(255,255,0), thickness=3)
            cv2.imwrite('opt_res.png',image)
             
    plt.imshow(main_array, interpolation='none')
    plt.waitforbuttonpress()



def cost_func_NFP(pos, main_array, arr, debug):
    y_pos = int(pos[0])
    x_pos = int(pos[1])
    main_arr_len = y_pos + arr.shape[0]
    main_arr_wid = x_pos + arr.shape[1]    
    cost = 0
    if y_pos < 0:
        cost += abs(y_pos)/10

    if y_pos > (main_array.shape[0] - arr.shape[0]):
        cost += (y_pos + arr.shape[0] - main_array.shape[0])/10

    if x_pos < 0:
        cost += abs(x_pos)/10

    if x_pos > (main_array.shape[1] - arr.shape[1]):
        cost += (x_pos + arr.shape[1] - main_array.shape[1])/10
    
    if cost != 0:
        if debug:
            plt.title(cost)
            plt.text(500,500,x_pos, fontsize=12)
            plt.text(600,600,y_pos, fontsize=12)
            plt.imshow(main_array, interpolation='none')
            plt.waitforbuttonpress()
        return cost


    main_arr_copy = main_array.copy()    
    main_arr_copy[y_pos:main_arr_len, x_pos:main_arr_wid] = np.multiply(main_arr_copy[y_pos:main_arr_len, x_pos:main_arr_wid], arr)
    init_sum = main_arr_copy[y_pos:main_arr_len, x_pos:main_arr_wid].sum()
    if init_sum == 0:
        cost = 1
    elif init_sum > 0:
        cost = 1/init_sum
    else:
        cost = -init_sum/1000
    
    if debug:
        main_arr_copy[y_pos:main_arr_len, x_pos:main_arr_wid] = arr
        plt.title(cost)
        # plt.text(x_pos,y_pos,x_pos, fontsize=12)
        plt.text(x_pos,y_pos,y_pos, fontsize=12)
        plt.imshow(main_arr_copy, interpolation='none')
        plt.waitforbuttonpress()

    return cost 






# Old cost function
# def cost_func(pos1, main_array, init_main_arr_sum, arr, x_flag, pos2):
#     norm_param = arr.size / main_array.size      # Normalization parameter
#     cost = 0
#     if (x_flag == 1):       # x manipulation only
#         x_pos = int(pos1)
#         y_pos = int(pos2)
#         if x_pos < 0:
#             cost = math.sqrt(1 - x_pos)
#             return cost
#         if x_pos > (main_array.shape[0] - arr.shape[0]):
#             cost = math.sqrt(x_pos -(main_array.shape[0] - arr.shape[0]))
#             return cost
#     elif (x_flag == 0):     # y manipulation only
#         x_pos = int(pos2)
#         y_pos = int(pos1)
#         if y_pos < 0:
#             cost = math.sqrt(1 - y_pos)
#             return cost
#         if y_pos > (main_array.shape[1] - arr.shape[1]):
#             cost = math.sqrt(y_pos -(main_array.shape[1] - arr.shape[1]))
#             return cost
#     else:                   # x_flag = 2, Manipulate both x and y
#         y_pos = int(pos1[0])
#         x_pos = int(pos1[1])
#         main_arr_len = y_pos+arr.shape[0]
#         main_arr_wid = x_pos+arr.shape[1]
#         arr_y_start = 0
#         arr_y_end = arr.shape[0]        
#         arr_x_start = 0
#         arr_x_end = arr.shape[1]
#         if y_pos < 0:
#             if abs(y_pos) > arr.shape[0]:
#                 cost += (abs(y_pos) - arr.shape[0]) * norm_param
#             else:
#                 arr_y_start = abs(y_pos)
#                 y_pos = 0
#         if y_pos > (main_array.shape[0] - arr.shape[0]):
#             if y_pos > main_array.shape[0]:
#                 cost += (y_pos - main_array.shape[0]) * norm_param
#             else:
#                 arr_y_end = main_array.shape[0] - y_pos
#                 main_arr_len = main_array.shape[0]  

#         if x_pos < 0:
#             if abs(x_pos) > arr.shape[1]:
#                 cost += (abs(x_pos) - arr.shape[1]) * norm_param
#             else:
#                 arr_x_start = abs(x_pos)
#                 x_pos = 0
#         if x_pos > (main_array.shape[1] - arr.shape[1]):
#             if x_pos > main_array.shape[0]:
#                 cost += (x_pos - main_array.shape[1]) * norm_param
#             else:
#                 arr_x_end = main_array.shape[1] - x_pos
#                 main_arr_wid = main_array.shape[1]            
    
#     if cost != 0:
#         return cost
    
#     main_arr_copy = main_array.copy()
#     main_arr_copy[y_pos:main_arr_len, x_pos:main_arr_wid] = np.multiply((main_arr_copy[y_pos:main_arr_len, x_pos:main_arr_wid]),arr[arr_y_start:arr_y_end, arr_x_start:arr_x_end])
#     cost = norm_param * main_arr_copy.sum() / init_main_arr_sum
#     # plt.imshow(main_arr_copy, interpolation='none')
#     # plt.waitforbuttonpress()
#     return cost