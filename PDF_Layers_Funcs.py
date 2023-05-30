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

# Global params
A0_Width  = 841
A0_Height = 1189    

# Extract relevant PDF layers
# Source: https://gist.github.com/jangxx/bd9256009b6698f1550fb7034003f877.
# Made relevant changes.
def pdfLayers(pdf_name, pdf_out, desired_layers):
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


# Save each pdf as an image
def pdf2image(desired_layers, pdf_out, img_out):

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
        new_height = height // 2
        new_width = width // 2
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
def find_pattern_contours(image):
    counter = 0
    img = cv2.imread(image)
        
    # Taking a matrix of size 7 as the kernel
    kernel_size = int(img.shape[0]*img.shape[1] * 0.0000002 + 0.5)
    if kernel_size < 1:
        kernel_size = 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # The first parameter is the original image,
    # kernel is the matrix with which image is
    # convolved and third parameter is the number
    # of iterations, which will determine how much
    # you want to erode/dilate a given image.
    img = cv2.erode(img, kernel, iterations=6)
    img = cv2.dilate(img, kernel, iterations=3)
    # img = cv2.erode(img, kernel, iterations=3)
    ## For debugging
    image_copy = img.copy()
    cv2.imwrite('image_copy.png',image_copy)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)

    # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
    contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)                                            
    good_contours = []
    ## Needed for old method. Using heir now.
    # shape = (len(contours), 1)              # len(contours) rows and 1 columns
    # x = np.zeros(shape).astype(int)
    # y = np.zeros(shape).astype(int)
    # w = np.zeros(shape).astype(int)
    # h = np.zeros(shape).astype(int)

    j = 0
    for cnt in contours:
        if counter == 0:                    # First contour encompasses entire image
            counter += 1
            continue
        ## Needed for old method. Using heir now.
        # x[j],y[j],w[j],h[j] = cv2.boundingRect(cnt)
        good_contours.append(cnt)
        heir = hierarchy[0][counter][3]     # [next, previous, first child, parent].
                                            # See source: https://stackoverflow.com/questions/11782147/python-opencv-contour-tree-hierarchy-structure
        if heir != 0:                       # If heir is 0, means it's the outer most contour, which is what I'm interested in.
            ## Needed for old method. Using heir now.
            # x = np.delete(x, j, 0)
            # y = np.delete(y, j, 0)
            # w = np.delete(w, j, 0)
            # h = np.delete(h, j, 0)
            good_contours.pop(j)
        else:             
            j += 1
            # cv2.drawContours(image=image_copy, contours=good_contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
            # cv2.imwrite('image_copy.png',image_copy)
        ## Needed for old method. Using heir now (See above).
        # for i in range(j+1): # Check if this contour is inside another contour
        #     if ((((x[j] + w[j]) < (x[i] + w[i])) and x[j] > x[i]) and
        #         (((y[j] + h[j]) < (y[i] + h[i])) and y[j] > y[i]) or
        #         (w[j] < min_pixels and h[j] < min_pixels)):
        #         x = np.delete(x, j, 0)
        #         y = np.delete(y, j, 0)
        #         w = np.delete(w, j, 0)
        #         h = np.delete(h, j, 0)
        #         good_contours.pop(j)
        #         break
        #     elif i == j:
        #         j += 1
        counter += 1
    ## For debugging
    # cv2.drawContours(image=image_copy, contours=good_contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    # cv2.imwrite('image_copy.png',image_copy)
    return good_contours 


# Find grainline contours
def find_potential_direction_contours(image, ptrn_cntrs):
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
    # TODO: crop image will need to be changed.
    # ATM it crops and rotates the pattern so that the minAreaRect is 0 deg.
    # OR, it crops and rotates the direction contours from the cropped pattern contours
    ### Need to add the following:
    ### When the 'direction' method is used, save the rotated image before cropping.
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
            kernel_size = 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        img = cv2.erode(img, kernel, iterations=2)
        img = cv2.dilate(img, kernel, iterations=2)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)        
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
            cnt_np = np.asarray(cnt)
            max_val = cnt_np.max(axis=0, keepdims=False)
            min_val = cnt_np.min(axis=0, keepdims=False)
            for j in range(len(cnt)):
                if max_val[0][0] > x_max:
                    x_max = int(max_val[0][0])
                if max_val[0][1] > y_max:
                    y_max = int(max_val[0][1])
                if min_val[0][0] < x_min:
                    x_min = int(min_val[0][0])
                if min_val[0][1] < y_min:
                    y_min = int(min_val[0][1])
        
        px_buffer = int(1.5 + img0.shape[1] / A0_Width)

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
    



def find_text(image, pattern_contours, dir_cnt, dir_ptrn_cnt, ptrn_imgs):
    if platform.system() == 'Windows':
        pytesseract.tesseract_cmd=r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    else:
        pytesseract.tesseract_cmd=r'/usr/bin/tesseract'    
    img0 = cv2.imread(image)
    ang_inc = 90      #Potential TODO: Rotate pattern contour according to grainling angle and then rotate by 90 degrees.
    copies_list = []
    lining_list = []
    main_fabric_list = []
    fold_list = []
    dir_cnt_np = np.array(dir_cnt, dtype=object)
    dir_ptrn_cnt_np = np.array(dir_ptrn_cnt, dtype=object)
    ptrn_counter = 0
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
            for i in range (int(360/ang_inc) - 1):
                img = imutils.rotate_bound(dir_cropped_img, angle = (i * ang_inc))      # rotate_bound rotation is clockwise for positive values.
                cv2.imwrite('img_test.png',img)
                text = (pytesseract.image_to_string(img)).lower()
                print(text[:-1])
                if 'fold' in text:
                    fold.append(cnt)                
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
            print(text[:-1])                                    #print the text line by line
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
    return copies_list, lining_list, main_fabric_list, fold_list


def fold_patterns(fold_list, pattern_img, rot_ang, size):

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
        ptrn_img = imutils.rotate_bound(ptrn_img, angle = rot_ang[i])
        ptrn_img = cv2.resize(ptrn_img,(0, 0),fx=resize_x, fy=resize_y, interpolation = cv2.INTER_LINEAR)
        cv2.imwrite(pattern_img.format(num = i), ptrn_img)



def gen_array(ptrn_imgs, ptrn_num):
    for i in range(ptrn_num):
        img0 = cv2.imread(ptrn_imgs.format(num=i))
        img = img0.copy()
        cntr = find_pattern_contours(ptrn_imgs.format(num=i))
        cv2.drawContours(image=img, contours=cntr, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.imwrite('img_test.png',img)
        print(cntr)


# Turn images transparent
def transparent(myimage):

    Th = 50
    img = Image.open(myimage)
    img = img.convert("RGBA")

    pixdata = img.load()

    width, height = img.size
    for y in range(height):
        for x in range(width):
            (R,G,B,A) = pixdata[x, y]
            # if pixdata[x, y] == (255, 255, 255, 255):
            if R > Th and G > Th and B > Th and A > Th:
                pixdata[x, y] = (255, 255, 255, 0)

    img.save(myimage, "PNG")

# Merge two images
def mergeTwoImages(img_out,desired_layers):
    path = os.getcwd()
    img1 = Image.open(path + '/' + img_out.format(num=desired_layers[0]))
    img2 = Image.open(path + '/' + img_out.format(num=desired_layers[1]))
    img1 = img1.convert("RGBA")
    img2 = img2.convert("RGBA")

    final = Image.new("RGBA", img1.size)
    final = Image.alpha_composite(final, img1)
    final = Image.alpha_composite(final, img2)

    final.save("final.png","PNG")


# Turn images transparent
def white_bg_and_invert(myimage):

    Th = 50
    img = Image.open(myimage)
    img = img.convert("RGBA")

    pixdata = img.load()

    width, height = img.size
    for y in range(height):
        for x in range(width):
            (R,G,B,A) = pixdata[x, y]
            # if pixdata[x, y] == (255, 255, 255, 255):
            if A < Th:
                pixdata[x, y] = (255, 255, 255, 255)
    
    image = Image.open(myimage)
    r,g,b,a = image.split()
    rgb_image = Image.merge('RGB', (r,g,b))

    inverted_image = PIL.ImageOps.invert(rgb_image)

    r2,g2,b2 = inverted_image.split()

    final_transparent_image = Image.merge('RGBA', (r2,g2,b2,a))

    final_transparent_image.save(myimage)


