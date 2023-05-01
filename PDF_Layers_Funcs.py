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
        images = convert_from_path(pdf_path,dpi=300)
        # image.save(str(desired_layers[j]) +'.jpg', 'JPEG')
        for i in range(len(images)):        
            # Save pages as images in the pdf
            images[i].save(path + '/' + img_out.format(num=desired_layers[j]), 'PNG')




# Source: https://richardpricejones.medium.com/drawing-a-rectangle-with-a-angle-using-opencv-c9284eae3380
# Made relevant changes.
def draw_angled_rec(x0, y0, width, height, angle, img):

    _angle = angle * math.pi / 180.0
    b = math.cos(_angle) * 0.5
    a = math.sin(_angle) * 0.5
    pt0 = (int(x0 - a * height - b * width),
           int(y0 + b * height - a * width))
    pt1 = (int(x0 + a * height - b * width),
           int(y0 - b * height - a * width))
    pt2 = (int(2 * x0 - pt0[0]), int(2 * y0 - pt0[1]))
    pt3 = (int(2 * x0 - pt1[0]), int(2 * y0 - pt1[1]))

    cv2.line(img, pt0, pt1, (0, 255, 0), 5)
    cv2.line(img, pt1, pt2, (0, 255, 0), 5)
    cv2.line(img, pt2, pt3, (0, 255, 0), 5)
    cv2.line(img, pt3, pt0, (0, 255, 0), 5)




# Find grainline contours
def find_direction_contours(image):
    img = cv2.imread(image)
    image_copy = img.copy()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite('img.png',img)                
    ret, thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)
    # cv2.imwrite('thresh.png',thresh)                

    # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
    contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    good_contours = []
    shape = (len(contours), 1)  # len(contours) rows and 1 columns
    x = np.zeros(shape).astype(int)
    y = np.zeros(shape).astype(int)
    w = np.zeros(shape).astype(int)
    h = np.zeros(shape).astype(int)
    theta = np.zeros(shape).astype(int)
    # image_copy = img.copy()
    # cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    slender_rat = 3
    min_width = 20
    max_width = 120
    min_len = 150
    first = 1
    j = 0
    for cnt in contours:
        if first == 1:        # First contour encompasses entire image
            first = 0
            continue
        good_contours.append(cnt)
        rect = cv2.minAreaRect(cnt)
        x[j] = rect[0][0]
        y[j] = rect[0][1]
        w[j] = rect[1][0]
        h[j] = rect[1][1]
        theta[j] = rect[2]   
        if w[j] == 0 or h[j] == 0:
            x = np.delete(x, j, 0)
            y = np.delete(y, j, 0)
            w = np.delete(w, j, 0)
            h = np.delete(h, j, 0) 
            theta = np.delete(theta, j, 0)
            good_contours.pop(j)
            continue
        if w[j]/h[j] > slender_rat or h[j]/w[j] > slender_rat:
            if w[j]/h[j] > slender_rat:
                if (w[j] < min_len or h[j] < min_width or h[j] > max_width):
                    x = np.delete(x, j, 0)
                    y = np.delete(y, j, 0)
                    w = np.delete(w, j, 0)
                    h = np.delete(h, j, 0)
                    theta = np.delete(theta, j, 0)
                    good_contours.pop(j)
                else:
                    if "direction contour is inside one of the pattern contours":
                        img = draw_angled_rec(x[j], y[j], w[j], h[j], theta[j], image_copy)
                        cv2.imwrite('image_copy.png',image_copy) 
                        j += 1
                    else:
                        x = np.delete(x, j, 0)
                        y = np.delete(y, j, 0)
                        w = np.delete(w, j, 0)
                        h = np.delete(h, j, 0)
                        theta = np.delete(theta, j, 0)
                        good_contours.pop(j)
            elif h[j]/w[j] > slender_rat:
                if (h[j] < min_len or w[j] < min_width or w[j] > max_width):
                    x = np.delete(x, j, 0)
                    y = np.delete(y, j, 0)
                    w = np.delete(w, j, 0)
                    h = np.delete(h, j, 0)    
                    theta = np.delete(theta, j, 0)
                    good_contours.pop(j)
                else:
                    img = draw_angled_rec(x[j], y[j], w[j], h[j], theta[j], image_copy)
                    cv2.imwrite('image_copy.png',image_copy) 
                    j += 1
        else:
            x = np.delete(x, j, 0)
            y = np.delete(y, j, 0)
            w = np.delete(w, j, 0)
            h = np.delete(h, j, 0)
            theta = np.delete(theta, j, 0)
            good_contours.pop(j)

    return x,y,w,h,theta #(return good_contours)

# Find the pattern contours
def find_pattern_contours(image):
    counter = 0
    img = cv2.imread(image)
    # Taking a matrix of size 7 as the kernel
    kernel = np.ones((7, 7), np.uint8)
    # The first parameter is the original image,
    # kernel is the matrix with which image is
    # convolved and third parameter is the number
    # of iterations, which will determine how much
    # you want to erode/dilate a given image.
    img = cv2.erode(img, kernel, iterations=6)
    img = cv2.dilate(img, kernel, iterations=3)
    ## For debugging
    # image_copy = img.copy()

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


