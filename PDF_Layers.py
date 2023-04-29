from PDF_Layers_Funcs import *
import glob


A0_Width  = 841
A0_Height = 1189
Direction_Layer = 1
Pattern_Layer = 2

pdf_name = 'PS_LuluCardi_Pattern(COPYSHOP_24x36)_BCUP.pdf'
pdf_out = 'Out_{num}.pdf'
img_out = 'Out_{num}.png'
desired_layers = [Direction_Layer,Pattern_Layer]

# pdfLayers(pdf_name, pdf_out, desired_layers)
# pdf2image(desired_layers, pdf_out, img_out)
# x,y,w,h,theta = find_direction_contours(img_out.format(num=Direction_Layer))
pattern_contours = find_pattern_contours(img_out.format(num=Pattern_Layer))
print(pattern_contours)
# for image in glob.glob("*.png"):
#     transparent(image)

# mergeTwoImages(img_out,desired_layers)

# white_bg_and_invert('final.png')
    

# import cv2

# img = cv2.imread("Out_2.png")
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imwrite('img.png',img)                
# ret, thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)
# cv2.imwrite('thresh.png',thresh)                

# # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
# contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
                                      
# # draw contours on the original image
# image_copy = img.copy()
# cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
# slender_rat = 3
# min_width = 20
# max_width = 70
# min_len = 90
# for cnt in contours:
#     x,y,w,h = cv2.boundingRect(cnt)
#     if w/h > slender_rat:
#         if w > min_len and (h > min_width and h < max_width):
#             img = cv2.rectangle(image_copy,(x,y),(x+w,y+h),(0,0,255),2)
#     elif h/w > slender_rat:
#         if h > min_len and (w > min_width and w < max_width):
#             img = cv2.rectangle(image_copy,(x,y),(x+w,y+h),(0,0,255),2)
# cv2.imwrite('image_copy.png',image_copy)                
# # # # # see the results
# # # # cv2.imshow('None approximation', image_copy)

# # # # k = cv2.waitKey(0) & 0xFF
# # # # if k == 27:         # wait for ESC key to exit
# # # #     cv2.destroyAllWindows()

