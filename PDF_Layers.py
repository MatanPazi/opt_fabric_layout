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
pattern_contours = find_pattern_contours(img_out.format(num=Pattern_Layer))
potential_dir_contours, potnetial_contour_pattern, pattern_rect = find_potential_direction_contours(img_out.format(num=Direction_Layer), pattern_contours)
# Need to understand with each potential_dir_contours, what it is:
    # Grainline.
    # Cut on fold line.
    # General remark to be ignored.
# Use this article to detect what text is written around the arrows:
    # https://medium.com/pythoneers/text-detection-and-extraction-from-image-with-python-5c0c75a8ff14
    # Increase bounding box width times 3? 4? to bound the relevant text as well.
    # Tesseract path is required, use this:
        # pytesseract.tesseract_cmd=r'/usr/bin/tesseract'
    # Use the first answer by Rich and Oliver Wilken to crop the relevant image in which the text resides:
        # https://stackoverflow.com/questions/37177811/crop-rectangle-returned-by-minarearect-opencv-python
# TODO: Change image scale to A0.



# # Not sure I need these steps for now:
# for image in glob.glob("*.png"):
#     transparent(image)

# mergeTwoImages(img_out,desired_layers)

# white_bg_and_invert('final.png')
    
