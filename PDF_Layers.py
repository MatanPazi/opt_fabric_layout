from PDF_Layers_Funcs import *
import glob

# Direction_Layer = 1
# Pattern_Layer = 2
# pdf_name = 'PS_LuluCardi_Pattern(COPYSHOP_24x36)_BCUP.pdf'

Direction_Layer = 0
Pattern_Layer = 3
pdf_name = 'LL Leo Pattern Size A0.pdf'

pdf_out = 'Out_{num}.pdf'
img_out = 'Out_{num}.png'
ptrn_imgs = 'pattern_{num}.png'
desired_layers = [Direction_Layer,Pattern_Layer]

pdfLayers(pdf_name, pdf_out, desired_layers)
size = pdf2image(desired_layers, pdf_out, img_out)
pattern_contours = find_pattern_contours(img_out.format(num=Pattern_Layer))
potential_dir_contours, potential_contour_pattern, pattern_contours = find_potential_direction_contours(img_out.format(num=Direction_Layer), pattern_contours)
copies, lining, main_fabric, fold, dir_cnt = find_text(img_out.format(num=Direction_Layer), pattern_contours, potential_dir_contours, potential_contour_pattern, ptrn_imgs)
rotation_angles = save_patterns(img_out.format(num=Pattern_Layer), pattern_contours, dir_cnt, potential_contour_pattern, ptrn_imgs)
fold_patterns(fold, ptrn_imgs, rotation_angles, size)
gen_array(ptrn_imgs, len(pattern_contours))

img1 = cv2.imread(ptrn_imgs.format(num=0))
img2 = np.zeros((1000,1000,3), dtype=np.uint8)
img2[:] = 255
img2[0:img1.shape[0], 0:img1.shape[1]] = img1
cv2.imwrite('img_test.png',img2)





# print(fold)

# TODO:
# 1. Rescale pattern images.
# 2. Clean and comment code

# Continue on to optimization part:
# 3. Generate a 2x2 matrix for each pattern and give a relevant value to each pixel.
# 4. ...







# # Not sure I need these steps for now:
# for image in glob.glob("*.png"):
#     transparent(image)

# mergeTwoImages(img_out,desired_layers)

# white_bg_and_invert('final.png')
    
