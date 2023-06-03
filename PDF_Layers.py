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
pattern_contours = find_pattern_contours(img_out.format(num=Pattern_Layer), False)
potential_dir_contours, potential_contour_pattern, pattern_contours = find_potential_direction_contours(img_out.format(num=Direction_Layer), pattern_contours)
copies, lining, main_fabric, fold, dir_cnt = find_text(img_out.format(num=Direction_Layer), pattern_contours, potential_dir_contours, potential_contour_pattern)
rotation_angles = save_patterns(img_out.format(num=Pattern_Layer), pattern_contours, dir_cnt, potential_contour_pattern, ptrn_imgs)
fold_patterns(fold, ptrn_imgs, rotation_angles, size)
gen_array(ptrn_imgs, len(pattern_contours))