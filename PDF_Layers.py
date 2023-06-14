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

# Extraction section
pdfLayers(pdf_name, pdf_out, desired_layers)
size = pdf2image(desired_layers, pdf_out, img_out)
pattern_contours = find_pattern_contours(img_out.format(num=Pattern_Layer), False)
potential_dir_contours, potential_contour_pattern, pattern_contours = find_potential_direction_contours(img_out.format(num=Direction_Layer), pattern_contours)
copies, lining, main_fabric, fold, dir_cnt = find_text(img_out.format(num=Direction_Layer), pattern_contours, potential_dir_contours, potential_contour_pattern)
rotation_angles = save_patterns(img_out.format(num=Pattern_Layer), pattern_contours, dir_cnt, potential_contour_pattern, ptrn_imgs)
fold_patterns(fold, ptrn_imgs, rotation_angles, size)
## For debugging
# gen_array(ptrn_imgs, len(pattern_contours), False)

# Optimization section
Fabric_width = int(1.5 * 1000)   #1.5[m] to pixels, each pixel is 1[mm^2]
main_array = init_main_arr(Fabric_width, len(copies), ptrn_imgs)
opt_place(main_array, len(copies), ptrn_imgs)

# TODO:
# The function gen_array should also return the approx polygon created and the center of gravity (Reference point).
    # In the furure, can make sure the distance between points in the approx polygon aren't bigger than 50(?) pixels (Make it a configurable parameter).
        # Since the patterns are going to go point by point, so don't want to miss to much area.


# Place first pattern according to least amount of waste (Choose according to largest area for now, to be improved in the future)
# Next, each pattern's reference point is set to be on each of the previous pattern's polygon points.
# After each reference point placement, a minimization algo should run to move the added pattern from inside the other patterns (Preventing overlap).
    # This will be done by giving each pixel inside the pattern arrays a value equal to the distance from the pattern edge (min on x and y distances), 0 at the edge.
    # The cost func will return the sum of the area covered by the added pattern divided by the sum of the added pattern.
    # So the pattern will strive to be outside the other patterns where the values are 1(?)
# The lowest cost returned will be the optimal placement of that pattern.

# Go over rest of patterns one by one and check which had the lowest cost.
# In the future, can rule out points if there's no way the pattern can go there. Will reduce calc time.