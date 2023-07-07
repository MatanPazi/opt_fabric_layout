from opt_layout_funcs import *
import glob

# Direction_Layer = 1
# Pattern_Layer = 2
# pdf_name = 'PS_LuluCardi_Pattern(COPYSHOP_24x36)_BCUP.pdf'

# Direction_Layer = 0
# Pattern_Layer = 3
# pdf_name = 'LL Leo Pattern Size A0.pdf'

# Direction_Layer = 0
# Pattern_Layer = 1
# pdf_name = 'bt119-A0-pattern.pdf'

Direction_Layer = 0
Pattern_Layer = 1
pdf_name = 'bt67-A0-pattern.pdf'

## Seems like all the data is on layer 0...???
# Direction_Layer = 0
# Pattern_Layer = 1
# pdf_name = 'BLOMMA TANK-A0-copyshop.pdf'

# There's a legend for "cut on fold" so doesn't say so on the arrow itself. Need to add legend layer to read.
# Direction_Layer = 8
# Pattern_Layer = 9
# pdf_name = '9-BAS_trapeze_patronAVECmarges-AtelierCharlotteAuzou_A0_34-48.pdf'

# Made up of more than 1 page, need to handle differently...
# Direction_Layer = 1
# Pattern_Layer = 3
# pdf_name = 'PS_ByrdieButtonup_UniversalPatternPieces(A0).pdf'

pdf_out = 'Page_{page_num}_Layer_{layer_num}.pdf'
img_out_init = 'Page_{page_num}_Layer_{layer_num}.png'
img_out = 'Out_{num}.png'
ptrn_imgs = 'pattern_{num}.png'
desired_layers = [Direction_Layer,Pattern_Layer]

# Extraction section

# To deal with pdfs with multiple pages:
    # Consider first seperating pdf pages to independent pdf files and running the func pdfLayers on each pdf file.
    # Need to change pdf naming to include page #.
    # See reference: https://pikepdf.readthedocs.io/en/latest/topics/pages.html
    # OR
    # Consider saving a layer along all pages (several pages 1 layer)
    # And then converting the pages to images, see example:
    # https://stackoverflow.com/questions/62161218/lost-information-getting-pdf-page-as-image
page_count = pdfLayers(pdf_name, pdf_out, desired_layers)

size = pdf2image(desired_layers, pdf_out, img_out_init, img_out, page_count)
pattern_contours = find_pattern_contours(img_out.format(num=Pattern_Layer), False)
potential_dir_contours, potential_contour_pattern, pattern_contours = find_potential_direction_contours(img_out.format(num=Direction_Layer), pattern_contours)
copies, lining, main_fabric, fold, dir_cnt = find_text(img_out.format(num=Direction_Layer), pattern_contours, potential_dir_contours, potential_contour_pattern)
rotation_angles = save_patterns(img_out.format(num=Pattern_Layer), pattern_contours, dir_cnt, potential_contour_pattern, ptrn_imgs)
fold_patterns(fold, ptrn_imgs, rotation_angles, size, page_count)
## For debugging
# gen_array(ptrn_imgs, len(pattern_contours), False)

# Optimization section
fabric_width = int(1.5 * 1000)   #1.5[m] to pixels, each pixel is 1[mm^2]
# main_array = init_main_arr(Fabric_width, len(copies), ptrn_imgs)
opt_place(len(copies), ptrn_imgs, fabric_width)


'''
TODO:
1. Support copies
2. Support 180 rotation for each pattern
3. Support legends (E.g 9-BAS_trapeze_patronAVECmarges-AtelierCharlotteAuzou_A0_34-48.pdf pattern)
'''
