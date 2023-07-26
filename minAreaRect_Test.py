import cv2
import math


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


img = cv2.imread('minAreaRect_Test.png')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)
# detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)            
slender_rat = 3
min_width = 10
max_width = 120
min_len = 120
first = 1
image_copy = img.copy()
cv2.imwrite('image_copy.png',image_copy) 
# cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
# Used to add text on pattern
counter = 0
for cnt in contours:
    if counter == 0:        # First contour encompasses entire image
        counter += 1
        continue
    
    heir = hierarchy[0][counter][3]                         # [next, previous, first child, parent]. 
    if heir == 0:
        rect = cv2.minAreaRect(cnt)
        x = int(rect[0][0])
        y = int(rect[0][1])
        w = int(rect[1][0])
        h = int(rect[1][1])
        theta = int(rect[2])
        draw_angled_rec(x, y, w, h, theta, image_copy, 'green')
        image_tmp = cv2.putText(img=image_copy, text=str(theta)+'[deg]', org=(x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, color=(0,0,0), thickness=5)
        image_tmp = cv2.putText(img=image_copy, text='w='+str(w), org=(x, y+100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, color=(0,0,0), thickness=5)
        image_tmp = cv2.putText(img=image_copy, text='h='+str(h), org=(x, y+200), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, color=(0,0,0), thickness=5)
        cv2.imwrite('image_copy.png',image_copy)
    counter += 1