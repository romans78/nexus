import imutils
from main import *








image = cv2.imread('samples/82251504.png')
base_image = image.copy()
#cv2.imshow('source_image', image)
x,y,d = base_image.shape
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
#cv2.imshow('source_image', thresh)
horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,
horizontal_kernel, iterations=2)

cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
imutils.rotate(base_image,15)
cv2.imshow('source_image', base_image)

base_image[0:y,0:x] = base_image[3:y+3,0:x]

for c in cnts:
   #cv2.drawContours(image, [c], -1, (255, 255, 255), 2)
    x, y, w, h = cv2.boundingRect(c)
    if h <10 and w > 150:

        #roi = base_image[0:y+h, 0:x+im_width]
        base_image[y:y+h//3,x:x+w] = 255

cv2.imwrite('temp/result.png', base_image)
repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 6))

result = 255 - cv2.morphologyEx(255 - image, cv2.MORPH_CLOSE, repair_kernel,
iterations=1)

#cv2.imshow('resultant image', result)
cv2.waitKey()
cv2.destroyAllWindows()




#
# im_height, im_width, im_depth = image.shape
#
# print(im_height, im_width, im_depth)
# base_image = image.copy()
#
# def thin_font(image):
#     import numpy as np
#     image = cv2.bitwise_not(image)
#     kernel = np.ones((2,2),np.uint8)
#     image = cv2.erode(image, kernel, iterations=1)
#     image = cv2.bitwise_not(image)
#     return (image)
#
#
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# blur = cv2.GaussianBlur(gray,(7,7),0)
# thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
# cv2.imwrite('temp/thresh.png', thresh)
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50,5))
# dilate = cv2.dilate(thresh, kernel, iterations=1)
# cv2.imwrite('temp/dilate.png', dilate)
# cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cnts = cnts[0] if len(cnts) == 2 else cnts[1]
# cnts = sorted(cnts, key=lambda x: cv2.boundingRect(x)[1])
# for c in cnts:
#     x,y,w,h = cv2.boundingRect(c)
#     if h <20 and w > 150:
#         roi = base_image[0:y+h, 0:x+im_width]
#         base_image[y:y+h,x:x+w] = 255
#         cv2.rectangle(image, (x,y), (x+w,y+h),(36,255,12),1)
# cv2.imwrite('temp/output.png', base_image)
#
# image=thin_font(base_image)
#
# cv2.imwrite('temp/output_image.png', image)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cnts = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cnts = cnts[0] if len(cnts) == 2 else cnts[1]
# cnts = sorted(cnts, key=lambda x: cv2.boundingRect(x)[1])
# for c in cnts:
#
#     x,y,w,h = cv2.boundingRect(c)
#     if h <20 and w > 150:
#
#         roi = base_image[0:y+h, 0:x+im_width]
#         base_image[y:y+h,x:x+w] = 255
#         cv2.rectangle(base_image, (x,y), (x+w,y+h),(36,255,12),1)
# cv2.imwrite('temp/output_image_image.png', base_image)
text = pytesseract.image_to_string('temp/result.png', lang="eng")

with open(f"temp_output/output_image_im2.txt",'w') as output_file:
    output_file.write(text)