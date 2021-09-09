import cv2
import imutils
import pytesseract
import pdf2image
import os
import shutil
import numpy as np

def preprocess_82251504(image):
    #image = cv2.imread('samples/82251504.png')
    x, y, d = image.shape

    image[y+15:y+250, 0:x] = 255
    image[0:145,0:x] = 255

    base_image = image.copy()
    x, y, d = base_image.shape
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,
                                      horizontal_kernel, iterations=2)
    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    imutils.rotate(base_image, 1)
    base_image[0:y, 0:x] = base_image[3:y + 3, 0:x] #move image 3 points up

    for c in cnts:
        # cv2.drawContours(image, [c], -1, (255, 255, 255), 2)
        x, y, w, h = cv2.boundingRect(c)
        if h < 10 and w > 150:
            # roi = base_image[0:y+h, 0:x+im_width]
            base_image[y:y + h // 3, x:x + w] = 255

    cv2.imwrite('temp/result.png', base_image)
    # repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 6))
    #
    # result = 255 - cv2.morphologyEx(255 - image, cv2.MORPH_CLOSE, repair_kernel,
    #                                 iterations=1)

    return 'temp/result.png'

def preprocess_pdf(pdf_file):
    directory = 'pdf_file'
    if not os.path.exists(directory):
        os.mkdir(directory)
    else:
        shutil.rmtree(directory)
        os.mkdir(directory)
    pdf2image.convert_from_path(pdf_file, dpi=300, output_folder=directory, output_file='page', fmt='png',
                                grayscale=True)
    input_files = [directory + '/' + f for f in os.listdir(directory)]


    for png_file in input_files:
        pfile = cv2.cvtColor(cv2.imread(png_file), cv2.COLOR_BGR2GRAY)
        if png_file.endswith('1.png'):
            x,y = pfile.shape
            pfile[0:y,0:200] = 255
            pfile[y+313:y+520, 35+x//3:x] = 255
        cv2.imwrite(png_file,pfile)
    input_files.sort()

    return input_files


def preprocess_test_scan(image):
    base_image = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    (thresh, image) = cv2.threshold(image, 64, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 3)

    image = cv2.bitwise_not(image)
    kernel = np.ones((2, 2), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    image = imutils.rotate(image, 1)
    cv2.imwrite('temp/result.jpg', image)

#preprocess_82251504(cv2.imread('samples/82251504.png'))

#text = pytesseract.image_to_string('temp/result.png', lang="eng")

# with open(f"temp_output/output_image_82251504.txt",'w') as output_file:
#     output_file.write(text)
#
# for i in preprocess_pdf('samples/2003.00744v1_image_pdf.pdf'):
#     text = pytesseract.image_to_string(i, lang="eng")
#     with open(f"{i}.txt", 'w') as output_file:
#         output_file.write(text)

preprocess_test_scan(cv2.imread('samples/test_scan.jpg'))
text = pytesseract.image_to_string('temp/result.jpg', lang="eng")

with open(f"temp_output/output_test_scan.txt",'w') as output_file:
     output_file.write(text)