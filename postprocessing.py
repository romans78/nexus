# def rotate_image(image, angle):
#     return imutils.rotate(image, angle)
#
# def remove_lines(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
#     horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
#     detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,
#                                       horizontal_kernel, iterations=2)
#
#     cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     cnts = cnts[0] if len(cnts) == 2 else cnts[1]

import word2vec as w2v


