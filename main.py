import click
from logging import getLogger
import pytesseract
from PIL import Image
import cv2
import pdf2image

image = cv2.imread("samples/test_scan.jpg")
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

(thresh, im_bw) = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
cv2.imwrite("samples/gray_image.jpg", gray_image)
cv2.imwrite("samples/imbw_image.jpg", im_bw)
print(pytesseract.image_to_string("samples/imbw_image.jpg", lang="eng"))

logger = getLogger(__name__)

images = pdf2image.convert_from_path('samples/2003.00744v1_image_pdf.pdf',dpi=300, grayscale=True)
for i in range(len(images)):
    images[i].save('samples/page' + str(i) + '.png','PNG')

for i in range(len(images)):
    print(pytesseract.image_to_string(f"samples/page{i}.png", lang="eng"))
# @click.command()
# @click.option('--count', default=1, help='Number of greetings.')
# @click.option('--name', prompt='Your name',
#               help='The person to greet.')
# @click.option('--test', default=3, prompt='Number of cycles')
# def hello(count, name, test):
#     """Simple program that greets NAME for a total of COUNT times."""
#     for x in range(count):
#         click.echo(f"Hello {name}!")
#     for i in range(test):
#         print("testing...")
#
# if __name__ == '__main__':
#     hello()
