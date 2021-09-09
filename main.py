import os
import shutil
import click
import logging
import pytesseract
import cv2
import pdf2image
import numpy as np

is_pdf = False

@click.command()
@click.option('--input', help='input file')
@click.option('--output', help='output file')
@click.option('--verbose', is_flag=True, help='verbose mode')
def start(input, output, verbose):
    """OCR Reader"""
    logging.basicConfig(format='[INFO] %(asctime)s - %(message)s', level=logging.INFO, datefmt='%d-%b-%y %H:%M:%S')

    ext = input.rsplit(maxsplit=1, sep='.')
    if len(ext) > 1 and ext[-1].lower() in ('png', 'pdf', 'jpg', 'jpeg'):
        input_files = []
        if ext[-1].lower() == 'pdf':
            directory = ext[0].rsplit(maxsplit=1, sep='/')[-1]
            if os.path.exists(directory):
                shutil.rmtree(directory)
            os.mkdir(directory)
            pdf2image.convert_from_path(input, dpi=300, output_folder=directory, output_file='page', fmt='png',
                                        grayscale=True)
            input_files = [directory + '/' + f for f in os.listdir(directory)]
            global is_pdf
            is_pdf = True
        else:
            input_files.append(input)
        output_files = []
        input_files.sort()
        print(input_files)
        if not os.path.exists('temp_output'):
            os.mkdir('temp_output')
        else:
            shutil.rmtree('temp_output')
            os.mkdir('temp_output')
        if not os.path.exists('temp'):
            os.mkdir('temp')
        else:
            shutil.rmtree('temp')
            os.mkdir('temp')

        for input_file in input_files:
            preprocessed_file = preprocess(input_file, verbose)
            output_files.append(OCR_reader(preprocessed_file, verbose))

        with open('output_ocr.txt','w+') as output_file:
            for of in output_files:
                with open(of, 'r') as out_file:
                    output_file.write(out_file.read())
        postprocess(input,output, verbose)

    else:
        click.echo(f'Unsupported file format: {input}')


def preprocess(input_file, verbose_mode):
    output_file = ""
    global is_pdf

    # grayscaling
    if verbose_mode:
        logging.info(msg=f"Performing grayscaling for file {input_file}")
    image = cv2.imread(input_file)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    output_file = f"temp/gray_{input_file.rsplit(maxsplit=1, sep='/')[-1]}"
    cv2.imwrite(output_file, gray_image)
    # if not is_pdf:
    #     #binarization
    #     if verbose_mode:
    #         logging.info(msg=f"Performing binarization for file {input_file}")
    #     (thresh, im_bw) = cv2.threshold(gray_image, 64, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #     output_file = f"temp/imbw_{input_file.rsplit(maxsplit=1, sep='/')[-1]}"
    #     cv2.imwrite(output_file, im_bw)
    #
    # if not is_pdf:
    #     #noise removal
    #     if verbose_mode:
    #         logging.info(msg=f"Performing noise removal for file imbw_{input_file.rsplit(maxsplit=1, sep='/')[-1]}")
    #     kernel = np.ones((1, 1), np.uint8)
    #     image = cv2.dilate(im_bw, kernel, iterations=1)
    #     kernel = np.ones((1, 1), np.uint8)
    #     image = cv2.erode(image, kernel, iterations=1)
    #     image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    #     image = cv2.medianBlur(image, 3)
    #     output_file = f"temp/nonoise_{input_file.rsplit(maxsplit=1, sep='/')[-1]}"
    #     cv2.imwrite(output_file, image)
    #
    # if not is_pdf:
    #     #dilation
    #     if verbose_mode:
    #         logging.info(msg=f"Performing dilation for file nonoise_{input_file.rsplit(maxsplit=1, sep='/')[-1]}")
    #     image = cv2.bitwise_not(image)
    #     kernel = np.ones((2, 2), np.uint8)
    #     image = cv2.dilate(image, kernel, iterations=1)
    #     image = cv2.bitwise_not(image)
    #     output_file = f"temp/dilate_{input_file.rsplit(maxsplit=1, sep='/')[-1]}"
    #     cv2.imwrite(output_file, image)

    return output_file


def postprocess(input_file,output_file, verbose_mode):
    pass


def OCR_reader(input_file, verbose_mode):

    text = pytesseract.image_to_string(input_file, lang="eng")

    with open(f"temp_output/output_{input_file.rsplit(maxsplit=1,sep='/')[-1].rsplit(maxsplit=1,sep='.')[0]}.txt",'x') as output_file:
        output_file.write(text)
    return output_file.name


if __name__ == '__main__':
    start()
