## This script is used to process the images and extract the text from the images using EasyOCR
## Author: Valentin Jung - 2024
## Status: development

import os
import warnings

warnings.filterwarnings("ignore", message="There is an imbalance between your GPUs.*") 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from pdf_to_image import convert_pdf_to_images
from text_recognition_eo import process_image_easyocr
from image_processing import gen_bounding_boxes, write_text, gen_out

pdf_dir = '../pdf'
output_dir = '../pdf/img'
image_paths = convert_pdf_to_images(pdf_dir, output_dir)

output_folder = '../results'
scale_factor = 2.0

for image_path in image_paths:

    output_path, base_name = gen_out(image_path, output_folder)

    results, image = process_image_easyocr(image_path)

    print(results)

    unique_boxes = gen_bounding_boxes(results, scale_factor)

    #base_name = draw_bounding_boxes(image, unique_boxes, output_folder, output_path)

    write_text(results, output_folder, base_name)

