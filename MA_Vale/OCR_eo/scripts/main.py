import os
import timeit
import warnings

warnings.filterwarnings("ignore", message="There is an imbalance between your GPUs.*") 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

start = timeit.default_timer()

from pdf_to_image import convert_pdf_to_images
from text_recognition import process_image_easyocr
from image_processing import gen_bounding_boxes, draw_bounding_boxes, draw_text_boxes

pdf_dir = '../pdf'
output_dir = '../pdf/img'
image_paths = convert_pdf_to_images(pdf_dir, output_dir)

output_folder = '../results'
scale_factor = 2.0

for image_path in image_paths:
    results, image = process_image_easyocr(image_path)
    unique_boxes = gen_bounding_boxes(results, scale_factor)
    base_name = draw_bounding_boxes(image, unique_boxes, output_folder, image_path)
    draw_text_boxes(results, output_folder, base_name)

end = timeit.default_timer()
print(f"Verarbeitungszeit: {end - start} Sekunden")
