import os
import timeit
import warnings

start = timeit.default_timer()

from pdf_to_image import convert_pdf_to_images
from text_recognition_eo import process_image_easyocr
from text_recognition_tess import process_image_tess
from image_processing import gen_bounding_boxes, draw_bounding_boxes, write_text, gen_out
from cer import evaluate_cer

warnings.filterwarnings("ignore", message="There is an imbalance between your GPUs.*") 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

pdf_dir = '../pdf' 
output_dir_img = '../pdf/img'

output_dir = '../results'
gt_path = '../../labels'

image_paths = convert_pdf_to_images(pdf_dir, output_dir_img)

for image_path in image_paths:
    #results, image = process_image_easyocr(image_path)
    results, image = process_image_tess(image_path)

    base_name, image_height, image_width = gen_out(image_path, image)
    unique_boxes = gen_bounding_boxes(results)

    draw_bounding_boxes(image, unique_boxes, output_dir, base_name)
    write_text(results, output_dir, base_name, image_height, image_width)

end = timeit.default_timer()
print(f"Verarbeitungszeit: {end - start} Sekunden")

start_eva = timeit.default_timer()

cer_result = evaluate_cer(gt_path, output_dir)
print(f"CER: {cer_result}")

end_eva = timeit.default_timer()
print(f"Evaluationszeit: {end_eva - start_eva} Sekunden")









