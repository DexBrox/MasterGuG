import os
import timeit
import warnings

start = timeit.default_timer()

from pdf_to_image import convert_pdf_to_images
from text_recognition_eo import process_image_easyocr
from text_recognition_tess import process_image_tess
from text_recognition_hybrid import process_image_hy
from image_processing import draw_bounding_boxes, write_text, gen_out

warnings.filterwarnings("ignore", message="There is an imbalance between your GPUs.*") 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

pdf_dir = '../pdf' 
output_dir_img = '../pdf/img'

output_dir = '../results/txt'
gt_path = '../../labels'

count = 0

image_paths = convert_pdf_to_images(pdf_dir, output_dir_img)

for image_path in image_paths:
    #results, image = process_image_easyocr(image_path)
    results, image = process_image_tess(image_path)
    #results, image = process_image_hy(image_path)
 
    base_name, image_height, image_width = gen_out(image_path, image)

    draw_bounding_boxes(image, results, output_dir, base_name)
    write_text(results, output_dir, base_name, image_height, image_width)

    count += 1

end = timeit.default_timer()
print(f"Verarbeitungszeit: {(end - start)/count} Sekunden pro Bild")

#"""
from cer import evaluate_cer

i = []
for i in range(1, 3):
    start_eva = timeit.default_timer()

    cer_result = evaluate_cer(gt_path, output_dir, i)
    print(f"CER für {i}: {cer_result}")
    end_eva = timeit.default_timer()

    print(f"Evaluationszeit für {i}: {end_eva - start_eva} Sekunden")

#"""



