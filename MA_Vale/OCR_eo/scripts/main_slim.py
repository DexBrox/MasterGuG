import timeit
start_imp = timeit.default_timer()
start_def = timeit.default_timer()
import os
import warnings

warnings.filterwarnings("ignore", message="There is an imbalance between your GPUs.*") 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from pdf_to_image import convert_pdf_to_images
from text_recognition import process_image_easyocr
from image_processing import gen_bounding_boxes, write_text, gen_out

end_imp = timeit.default_timer()
start_pdf = timeit.default_timer()

pdf_dir = '../pdf'
output_dir = '../pdf/img'
image_paths = convert_pdf_to_images(pdf_dir, output_dir)

end_pdf = timeit.default_timer()

output_folder = '../results'
scale_factor = 2.0

for image_path in image_paths:
    output_path, base_name = gen_out(image_path, output_folder)

    start_img = timeit.default_timer()
    results, image = process_image_easyocr(image_path)
    end_img = timeit.default_timer()

    unique_boxes = gen_bounding_boxes(results, scale_factor)

    #base_name = draw_bounding_boxes(image, unique_boxes, output_folder, output_path)

    write_text(results, output_folder, base_name)

end_def = timeit.default_timer()
print(f"Importzeit: {end_imp - start_imp} Sekunden")
print(f"PDF-Konvertierungszeit: {end_pdf - start_pdf} Sekunden")
print(f"Bildverarbeitungszeit: {end_img - start_img} Sekunden")

print(f"Gesamt: {end_def - start_def} Sekunden")
