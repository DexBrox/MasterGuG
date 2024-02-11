import os

from pdf_to_image import convert_pdf_to_images
from text_recognition import process_image_for_text
from image_processing import draw_bounding_boxes_and_extract_patches

pdf_dir = './pdf'
image_paths = convert_pdf_to_images(pdf_dir)

for image_path in image_paths:
    results, image = process_image_for_text(image_path)
    # Die Funktion draw_bounding_boxes_and_extract_patches kümmert sich nun um das Speichern der Bilder.
    # Der Originalname (ohne .pdf-Endung) wird beibehalten und "_boxed" wird hinzugefügt.
    draw_bounding_boxes_and_extract_patches(image, results, image_path)
