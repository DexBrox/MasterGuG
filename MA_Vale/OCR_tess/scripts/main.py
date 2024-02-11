import os
import pytesseract
import timeit

start = timeit.default_timer()

from pdf_to_image import convert_pdf_to_images
from text_recognition import process_image_for_text
from image_processing import draw_bounding_boxes_and_extract_patches

# Pfad zur Tesseract-Executable unter Linux oder macOS (normalerweise nicht notwendig, wenn richtig im PATH)
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'  # Der Pfad kann variieren


pdf_dir = '../pdf'
image_paths = convert_pdf_to_images(pdf_dir)

for image_path in image_paths:
    text, boxes, image = process_image_for_text(image_path)
    # Die Funktion draw_bounding_boxes_and_extract_patches kümmert sich nun um das Speichern der Bilder.
    # Der Originalname (ohne .pdf-Endung) wird beibehalten und "_boxed" wird hinzugefügt.
    draw_bounding_boxes_and_extract_patches(image, boxes, image_path)

end = timeit.default_timer()
print(f"pytesseract Laufzeit: {end - start} Sekunden")