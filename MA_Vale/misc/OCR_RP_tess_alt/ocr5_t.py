import os
import pytesseract
from PIL import ImageDraw
from pdf2image import convert_from_path

import shutil

# Lösche den gesamten "patches" Ordner und dessen Inhalt
shutil.rmtree('patches')

pdf_folder = "pdf"
output_folder = "output"
patches_folder = "patches"  # Neuer Ordner für Patches

# Create output and patches folders if they don't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

if not os.path.exists(patches_folder):
    os.makedirs(patches_folder)

# Get a list of all PDF files in the folder
pdf_files = [file for file in os.listdir(pdf_folder) if file.endswith(".pdf")]

# Loop through each PDF file
for pdf_file in pdf_files:
    pdf_path = os.path.join(pdf_folder, pdf_file)

    # Convert PDF to images
    pages = convert_from_path(pdf_path)

    for page_number, image in enumerate(pages):
        # Perform OCR using pytesseract
        ocr_result = pytesseract.image_to_data(image, 
                                       #lang='deu',  # Setzt die Sprache auf Deutsch
                                       config='--oem 1 --psm 0',  # Verwendet LSTM Engine und setzt PSM auf 6
                                       output_type=pytesseract.Output.DICT)

        # Draw bounding boxes and text
        draw = ImageDraw.Draw(image)
        for i in range(len(ocr_result["text"])):
            if int(ocr_result["conf"][i]) > 40 and ocr_result["text"][i].strip() != "":
                (x, y, w, h) = (ocr_result["left"][i], ocr_result["top"][i], ocr_result["width"][i], ocr_result["height"][i])
                # Ersetze '\u2014' durch '-' und '\u201c' durch '"'
                text = ocr_result["text"][i].replace('\u2014', '-').replace('\u201c', '"').replace('\u2018', '').replace('\u2019', '')
                text = text.encode('utf-8', 'ignore').decode('utf-8')
                draw.rectangle((x, y, x + w, y + h), outline="green", width=2)
                draw.text((x, y-10), text, fill="black")
                
                # Erstelle den Patch aus der Bounding Box
                patch = image.crop((x, y, x + w, y + h))
                # Speichere den Patch im "patches" Ordner
                patch_output_path = os.path.join(patches_folder, f"{os.path.splitext(pdf_file)[0]}_page_{page_number + 1}_patch_{i + 1}.jpg")
                patch.save(patch_output_path)

        # Save the image with bounding boxes and text
        output_image_path = os.path.join(output_folder, f"{os.path.splitext(pdf_file)[0]}_page_{page_number + 1}.jpg")
        image.save(output_image_path)
