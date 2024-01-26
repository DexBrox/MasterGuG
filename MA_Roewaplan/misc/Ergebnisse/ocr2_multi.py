import cv2
import easyocr
import os
from pdf2image import convert_from_path
from pathlib import Path
import csv

# Setze CUDA_VISIBLE_DEVICES auf GPU 1
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Ordnerpfade
input_dir = 'Data/Roewaplan'
output_dir = 'Results/Roewaplan'
images_dir = os.path.join('Data', 'Images')
csv_file_path = os.path.join(output_dir, 'ocr_results.csv')

# Ordner für Ergebnisse erstellen, falls noch nicht vorhanden
os.makedirs(output_dir, exist_ok=True)
os.makedirs(images_dir, exist_ok=True)

# Liste der PDF-Dateien im Eingabeordner erstellen
pdf_files = [f for f in Path(input_dir).glob('*.pdf')]

# Initialisiere den easyocr Reader
reader = easyocr.Reader(['de', 'en'])

# CSV-Datei vorbereiten
with open(csv_file_path, 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['Image', 'TextNumber', 'Text'])

    # Bildnummerierung
    image_count = 1

    # Verarbeite jede PDF-Datei
    for pdf_path in pdf_files:
        images = convert_from_path(str(pdf_path))
        for page_num, image in enumerate(images):
            image_path = os.path.join(images_dir, f'{pdf_path.stem}_page_{page_num}.jpg')
            image.save(image_path, 'JPEG')

            image = cv2.imread(image_path)
            scale_factor = 3
            scaled_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
            result = reader.readtext(scaled_image, detail=1)

            for i, (bbox, text, prob) in enumerate(result):
                top_left = (int(bbox[0][0] / scale_factor), int(bbox[0][1] / scale_factor))
                bottom_right = (int(bbox[2][0] / scale_factor), int(bbox[2][1] / scale_factor))
                box_color = (0, 255, 0) if prob > 0.80 else (0, 255, 255) if prob > 0.40 else (0, 0, 255)
                cv2.rectangle(image, top_left, bottom_right, box_color, 2)
                cv2.putText(image, str(i+1), (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
                writer.writerow([f'{image_count}_{pdf_path.stem}_page_{page_num}', i+1, text])

            output_filename = f'{image_count}_{pdf_path.stem}_ocr_boxes_page_{page_num}.jpg'
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, image)

            image_count += 1
