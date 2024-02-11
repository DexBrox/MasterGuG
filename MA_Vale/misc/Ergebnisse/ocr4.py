import os
import numpy as np
from pdf2image import convert_from_path
from PIL import ImageDraw, ImageFont, Image
import easyocr
import csv

# Definieren Sie den Pfad zu Ihren Ordnern
input_folder = 'Data/Roewaplan'  # Der Ordner, in dem die PDF-Dateien liegen
output_folder = 'Results/Roewaplan'  # Der Ausgabeordner
patches_folder = os.path.join(output_folder, 'Patches')  # Der Ordner für die geschnittenen Patches

# Einstellungen für EasyOCR
languages = ['en', 'de']  # Die Sprachen, die für die OCR-Texterkennung verwendet werden sollen
rotation_info = [0, 90, 180, 270]  # Die Rotationen, die für die Texterkennung berücksichtigt werden sollen
contrast_ths = 0.1  # Der Kontrastschwellenwert für die Texterkennung
adjust_contrast = 0.5  # Der Kontrastwert für die Texterkennung
add_margin = 0.1  # Der Rand, der um die Bounding Box hinzugefügt wird
paragraph = True  # Legt fest, ob Text in Absätzen oder Zeilen erkannt werden soll
min_size = 1  # Die minimale Größe für erkannte Texte
text_threshold = 0.1  # Der Schwellenwert für die Texterkennung
low_text = 0.2  # Der Schwellenwert für niedrigen Text
link_threshold = 0.4  # Der Schwellenwert für die Verknüpfung von Text

# Erstellen Sie das Ausgabeverzeichnis und das Verzeichnis für Patches, falls sie nicht existieren
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
if not os.path.exists(patches_folder):
    os.makedirs(patches_folder)

# Initialisieren des EasyOCR Readers
reader = easyocr.Reader(['en', 'de'])

# CSV-Datei vorbereiten
csv_file_path = os.path.join(output_folder, 'extracted_texts1.csv')
csv_data = {}

# Funktion zum Ausführen von OCR auf einem Bild mit den angegebenen Einstellungen
def ocr_image(image):
    np_image = np.array(image)
    return reader.readtext(np_image, detail=1, rotation_info=rotation_info, contrast_ths=contrast_ths, adjust_contrast=adjust_contrast, add_margin=add_margin)

# Durchlaufen aller PDFs im Verzeichnis
for filename in os.listdir(input_folder):
    if filename.endswith('.pdf'):
        file_path = os.path.join(input_folder, filename)
        images = convert_from_path(file_path, dpi=300)  # Erhöhen der DPI für bessere Qualität

        for page_number, image in enumerate(images):
            ocr_results = ocr_image(image)
            image_name = f'{filename[:-4]}_page_{page_number}'
            csv_data[image_name] = []

            draw = ImageDraw.Draw(image)
            font = ImageFont.load_default()
            for index, (bbox, text, conf) in enumerate(ocr_results):
                bbox_coords = [bbox[0][0], bbox[0][1], bbox[2][0], bbox[2][1]]

                # Ausschneiden und Speichern der Bounding-Box als separates Bild
                patch = image.crop(bbox_coords)
                patch_name = f'Bild{page_number}_{index}.png'
                patch_file_path = os.path.join(patches_folder, patch_name)
                patch.save(patch_file_path)

                # Text in CSV-Datei speichern
                csv_data[image_name].append(text)

                # Zeichnen der Bounding-Box
                if conf > 0.66:
                    color = 'green'
                elif conf < 0.33:
                    color = 'red'
                else:
                    color = 'yellow'
                draw.rectangle(bbox_coords, outline=color)
                label = f"{index} - {conf:.2f}"
                text_position = (bbox_coords[0], bbox_coords[1] - 10) if bbox_coords[1] - 10 > 0 else (bbox_coords[0], bbox_coords[3] + 5)
                draw.text(text_position, label, fill="black", font=font)

            # Gesamtes Bild speichern
            output_file_path = os.path.join(output_folder, f'{image_name}.png')
            image.save(output_file_path)

# CSV-Datei schreiben
with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
    csvwriter = csv.writer(csvfile)
    headers = ['Punktnummer'] + list(csv_data.keys())
    csvwriter.writerow(headers)

    max_length = max(len(texts) for texts in csv_data.values())
    for i in range(max_length):
        row = [f'Patch {i}']
        for image_name in csv_data:
            try:
                row.append(csv_data[image_name][i])
            except IndexError:
                row.append('')
        csvwriter.writerow(row)
