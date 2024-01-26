import cv2
import pytesseract
import os
import numpy as np

def extract_and_draw_boxes(image, data, box_color, scale_factor):
    for i in range(len(data['text'])):
        if int(data['conf'][i]) > 60:  # Nur Boxen mit einer Konfidenz > 60 anzeigen
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            x, y, w, h = int(x / scale_factor), int(y / scale_factor), int(w / scale_factor), int(h / scale_factor)

            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), -1)  # Weiß gefüllte Box
            cv2.rectangle(image, (x, y), (x + w, y + h), box_color, 2)  # Farbige Umrandung

            text = data['text'][i]
            cv2.putText(image, text, (x, y + h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

# Setze den Pfad für Tesseract, falls erforderlich
# pytesseract.pytesseract.tesseract_cmd = r'Pfad_zur_Tesseract-Exe'

# Lade das Bild
image_path = 'Data/PLAN.jpg'
image = cv2.imread(image_path)
scale_factor = 2  # Skalierungsfaktor
scaled_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

# Konfiguration für Tesseract
config = '--psm 6'

# Erkenne horizontalen Text und Bounding-Boxen
data_horizontal = pytesseract.image_to_data(scaled_image, config=config, output_type=pytesseract.Output.DICT)
extract_and_draw_boxes(image, data_horizontal, (0, 255, 0), scale_factor)  # Grün für horizontalen Text

# Drehen des Bildes um 90 Grad für die Erkennung von vertikalem Text
rotated_scaled_image = cv2.rotate(scaled_image, cv2.ROTATE_90_CLOCKWISE)
data_vertical = pytesseract.image_to_data(rotated_scaled_image, config=config, output_type=pytesseract.Output.DICT)
extract_and_draw_boxes(image, data_vertical, (255, 0, 0), scale_factor)  # Blau für vertikalen Text

# Speichere das bearbeitete Bild
output_dir = 'Results'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_filename = os.path.splitext(os.path.basename(image_path))[0] + '_ocr_pt.jpg'
output_path = os.path.join(output_dir, output_filename)
cv2.imwrite(output_path, image)

print(f"Bild gespeichert unter: {output_path}")
