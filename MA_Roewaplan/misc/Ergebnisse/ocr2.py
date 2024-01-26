import cv2
import easyocr
import os
from pdf2image import convert_from_path
from pathlib import Path
import numpy as np
import shutil

def rotate_image(image, angle):
    """ Rotiert das Bild um den angegebenen Winkel. """
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated, M

def transform_back(bbox, M):
    """ Transformiert eine Bounding-Box zurück in das Koordinatensystem des Originalbildes. """
    points = np.array([[bbox[0][0], bbox[0][1]], [bbox[1][0], bbox[1][1]], 
                       [bbox[2][0], bbox[2][1]], [bbox[3][0], bbox[3][1]]])
    ones = np.ones(shape=(len(points), 1))
    points_ones = np.hstack([points, ones])
    transformed_points = np.dot(np.linalg.inv(M), points_ones.T).T
    transformed_points = transformed_points[:, :2].astype(int)
    return transformed_points

# GPU-Konfiguration
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Pfadeinstellungen
input_dir = 'Data/Roewaplan'
output_dir = 'Results/Roewaplan'
images_dir = os.path.join('Data', 'Images')
patches_dir = os.path.join(output_dir, 'Patches')

# Verzeichnisse erstellen, falls sie nicht existieren
os.makedirs(output_dir, exist_ok=True)
os.makedirs(images_dir, exist_ok=True)
os.makedirs(patches_dir, exist_ok=True)

# OCR-Reader initialisieren
reader = easyocr.Reader(['de', 'en'])

# PDF-Dateien verarbeiten
pdf_files = [f for f in Path(input_dir).glob('*.pdf')]
image_count = 1

for pdf_path in pdf_files:
    images = convert_from_path(str(pdf_path))
    for page_num, image in enumerate(images):
        image_path = os.path.join(images_dir, f'{pdf_path.stem}_page_{page_num}.jpg')
        image.save(image_path, 'JPEG')
        image = cv2.imread(image_path)

        # Bild skalieren
        scale_factor = 2
        scaled_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

        # OCR auf dem Originalbild
        result_original = reader.readtext(scaled_image, detail=1)

        # OCR auf dem um 90 Grad gedrehten Bild
        rotated_image, M = rotate_image(scaled_image, 90)
        result_rotated = reader.readtext(rotated_image, detail=1)

        # Matrix erweitern
        M = np.vstack([M, [0, 0, 1]])

        # Bounding-Boxen zeichnen und nummerieren
        bbox_number = 1
        for result in result_original:
            bbox, text, prob = result[0], result[1], result[2]
            pts = np.array([[(int(point[0] / scale_factor), int(point[1] / scale_factor)) for point in bbox]], np.int32)
            box_color = (0, 0, 255) if prob <= 0.33 else (0, 255, 255) if prob <= 0.66 else (0, 255, 0)
            cv2.polylines(image, pts, True, box_color, 2)

            # Patch ausschneiden
            patch = image[np.min(pts[:, :, 1]):np.max(pts[:, :, 1]), np.min(pts[:, :, 0]):np.max(pts[:, :, 0])]

            # Patch rotieren, wenn Wahrscheinlichkeit < 0,25
            if prob < 0.25:
                patch, _ = rotate_image(patch, -90)

            # OCR auf dem (gedrehten) Patch
            patch_text = reader.readtext(patch, detail=1)
            extracted_text = ' '.join([pt[1] for pt in patch_text])

            # Text, Wahrscheinlichkeit und Bounding-Box-Nummer ausgeben
            cv2.putText(image, f"{bbox_number}: {prob:.2f}", (pts[0][0][0], pts[0][0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
            print(f"Bounding-Box {bbox_number}: {extracted_text}, Wahrscheinlichkeit: {prob:.2f}")

            # Patch mit gedrehtem Format speichern und altes Patch löschen
            patch_filename = os.path.join(patches_dir, f'patch_{image_count}_{bbox_number}.jpg')
            if os.path.exists(patch_filename):
                os.remove(patch_filename)
            cv2.imwrite(patch_filename, patch)

            bbox_number += 1

        # Bearbeitetes Bild speichern
        output_filename = f'{image_count}_{pdf_path.stem}_ocr_boxes_page_{page_num}.jpg'
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, image)

        image_count += 1
