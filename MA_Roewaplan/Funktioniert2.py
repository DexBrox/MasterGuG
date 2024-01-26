import cv2
import easyocr
import numpy as np
from pdf2image import convert_from_path
import os

def check_overlap(box1, box2):
    # Prüft, ob zwei Bounding-Boxen sich überlappen
    if box1[0][0] > box2[1][0] or box1[1][0] < box2[0][0]:
        return False
    if box1[0][1] > box2[1][1] or box1[1][1] < box2[0][1]:
        return False
    return True

# Optionen
scale_factor = 2.0

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Verzeichnispfad für PDFs
pdf_dir = 'pdf'

# Liste zum Speichern der Bildpfade
image_paths = []

# Lade alle PDFs im Verzeichnis
for filename in os.listdir(pdf_dir):
    if filename.endswith('.pdf'):
        pdf_path = os.path.join(pdf_dir, filename)

        # Konvertiere PDF in Bilder
        images = convert_from_path(pdf_path)

        # Speichere die Bilder und speichere ihre Pfade
        for i, image in enumerate(images):
            image_path = f'pdf/img/{filename}_{i+1}.jpg'
            image.save(image_path, 'JPEG')
            image_paths.append(image_path)

# Verarbeite jedes Bild mit easyocr
for image_path in image_paths:
    image = cv2.imread(image_path)
    original_image = image.copy()  # Kopiere das Originalbild

    scaled_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

    # Initialisiere den easyocr Reader
    reader = easyocr.Reader(['de', 'en'])

    # Lies den Text vom skalierten Bild und erhalte die Bounding-Boxen
    original_result = reader.readtext(scaled_image, detail=1)

    # Liste für die einzigartigen Bounding-Boxen
    unique_bounding_boxes = []
    original_boxes = []  # Speichert die Bounding-Boxen des Originalbildes mit Wahrscheinlichkeit

    # Speichere die Bounding-Boxen aus dem Originalbild
    for bbox, text, prob in original_result:
        top_left = tuple([int(val / scale_factor) for val in bbox[0]])
        bottom_right = tuple([int(val / scale_factor) for val in bbox[2]])
        original_boxes.append((top_left, bottom_right, 'green', prob))

    # Rotiere das Bild um 90 Grad nach rechts und skaliere es
    rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    scaled_rotated_image = cv2.resize(rotated_image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

    # Lies den Text vom skalierten, rotierten Bild und erhalte die Bounding-Boxen
    rotated_result = reader.readtext(scaled_rotated_image, detail=1)

    # Transformiere die Bounding-Boxen zurück und füge sie hinzu, wenn sie nicht doppelt sind
    for bbox, text, prob in rotated_result:
        # Rücktransformation der Bounding-Box
        top_left = (int(bbox[0][1] / scale_factor), image.shape[0] - int(bbox[2][0] / scale_factor))
        bottom_right = (int(bbox[2][1] / scale_factor), image.shape[0] - int(bbox[0][0] / scale_factor))

        # Füge hinzu, falls keine Überlappung mit bestehenden grünen Boxen
        if not any(check_overlap((top_left, bottom_right), (obox[0], obox[1])) for obox in original_boxes):
            unique_bounding_boxes.append((top_left, bottom_right, 'blue', prob))
    
    unique_bounding_boxes.extend(original_boxes)

    # Weiterverarbeitung mit den eindeutigen Bounding-Boxen
    # Zeichne die Bounding-Boxen und speichere die Patches
    image_patches = []
    patch_texts = []
    for i, (top_left, bottom_right, color, prob) in enumerate(unique_bounding_boxes):
        patch = original_image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]  # Verwende das Originalbild für das Patch
        if patch.size > 0:  # Überprüfe, ob das Patch nicht leer ist
            box_color = (0, 255, 0) if color == 'green' else (255, 0, 0)
            cv2.rectangle(image, top_left, bottom_right, box_color, 2)
            cv2.putText(image, f"{i + 1} ({prob:.2f})", (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            # Speichere das Patch als Bild ab und lies Text aus dem Patch
            cv2.imwrite(f"patches/{i+1}.jpg", patch)


    for i, (top_left, bottom_right, color, prob) in enumerate(unique_bounding_boxes):
        patch = original_image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]  # Use the original image for the patch
        if patch.size > 0:  # Check if the patch is not empty
            if patch.shape[0] > patch.shape[1]:  # Check if the patch is taller than wider
                patch = cv2.rotate(patch, cv2.ROTATE_90_CLOCKWISE)  # Rotate the patch counter-clockwise by 90 degrees
            box_color = (0, 255, 0) if color == 'green' else (255, 0, 0)
            cv2.rectangle(image, top_left, bottom_right, box_color, 2)
            cv2.putText(image, f"{i + 1} ({prob:.2f})", (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # Save the patch as an image and read text from the patch
        cv2.imwrite(f"patches/{i+1}.jpg", patch)
        patch_text = reader.readtext(patch, detail=0)
        patch_texts.append(f"{i+1}: {' '.join(patch_text)}")
        
        
    # Speichere das Bild mit den Bounding-Boxen
    cv2.imwrite(f"results/{os.path.basename(image_path)}", image)

    # Gebe die erkannten Texte aus den Patches aus
    for text in patch_texts:
        print(text)
