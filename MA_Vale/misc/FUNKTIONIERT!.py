import cv2
import easyocr
import os
from pdf2image import convert_from_path
import os
import cv2

# Optionen
scale_factor = 2.0

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
            image_path = f'pdf/{filename}_{i+1}.jpg'
            image.save(image_path, 'JPEG')
            image_paths.append(image_path)

# Verarbeite jedes Bild mit easyocr
for image_path in image_paths:
    image = cv2.imread(image_path)
    scaled_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

    # Initialisiere den easyocr Reader
    reader = easyocr.Reader(['de', 'en'])

    # Lies den Text vom skalierten Bild und erhalte die Bounding-Boxen
    result = reader.readtext(scaled_image, detail=1)

    # Liste für die Bildpatches
    image_patches = []

    # Liste für die erkannten Texte aus den Patches
    patch_texts = []

    # Bounding-Boxen ausschneiden und in die Liste einfügen
    for i, (bbox, text, prob) in enumerate(result):
        top_left = tuple([int(val / scale_factor) for val in bbox[0]])
        bottom_right = tuple(int(val / scale_factor) for val in bbox[2])
        w = bottom_right[0] - top_left[0]
        h = bottom_right[1] - top_left[1]

        # Schneide das Patch aus und füge es zur Liste hinzu
        patch = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
        if h > w:
            patch = cv2.rotate(patch, cv2.ROTATE_90_CLOCKWISE)
        image_patches.append(patch)

        # Zeichne die Bounding-Box auf das Bild
        top_left = tuple(int(val / scale_factor) for val in bbox[0])
        bottom_right = tuple(int(val / scale_factor) for val in bbox[2]) 
        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

        # Füge die Nummer zur Bounding-Box hinzu
        cv2.putText(image, str(i+1), (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Speichere das Bild mit den Bounding-Boxen
        cv2.imwrite(f"results/{os.path.basename(image_path)}", image)

    # Speichere das Bild mit den Bounding-Boxen
    cv2.imwrite(f"output/{os.path.basename(image_path)}", scaled_image)

    # Speichere jedes Patch als Bild ab
    for i, patch in enumerate(image_patches):
        cv2.imwrite(f"patches/{i+1}.jpg", patch)

    # Lies Text aus jedem Patch
    for i, patch in enumerate(image_patches):
        patch_text = reader.readtext(patch, detail=0)
        patch_texts.append(f"{i+1}: {' '.join(patch_text)}")

    # Gebe die erkannten Texte aus den Patches aus
    for text in patch_texts:
        print(text)
