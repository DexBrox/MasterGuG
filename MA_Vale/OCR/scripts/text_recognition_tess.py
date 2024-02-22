import cv2
import pytesseract

def process_image_tess(image_path):
    image = cv2.imread(image_path)
    
    # Konfiguration für PyTesseract, inklusive deutscher und englischer Sprache
    config = '--oem 3 --psm 1 -l deu+eng'
    
    # Extrahiert detaillierte Bounding-Box-Informationen aus dem Bild
    boxes = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)

    results = []
    n_boxes = len(boxes['level'])
    for i in range(n_boxes):
        # Prüfen, ob der Block ein Wort enthält
        if boxes['text'][i].strip() != '':
            (x, y, w, h) = (boxes['left'][i], boxes['top'][i], boxes['width'][i], boxes['height'][i])
            # Berechnung der Bounding-Box-Koordinaten im Format von EasyOCR
            box_coords = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
            # Hinzufügen des Texts und der Konfidenz zusammen mit den Koordinaten
            results.append((box_coords, boxes['text'][i], float(boxes['conf'][i])/100))

    return results, image
