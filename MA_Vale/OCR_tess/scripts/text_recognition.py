import cv2
import pytesseract

# Pfad zur Tesseract-Executable, falls nötig
# pytesseract.pytesseract.tesseract_cmd = r'<Pfad_zu_Ihrer_Tesseract_Executable>'

def process_image_for_text(image_path, scale_factor=2.0):
    image = cv2.imread(image_path)
    # Skaliert das Bild basierend auf dem scale_factor
    scaled_image = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor)
    
    # Konfiguration für PyTesseract, inklusive deutscher Sprache
    config = '--oem 3 --psm 1 -l deu+eng'
    
    # Extrahiert den Text aus dem skalierten Bild
    text = pytesseract.image_to_string(scaled_image, config=config)

    # Extrahiert detaillierte Bounding-Box-Informationen aus dem skalierten Bild
    boxes = pytesseract.image_to_data(scaled_image, config=config, output_type=pytesseract.Output.DICT)

    return text, boxes, image
