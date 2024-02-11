import cv2  # Importiert OpenCV, eine Bibliothek für Computer Vision Aufgaben.
import easyocr  # Importiert EasyOCR, eine Bibliothek zur Texterkennung.

# Konfigurationsvariablen für die Initialisierung des EasyOCR-Readers und die Texterkennung.
SCALE_FACTOR = 2.0  # Faktor zur Skalierung des Bildes, verbessert die Texterkennung bei kleineren Texten.
LANGUAGES = ['de', 'en']  # Definiert die Sprachen, die bei der Texterkennung berücksichtigt werden sollen.

# Konfigurationsvariablen für die readtext Methode von EasyOCR.
DECODER = 'greedy'  # Bestimmt den Dekodierungsalgorithmus ('greedy' oder 'beamsearch').
GPU = True  # Gibt an, ob die GPU für die Verarbeitung verwendet werden soll. Bei False wird die CPU verwendet.
ALLOWLIST = None  # Zeichensatz, der bei der Erkennung berücksichtigt werden soll. None bedeutet keine Einschränkung.
BLOCKLIST = ['!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '_', '+', '=', '{', '}', '[', ']', '|', '\\', ':', ';', '"', "'", '<', '>', ',', '.', '?', '/']
 # Zeichensatz, der von der Erkennung ausgeschlossen werden soll.
WORKERS = 10 # Anzahl der Worker-Prozesse. 0 verwendet den Standardwert.
BATCH_SIZE = 10  # Größe der Batches bei der Verarbeitung. Größere Batches können die Geschwindigkeit erhöhen.
PARAGRAPH = False  # Wenn True, versucht EasyOCR, zusammenhängende Textblöcke als Absätze zu erkennen.
ROTATION_INFO = [0, 90, 180, 270]  # Mögliche Rotationswinkel des Textes zur Verbesserung der Erkennung.

# Konfigurationsvariablen für den CRAFT-Textdetektor, der in EasyOCR verwendet wird.
text_threshold = 0.4  # Schwellenwert zur Bestimmung, ob ein Bereich Text enthält.
low_text = 0.5  # Schwellenwert zur Filterung von Text basierend auf der Wahrscheinlichkeit.
link_threshold = 0.2  # Schwellenwert zur Verbindung von Textkomponenten.
canvas_size = 1280  # Maximale Größe der längsten Seite des Bildes für die Verarbeitung.
mag_ratio = 5  # Bildvergrößerungsverhältnis, beeinflusst die Größe der erkannten Textbereiche.

# Funktion zur Überprüfung, ob sich zwei Bounding-Boxen überlappen.
def check_overlap(box1, box2):
    # Überprüft, ob die horizontale oder vertikale Position der ersten Box außerhalb der zweiten Box liegt.
    if box1[0][0] > box2[1][0] or box1[1][0] < box2[0][0] or box1[0][1] > box2[1][1] or box1[1][1] < box2[0][1]:
        return False  # Keine Überlappung.
    return False  # Überlappung vorhanden.

# Funktion zur Verarbeitung eines Bildes für Texterkennung.
def process_image_for_text(image_path):
    image = cv2.imread(image_path)  # Lädt das Bild von dem angegebenen Pfad.
    # Skaliert das Bild, um die Texterkennung zu verbessern.
    scaled_image = cv2.resize(image, None, fx=SCALE_FACTOR, fy=SCALE_FACTOR, interpolation=cv2.INTER_LINEAR)
    
    # Initialisiert den EasyOCR-Reader mit den festgelegten Sprachen und GPU-Einstellungen.
    reader = easyocr.Reader(LANGUAGES, gpu=GPU)

    # Verwendet den EasyOCR-Reader, um Text im skalierten Bild zu erkennen, unter Berücksichtigung der festgelegten Parameter.
    result = reader.readtext(scaled_image, detail=1, decoder=DECODER, paragraph=PARAGRAPH, allowlist=ALLOWLIST, blocklist=BLOCKLIST, workers=WORKERS, batch_size=BATCH_SIZE, rotation_info=ROTATION_INFO,
                             low_text=low_text, link_threshold=link_threshold, canvas_size=canvas_size, mag_ratio=mag_ratio, text_threshold=text_threshold)
    
    return result, image  # Gibt die Erkennungsergebnisse und das Bild zurück.
