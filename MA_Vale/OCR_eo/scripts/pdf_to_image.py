# Importiert die notwendige Funktion convert_from_path aus der Bibliothek pdf2image,
# die für die Konvertierung von PDF-Dateien in Bilder verwendet wird,
# sowie das os-Modul für Betriebssystem-interaktionen wie das Auflisten von Dateien in Verzeichnissen.
from pdf2image import convert_from_path
import os

# Definition der Funktion convert_pdf_to_images mit zwei Parametern:
# pdf_dir: das Verzeichnis, in dem die PDF-Dateien gespeichert sind,
# output_dir: das Verzeichnis, in dem die resultierenden Bilder gespeichert werden sollen.
# output_dir hat einen Standardwert 'pdf/img', der verwendet wird, falls beim Aufruf der Funktion kein Wert angegeben wird.
def convert_pdf_to_images(pdf_dir, output_dir='pdf/img'):
    # Initialisierung einer leeren Liste image_paths, die die Pfade der konvertierten Bilder speichern wird.
    image_paths = []

    # Durchläuft alle Dateien im angegebenen pdf_dir-Verzeichnis.
    for filename in os.listdir(pdf_dir):
        # Überprüft, ob die Datei mit '.pdf' endet, um sicherzustellen, dass nur PDF-Dateien verarbeitet werden.
        if filename.endswith('.pdf'):
            # Erstellt den vollständigen Pfad zur PDF-Datei, indem der Verzeichnispfad mit dem Dateinamen kombiniert wird.
            pdf_path = os.path.join(pdf_dir, filename)

            # Verwendet die convert_from_path-Funktion der pdf2image-Bibliothek,
            # um die PDF-Datei in eine Liste von Bildern zu konvertieren.
            # Jede Seite der PDF wird zu einem separaten Bild.
            images = convert_from_path(pdf_path)

            # Durchläuft alle konvertierten Bilder (Seiten der PDF).
            for i, image in enumerate(images):
                # Erstellt einen Pfad für das Bild, indem der Dateiname der PDF und die Seitennummer verwendet werden.
                # Die Seitennummerierung beginnt bei 1.
                image_path = f'{output_dir}/{filename}_{i+1}.jpg'

                # Speichert das Bild im angegebenen Verzeichnis und Format 'JPEG'.
                image.save(image_path, 'JPEG')

                # Fügt den Pfad des gespeicherten Bildes der Liste image_paths hinzu.
                image_paths.append(image_path)

    # Gibt die Liste der Bildpfade zurück, nachdem alle PDF-Dateien verarbeitet wurden.
    return image_paths
