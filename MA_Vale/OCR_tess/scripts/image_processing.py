import cv2
import os

def draw_bounding_boxes_and_extract_patches(image, boxes, image_path, scale_factor=2.0, output_folder='results', min_confidence=60):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Entfernt die Dateiendung und zusätzlich explizit die .pdf-Endung, falls vorhanden
    base_name = os.path.splitext(os.path.basename(image_path))[0].replace('.pdf', '')

    # Pfad für das Bild mit Bounding-Boxen
    output_path = os.path.join(output_folder, f"{base_name}_bb.jpg")

    # Erstellen und öffnen der Textdatei für das Schreiben
    txt_path = os.path.join(output_folder, f"{base_name}_text_data.txt")
    with open(txt_path, 'w') as txt_file:
        txt_file.write("Nummerierung\tText\n")  # Schreibt die Überschrift

        box_number = 1  # Startet die Nummerierung der Boxen bei 1

        for i in range(len(boxes['text'])):
            if int(boxes['conf'][i]) >= min_confidence:  # Filtert Boxen mit geringer Konfidenz
                (x, y, w, h) = (
                    int(boxes['left'][i] / scale_factor),
                    int(boxes['top'][i] / scale_factor),
                    int(boxes['width'][i] / scale_factor),
                    int(boxes['height'][i] / scale_factor)
                )
                top_left = (x, y)
                bottom_right = (x + w, y + h)

                # Zeichnet die Bounding-Boxen mit Nummerierung auf das Bild
                cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
                cv2.putText(image, str(box_number), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

                # Schreibt die Nummerierung und den erkannten Text in die Textdatei
                txt_file.write(f"Box {box_number}\t{boxes['text'][i]}\n")

                box_number += 1  # Erhöht die Nummerierung für die nächste Box

    # Speichert das bearbeitete Bild
    cv2.imwrite(output_path, image)

    return image, output_path, txt_path
