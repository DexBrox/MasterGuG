import cv2
import os
from text_recognition import check_overlap

def draw_bounding_boxes_and_extract_patches(image, results, image_path, scale_factor=2.0, output_folder='results'):
    # Stellt sicher, dass der Output-Ordner existiert.
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    unique_boxes = []
    box_number = 1  # Startet die Nummerierung der Boxen bei 1

    for bbox, text, prob in results:
        top_left = tuple([int(val / scale_factor) for val in bbox[0]])
        bottom_right = tuple([int(val / scale_factor) for val in bbox[2]])
        unique = True
        
        for ubox in unique_boxes:
            if check_overlap((top_left, bottom_right), (ubox[0], ubox[1])):
                unique = False
                break
        
        if unique:
            unique_boxes.append((top_left, bottom_right, 'green', prob))

    # Zeichnet die Bounding-Boxen mit Nummerierung und Wahrscheinlichkeit auf das Bild
    for box in unique_boxes:
        top_left, bottom_right, color, prob = box
        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
        cv2.putText(image, f"{box_number} ({prob:.2f})", (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        box_number += 1  # Erhöht die Nummerierung für die nächste Box

    # Entfernt die .pdf-Endung und fügt eine Kennzeichnung hinzu
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_folder, f"{base_name}_bb.jpg")
    
    # Speichert das bearbeitete Bild
    cv2.imwrite(output_path, image)

    # Erstellen und öffnen der Textdatei für das Schreiben
    txt_path = os.path.join(output_folder, f"{base_name}_text_data.txt")
    with open(txt_path, 'w') as txt_file:
        # Schreibt die Überschrift: Nummerierung und Bildname
        txt_file.write(f"Nummerierung\tText\n")
        
        # Schreibt die Nummerierung und den erkannten Text in die Textdatei
        for i, (_, text, _) in enumerate(results, start=1):
            txt_file.write(f"Box {i}\t{text}\n")

    # Rückgabe der ursprünglichen Rückgabewerte plus des Pfads zur Textdatei
    return image, unique_boxes, output_path, txt_path
