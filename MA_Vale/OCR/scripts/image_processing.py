import cv2
import os

def gen_out(image_path, image):
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    if base_name.endswith(".pdf"):
       base_name = base_name[:-4] + ""

    # Gibt Höhe und Breite des Bildes zurück
    image_height, image_width = image.shape[:2]
    return base_name, image_height, image_width

def check_overlap(box1, box2):
    if box1[0][0] > box2[1][0] or box1[1][0] < box2[0][0] or box1[0][1] > box2[1][1] or box1[1][1] < box2[0][1]:
        return False
    return True

def gen_bounding_boxes(results):
    unique_boxes = []

    for bbox, text, prob in results:
        top_left = tuple([int(val) for val in bbox[0]])
        bottom_right = tuple([int(val) for val in bbox[2]])
        unique = True
        for ubox in unique_boxes:
            if check_overlap((top_left, bottom_right), (ubox[0], ubox[1])):
                unique = False
                break
        if unique:
            unique_boxes.append((top_left, bottom_right, text, prob))
    return unique_boxes

def draw_bounding_boxes(image, unique_boxes, output_folder, base_name):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for box_number, (top_left, bottom_right, text, prob) in enumerate(unique_boxes, start=1):
        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
        cv2.putText(image, f"{box_number} ({prob:.2f})", (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    img_path = os.path.join(output_folder, f"{base_name}_bb.jpg")
    cv2.imwrite(img_path, image)
    return 

def write_text(results, output_folder, base_name, image_height, image_width):
    txt_path = os.path.join(output_folder, f"{base_name}.txt")
    
    with open(txt_path, 'w') as txt_file:
        #txt_file.write("Box#\tBoundingBox\n\tText")
        for i, (bbox, text, _) in enumerate(results, start=1):
            # Normalisierung der Koordinaten durch Teilen durch die Bildbreite und -höhe
            normalized_coords = [[x / image_width, y / image_height] for x, y in bbox]
            # Umwandlung der Koordinaten in Strings, Rundung auf 4 Dezimalstellen
            coords_str = ' '.join([' '.join(map(lambda x: f"{x:.4f}", point)) for point in normalized_coords])
            txt_file.write(f"{i}\t{coords_str}\t\"{text}\"\n")
    return txt_path
