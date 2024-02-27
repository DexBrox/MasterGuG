import cv2
import pytesseract

def process_image_tess(image_path):
    image = cv2.imread(image_path)

    conf = 0.0

    config = '--oem 3 --psm 1 -l deu+eng'
    boxes = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)

    results = []
    n_boxes = len(boxes['level'])
    for i in range(n_boxes):
        if boxes['text'][i].strip() != '' and float(boxes['conf'][i]) > conf*100: 
            (x, y, w, h) = (boxes['left'][i], boxes['top'][i], boxes['width'][i], boxes['height'][i])
            box_coords = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
            results.append((box_coords, boxes['text'][i], float(boxes['conf'][i])/100))
            
    return results, image
