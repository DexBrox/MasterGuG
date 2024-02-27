import cv2
import pytesseract
import easyocr
import numpy as np

def process_image_hy(image_path):
    image = cv2.imread(image_path)
    reader = easyocr.Reader(['de', 'en'], gpu=True)
    results = reader.readtext(image, rotation_info=[0, 90, 180, 270])

    for i, (bbox, _, _) in enumerate(results):
        (top_left, _, bottom_right, _) = bbox
        top_left = (max(int(top_left[0]), 0), max(int(top_left[1]), 0))
        bottom_right = (min(int(bottom_right[0]), image.shape[1] - 1), min(int(bottom_right[1]), image.shape[0] - 1))

        if top_left[0] < bottom_right[0] and top_left[1] < bottom_right[1]:
            cropped_image = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
            pytesseract_text = pytesseract.image_to_string(cropped_image, lang='deu+eng')

            results[i] = (bbox, pytesseract_text, results[i][2])
        else:
            print(f"Fehler in {i}.")

    return results, image
