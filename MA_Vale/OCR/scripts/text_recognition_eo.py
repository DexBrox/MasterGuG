import cv2
import easyocr

def process_image_easyocr(image_path):
    image = cv2.imread(image_path)
    reader = easyocr.Reader(['de', 'en'], gpu=True)
    results = reader.readtext(image, rotation_info=[0, 90, 180, 270])
    return results, image
