import cv2
import easyocr
import os
from pdf2image import convert_from_path
from pathlib import Path
import numpy as np

def rotate_image(image, angle):
    """ Rotiert das Bild um den angegebenen Winkel. """
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated, M

def transform_back(bbox, M):
    """ Transformiert eine Bounding-Box zurück in das Koordinatensystem des Originalbildes. """
    points = np.array([[bbox[0][0], bbox[0][1]], [bbox[1][0], bbox[1][1]], 
                       [bbox[2][0], bbox[2][1]], [bbox[3][0], bbox[3][1]]])
    ones = np.ones(shape=(len(points), 1))
    points_ones = np.hstack([points, ones])
    transformed_points = np.dot(np.linalg.inv(M), points_ones.T).T
    transformed_points = transformed_points[:, :2].astype(int)
    return transformed_points

def overlap_or_contained(box1, box2):
    """ Überprüft, ob sich zwei Bounding-Boxen überschneiden oder eine in der anderen enthalten ist. """
    x_overlap = max(0, min(box1[1][0], box2[1][0]) - max(box1[0][0], box2[0][0]))
    y_overlap = max(0, min(box1[1][1], box2[1][1]) - max(box1[0][1], box2[0][1]))
    overlap_area = x_overlap * y_overlap
    return overlap_area > 0 or overlap_area == (box1[1][0] - box1[0][0]) * (box1[1][1] - box1[0][1]) or overlap_area == (box2[1][0] - box2[0][0]) * (box2[1][1] - box2[0][1])

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

input_dir = 'Data/Roewaplan'
output_dir = 'Results/Roewaplan'
images_dir = os.path.join('Data', 'Images')
patches_dir = os.path.join(output_dir, 'Patches')

os.makedirs(output_dir, exist_ok=True)
os.makedirs(images_dir, exist_ok=True)
os.makedirs(patches_dir, exist_ok=True)

reader = easyocr.Reader(['de', 'en'])

pdf_files = [f for f in Path(input_dir).glob('*.pdf')]
image_count = 1

for pdf_path in pdf_files:
    images = convert_from_path(str(pdf_path))
    for page_num, image in enumerate(images):
        image_path = os.path.join(images_dir, f'{pdf_path.stem}_page_{page_num}.jpg')
        image.save(image_path, 'JPEG')
        image = cv2.imread(image_path)

        scale_factor = 2
        scaled_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

        result_original = reader.readtext(scaled_image, detail=1)

        rotated_image, M = rotate_image(scaled_image, 90)
        result_rotated = reader.readtext(rotated_image, detail=1)

        M = np.vstack([M, [0, 0, 1]])

        filtered_rotated_results = []
        for bbox_rotated, text_rotated, prob_rotated in result_rotated:
            transformed_bbox_rotated = transform_back(bbox_rotated, M)
            if not any(overlap_or_contained(transformed_bbox_rotated, bbox_original[:2]) for bbox_original, _, _ in result_original):
                filtered_rotated_results.append((transformed_bbox_rotated, text_rotated, prob_rotated))

        unique_results = result_original[:]
        for bbox, text, prob in filtered_rotated_results:
            if not any(overlap_or_contained(bbox, original_bbox[:2]) for original_bbox, _, _ in unique_results):
                unique_results.append((bbox, text, prob))

        bbox_number = 1
        for bbox, text, prob in unique_results:
            pts = np.array([[(int(point[0] / scale_factor), int(point[1] / scale_factor)) for point in
