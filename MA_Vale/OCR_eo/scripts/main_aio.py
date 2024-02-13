import timeit
start_in_import = timeit.default_timer()
start_def = timeit.default_timer()
import os
import warnings
import cv2
import easyocr
from pdf2image import convert_from_path
end_in_import = timeit.default_timer()

start_imp = timeit.default_timer()

warnings.filterwarnings("ignore", message="There is an imbalance between your GPUs.*") 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def gen_out(image_path, output_folder):
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_folder, f"{base_name}_bb.jpg")

    return output_path, base_name

def check_overlap(box1, box2):
    if box1[0][0] > box2[1][0] or box1[1][0] < box2[0][0] or box1[0][1] > box2[1][1] or box1[1][1] < box2[0][1]:
        return False
    
    return True

def gen_bounding_boxes(results, scale_factor):
    unique_boxes = []

    for bbox, text, prob in results:
        top_left = tuple([int(val / scale_factor) for val in bbox[0]])
        bottom_right = tuple([int(val / scale_factor) for val in bbox[2]])
        unique = True
        for ubox in unique_boxes:
            if check_overlap((top_left, bottom_right), (ubox[0], ubox[1])):
                unique = False
                break
        if unique:
            unique_boxes.append((top_left, bottom_right, text, prob))

    return unique_boxes

def draw_bounding_boxes(image, unique_boxes, output_folder, output_path):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for box_number, (top_left, bottom_right, text, prob) in enumerate(unique_boxes, start=1):
        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
        cv2.putText(image, f"{box_number} ({prob:.2f})", (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imwrite(output_path, image)

    return 

def write_text(results, output_folder, base_name):
    
    txt_path = os.path.join(output_folder, f"{base_name}_text_data.txt")
    
    with open(txt_path, 'w') as txt_file:
        txt_file.write("Box#\tText\tBoundingBox\n")
        for i, (bbox, text, _) in enumerate(results, start=1):
            coords = ' '.join([' '.join(map(str, map(int, point))) for point in bbox])
            txt_file.write(f"{i}\t{text}\t{coords}\n")

    return txt_path

def convert_pdf_to_images(pdf_dir, output_dir):
    image_paths = []
    for filename in os.listdir(pdf_dir):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(pdf_dir, filename)
            images = convert_from_path(pdf_path)
            for i, image in enumerate(images):
                image_path = f'{output_dir}/{filename}_{i+1}.jpg'
                image.save(image_path, 'JPEG')
                image_paths.append(image_path)
    return image_paths

SCALE_FACTOR = 2.0
LANGUAGES = ['de', 'en']
DECODER = 'greedy'
GPU = True
ALLOWLIST = None
BLOCKLIST = ['!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '_', '+', '=', '{', '}', '[', ']', '|', '\\', ':', ';', '"', "'", '<', '>', ',', '.', '?', '/']
WORKERS = 10
BATCH_SIZE = 10
PARAGRAPH = False
ROTATION_INFO = [0, 90, 180, 270]
text_threshold = 0.4
low_text = 0.5
link_threshold = 0.2
canvas_size = 1280
mag_ratio = 5

def process_image_easyocr(image_path):
    start_img_pre = timeit.default_timer()
    image = cv2.imread(image_path)
    scaled_image = cv2.resize(image, None, fx=SCALE_FACTOR, fy=SCALE_FACTOR, interpolation=cv2.INTER_LINEAR)
    reader = easyocr.Reader(LANGUAGES, gpu=GPU)
    end_img_pre = timeit.default_timer()
    print(f"OCR-Vorbereitungszeit: {end_img_pre - start_img_pre} Sekunden")

    start_tim_results = timeit.default_timer()
    result = reader.readtext(image, detail=1, decoder=DECODER, paragraph=PARAGRAPH, allowlist=ALLOWLIST, blocklist=BLOCKLIST, workers=WORKERS, batch_size=BATCH_SIZE, rotation_info=ROTATION_INFO,
                             low_text=low_text, link_threshold=link_threshold, canvas_size=canvas_size, mag_ratio=mag_ratio, text_threshold=text_threshold)
    end_tim_results = timeit.default_timer()
    print(f"OCR-Verarbeitungszeit: {end_tim_results - start_tim_results} Sekunden")

    return result, image

end_imp = timeit.default_timer()
start_pdf = timeit.default_timer()

pdf_dir = '../pdf'
output_dir = '../pdf/img'
image_paths = convert_pdf_to_images(pdf_dir, output_dir)

end_pdf = timeit.default_timer()

output_folder = '../results'
scale_factor = 2.0

for image_path in image_paths:
    output_path, base_name = gen_out(image_path, output_folder)

    start_img = timeit.default_timer()
    results, image = process_image_easyocr(image_path)
    end_img = timeit.default_timer()

    unique_boxes = gen_bounding_boxes(results, scale_factor)

    #base_name = draw_bounding_boxes(image, unique_boxes, output_folder, output_path)

    write_text(results, output_folder, base_name)

end_def = timeit.default_timer()

print(f"Initiale Importzeit: {end_in_import - start_in_import} Sekunden")
print(f"Importzeit: {end_imp - start_imp} Sekunden")
print(f"PDF-Konvertierungszeit: {end_pdf - start_pdf} Sekunden")
print(f"Bildverarbeitungszeit: {end_img - start_img} Sekunden")

print(f"Gesamt: {end_def - start_def} Sekunden")

