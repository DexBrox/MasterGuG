import cv2
import easyocr

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
    image = cv2.imread(image_path)
    scaled_image = cv2.resize(image, None, fx=SCALE_FACTOR, fy=SCALE_FACTOR, interpolation=cv2.INTER_LINEAR)
    reader = easyocr.Reader(LANGUAGES, gpu=GPU)
    result = reader.readtext(scaled_image, detail=1, decoder=DECODER, paragraph=PARAGRAPH, allowlist=ALLOWLIST, blocklist=BLOCKLIST, workers=WORKERS, batch_size=BATCH_SIZE, rotation_info=ROTATION_INFO,
                             low_text=low_text, link_threshold=link_threshold, canvas_size=canvas_size, mag_ratio=mag_ratio, text_threshold=text_threshold)
    
    return result, image
