from ultralytics import YOLO
from PIL import Image
import warnings
import os

#warnings.filterwarnings("ignore", message="There is an imbalance between your GPUs.*") 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Create a new YOLO model from scratch
model = YOLO('yolov8n.yaml')

# Load a pretrained YOLO model (recommended for training)
model = YOLO('yolov8n.pt')

# Train the model using the 'coco128.yaml' dataset for 3 epochs
#results = model.train(data='coco128.yaml', epochs=3)
results = model.train(data='coco128.yaml', epochs=1, project='/workspace/MA_Vale/Yolov8/results')

# Evaluate the model's performance on the validation set
results = model.val()

# Perform object detection on an image using the model
results = model('img/stadt.jpg')
print ('Jetzt kommen die Results des img')
print (results)
print ('Das waren die Results des img')
# show image
results[0].show()

# Rendern Sie die Erkennungen auf dem Bild
rendered_images = results.render()  # Dies gibt eine Liste von Bildern im NumPy-Array-Format zurück

# Speichern Sie das erste Bild (wenn Sie nur ein Bild zur Erkennung eingegeben haben)
img = Image.fromarray(rendered_images[0])
save_path = 'results/test_image_with_detections.jpg'  # Pfad, unter dem das annotierte Bild gespeichert werden soll
img.save(save_path)
print(f"Bild mit Annotationen gespeichert unter: {save_path}")

# Export the model to ONNX format
success = model.export(format='onnx')