import os

# Set environment variable for CUDA (if necessary)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Load a pretrained YOLOv9 model
model = YOLO('path/to/yolov9/pretrained/model.pt')

# Train the model on your dataset
results = model.train(data='RPDS.yaml', epochs=100, imgsz=640, project='/path/to/save/results')
