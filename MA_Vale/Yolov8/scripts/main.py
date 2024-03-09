from ultralytics import YOLO
import os

#warnings.filterwarnings("ignore", message="There is an imbalance between your GPUs.*") 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Load a model
model = YOLO('yolov8n-obb.yaml')  # build a new model from YAML
model = YOLO('yolov8n-obb.pt')  # load a pretrained model (recommended for training)
model = YOLO('yolov8n-obb.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

# Train the model
results = model.train(data='RPDS.yaml', epochs=100, imgsz=640, project='/workspace/MA_Vale/Yolov8/results')

