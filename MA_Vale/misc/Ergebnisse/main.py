from ultralytics import YOLO

# Erstellen Sie ein neues YOLO Modell von Grund auf
model = YOLO('yolov8n.yaml')

# Laden Sie ein vortrainiertes YOLO Modell (empfohlen für das Training)
model = YOLO('yolov8n.pt')

# Trainieren Sie das Modell mit dem Datensatz 'coco128.yaml' für 3 Epochen
results = model.train(data='coco128.yaml', epochs=3)

# Bewerten Sie die Leistung des Modells am Validierungssatz
results = model.val()

# Führen Sie eine Objekterkennung an einem Bild mit dem Modell durch
results = model('https://ultralytics.com/images/bus.jpg')

# Exportieren Sie das Modell ins ONNX-Format
success = model.export(format='onnx')