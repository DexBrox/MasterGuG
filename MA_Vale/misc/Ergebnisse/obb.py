import subprocess
from ultralytics import YOLO

# Load YOLO model and process image
model = YOLO('yolov8n-obb.pt')
results = model('https://content.wirkaufendeinauto.de/static/car_images/mythen-im-strassenverkehr.jpg', save=True, device=1)

# Extract the path where YOLO saves its output 
output_path = "/usr/src/ultralytics/runs/obb/predict6"

# Define your local destination path
local_path = "./MA_Roewaplan/Ergebnisse"

# Docker container name
container_name = "ma_jung"

# Beispiel: Angenommen, der Pfad zu Docker ist "/usr/bin/docker"
docker_path = "/usr/bin/docker"

# Verwenden Sie diesen Pfad im Befehl
command = f"{docker_path} cp {container_name}:{output_path} {local_path}"
subprocess.run(command, shell=True)




