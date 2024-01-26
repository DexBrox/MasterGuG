FROM ultralytics/ultralytics:latest
# Umgebungsvariablen setzen
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID
ENV CUDA_VISIBLE_DEVICES=0,1,2,3
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
ENV PATH=/usr/local/cuda/bin:$PATH
# Keine GUI-Interaktion waehrend der Installation:
ARG DEBIAN_FRONTEND=noninteractive
# Working Directory
WORKDIR /workspace
# Wird gebraucht, damit die packages gefunden werden
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

