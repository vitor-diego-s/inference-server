import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

def run_inference(image_bytes):
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    results = model(image)
    return (results, results[0].tojson())
