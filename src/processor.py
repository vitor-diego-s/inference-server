import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

def execute(image_bytes):
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    results = model(image)
    return (results, results[0].to_json())

def download_image_from_link(image_link):
    """
    Downloads an image from a given URL and returns the image bytes.
    """
    import requests
    response = requests.get(image_link)
    if response.status_code == 200:
        return response.content
    else:
        raise Exception(f"Failed to download image: {response.status_code}")