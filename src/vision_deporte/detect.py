import numpy as np
import os

from detectArucos import detectArucos

image_dir = "datas/calibration_images"

images = [
    os.path.join(image_dir, f)
    for f in os.listdir(image_dir)
    if f.endswith(".png") or f.endswith(".jpg")
]

cameraParams = np.load("src/vision_deporte/cameraParams.npz")
detectorBasedeporte = detectArucos(images[1], cameraParams)

print(detectorBasedeporte.detect())
