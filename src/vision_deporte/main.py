import numpy as np
import cv2, PIL, os 
from cv2 import aruco
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import os

arucoTable = np.array([
    [-50.0, 50.0, 0.0], 
    [50.0, 50.0, 0.0], 
    [50.0, -50.0, 0.0], 
    [-50.0, -50.0, 0.0]
])

aruco_dict_4 = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
arucoDetector = aruco.ArucoDetector(aruco_dict_4)

image_dir = "datas/calibration_images"

images = [
    os.path.join(image_dir, f)
    for f in os.listdir(image_dir)
    if f.endswith(".png") or f.endswith(".jpg")
]

image = cv2.imread(images[0])

detection_corners, detection_ids, _ = arucoDetector.detectMarkers(image)
detection_corners = np.array(detection_corners).reshape(-1, 4, 2)
num_detections = len(detection_ids)
centers = []
for k in range(num_detections):
    center = np.sum(detection_corners[k], axis=0)
    center /= 4
    centers.append(center)

centers = np.array(centers)

cameraParams = np.load("src/vision_deporte/cameraParams.npz")

detectedArucoTable = []
for k in range(num_detections):
    detectedArucoTable.append(
        cv2.solvePnP(
            arucoTable, 
            detection_corners[k], 
            cameraParams['cameraMatrix'], 
            cameraParams['distCoeffs']
    ))

print(detectedArucoTable)