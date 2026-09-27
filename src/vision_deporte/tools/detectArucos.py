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

class detectArucos:
    """detection des arucos dans une image"""

    def __init__(self, image, cameraParams):        
        self.image = cv2.imread(image)

        aruco_dict_4 = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.arucoDetector = aruco.ArucoDetector(aruco_dict_4)
        self.detectedArucos = []
        self.cameraParams = cameraParams

    def changeImage(self, image):
        self.image = cv2.imread(image)
        self.detectedArucos = []

    def detect(self):
        detection_corners, detection_ids, _ = self.arucoDetector.detectMarkers(self.image)
        print(f"detection_ids : {detection_ids}")
        detection_corners = np.array(detection_corners).reshape(-1, 4, 2)
        num_detections = len(detection_ids)

        # centers = []
        # for k in range(num_detections):
        #     center = np.sum(detection_corners[k], axis=0)
        #     center /= 4
        #     centers.append(center)

        # centers = np.array(centers)

        for k in range(num_detections):
            self.detectedArucos.append(
                cv2.solvePnP(
                    arucoTable, 
                    detection_corners[k], 
                    self.cameraParams['cameraMatrix'], 
                    self.cameraParams['distCoeffs']
            ))

        return(self.detectedArucos)