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
        self.imagePath = image     
        self.image = plt.imread(image)
        self.image = np.rot90(self.image, 2)

        aruco_dict_4 = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.arucoDetector = aruco.ArucoDetector(aruco_dict_4)
        self.detectedArucos = []
        self.cameraParams = cameraParams
        self.detected_corners = []
        self.detected_ids = []

    def changeImage(self, image):
        self.imagePath = image
        self.image = cv2.imread(image)
        self.detectedArucos = []

    def showImage(self):
        for k in range(len(self.detected_corners)):
            corners_closed = np.vstack([self.detected_corners[k], self.detected_corners[k, 0]])
            plt.plot( 
                corners_closed[:, 0], corners_closed[:, 1], "r-"
            )
            plt.text(corners_closed[0, 0]-20, corners_closed[0, 1]-20, str(self.detected_ids[k]))
            
        plt.imshow(self.image)

    def detect(self):
        detection_corners, self.detected_ids, _ = self.arucoDetector.detectMarkers(self.image)
        print(f"detection_ids : {self.detected_ids}")
        self.detected_corners = np.array(detection_corners).reshape(-1, 4, 2)
        num_detections = len(self.detected_ids)

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
                    self.detected_corners[k], 
                    self.cameraParams['cameraMatrix'], 
                    self.cameraParams['distCoeffs']
            ))

        return(self.detectedArucos)