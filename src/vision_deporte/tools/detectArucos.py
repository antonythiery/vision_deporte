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

arucoRobotcote = 70 #mm
arucoRobot = np.array([
    [-arucoRobotcote/2, arucoRobotcote/2, 0.0], 
    [arucoRobotcote/2, arucoRobotcote/2, 0.0], 
    [arucoRobotcote/2, -arucoRobotcote/2, 0.0], 
    [-arucoRobotcote/2, -arucoRobotcote/2, 0.0]
])

arucoDirectory = {
    "2" : arucoRobot,
    "20" : arucoTable,
    "21" : arucoTable,
    "22" : arucoTable,
    "23" : arucoTable
}


class detectArucos:
    """detection des arucos dans une image"""

    def __init__(self, image, cameraParams):
        self.image = image
        self.image = cv2.rotate(self.image, cv2.ROTATE_180)

        aruco_dict_4 = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.arucoDetector = aruco.ArucoDetector(aruco_dict_4)
        self.detectedArucos = []
        self.cameraParams = cameraParams
        self.detected_corners = []
        self.detected_ids = []

    def changeImage(self, image):
        self.image = image
        self.image = cv2.rotate(self.image, cv2.ROTATE_180)
        self.detectedArucos = []

    def showImage(self):
        for k in range(len(self.detected_corners)):
            corners_closed = np.vstack([self.detected_corners[k], self.detected_corners[k, 0]])
            pts = corners_closed.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines( 
                self.image, [pts], isClosed=False, color=(0, 0, 255), thickness=2
            )
            cv2.putText(
                self.image, 
                str(self.detected_ids[k]),
                (int(corners_closed[0, 0]-20), int(corners_closed[0, 1]-20)), 
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2,color=(0, 0, 255)
            )

        cv2.imshow("cam", cv2.resize(self.image, (1280, 720)))
        cv2.waitKey(1)

    def detect(self):
        detection_corners, self.detected_ids, _ = self.arucoDetector.detectMarkers(self.image)
        self.detected_corners = np.array(detection_corners).reshape(-1, 4, 2)

        for k in range(len(self.detected_ids)):
            self.detectedArucos.append( 
                cv2.solvePnP(
                    arucoDirectory[str(self.detected_ids[k])], 
                    self.detected_corners[k], 
                    self.cameraParams['cameraMatrix'], 
                    self.cameraParams['distCoeffs'],
                    flags=cv2.SOLVEPNP_IPPE_SQUARE
            ))
        return(self.detectedArucos)