import numpy as np
import cv2, PIL, os 
from cv2 import aruco
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import os

aruco_dict_6 = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

def create_charuco_board():
    workdir = "./datas/charuco/"
    board = aruco.CharucoBoard((7, 5), 1, .8, aruco_dict_6)
    imboard = board.generateImage((2000, 2000))
    cv2.imwrite(workdir + "chessboard.tiff", imboard)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.imshow(imboard, cmap = mpl.cm.gray, interpolation = "nearest")
    ax.axis("off")
    plt.imshow(imboard, cmap="gray")
    plt.savefig("datas/charuco/chessboard.png")

printedBoard = aruco.CharucoBoard(
    (7, 5),
    38.4,
    30.8,
    aruco_dict_6
)
charuco3D = printedBoard.getChessboardCorners()

# cam resolution max 3840 x 2160

detector = aruco.CharucoDetector(printedBoard)

image_dir = "datas/calibration_images"

images = [
    os.path.join(image_dir, f)
    for f in os.listdir(image_dir)
    if f.endswith(".png") or f.endswith(".jpg")
]

all_charuco_corners = []
all_charucos_ids = []
all_object_points = []
all_image_points = []

image_size = None

for k in range(len(images)):
    image = cv2.imread(images[k])
    if(image is None):
        continue

    image_size = (image.shape[1], image.shape[0])
    charucoDetected = detector.detectBoard(image)
    charucos_corners = charucoDetected[0]
    charucos_ids = charucoDetected[1]

    if charucos_corners is None or len(charucos_corners) < 4:
        continue

    object_points = charuco3D[charucos_ids]
    image_points = charucos_corners

    all_charuco_corners.append(charucos_corners)
    all_charucos_ids.append(charucos_ids)
    all_object_points.append(object_points)
    all_image_points.append(image_points)


RMS, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
    all_object_points, 
    all_image_points, 
    image_size,
    None,
    None
)

print(f"RMS : {RMS} \n cameraMatrix : \n{cameraMatrix} \n distCoeffs : {distCoeffs}")

np.savez("src/vision_deporte/cameraParams", RMS=RMS, cameraMatrix=cameraMatrix, distCoeffs=distCoeffs)