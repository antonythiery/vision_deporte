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
    workdir = "./workdir/"
    board = aruco.CharucoBoard((7, 5), 1, .8, aruco_dict_6)
    imboard = board.generateImage((2000, 2000))
    cv2.imwrite(workdir + "chessboard.tiff", imboard)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.imshow(imboard, cmap = mpl.cm.gray, interpolation = "nearest")
    ax.axis("off")
    plt.imshow(imboard, cmap="gray")
    plt.savefig("workdir/chessboard.png")

printedBoard = aruco.CharucoBoard(
    (7, 5),
    38.4,
    30.8,
    aruco_dict_6
)

# cam resolution max 3840 x 2160

detector = aruco.CharucoDetector(printedBoard)

image_dir = "calibration_images"

images = [
    os.path.join(image_dir, f)
    for f in os.listdir(image_dir)
    if f.endswith(".png") or f.endswith(".jpg")
]

all_charuco_corners = []
all_charucos_ids = []

image_size = None

for k in range(len(images)):
    image = cv2.imread(images[k])
    if(image is None):
        continue

    print(f"image size = {image.shape}")

    charucoDetected = detector.detectBoard(image)

    if charucoDetected[0] is None or len(charucoDetected[0]) < 4:
        continue

    all_charuco_corners.append(charucoDetected[0])
    all_charucos_ids.append(charucoDetected[1])