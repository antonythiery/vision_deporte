import numpy as np
import time
import mujoco
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from pathlib import Path
import cv2

from computePosition import computePosition
from tools.detectArucos import detectArucos

mujocoOn = False


cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("Erreur ouverture cam")
    exit()

cam.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)

cameraParams = np.load(Path(__file__).parent / "tools/cameraParams.npz")


while True : 
    ret, frame = cam.read()
    if not ret:
        print("erreur capture")
        break
    
    
    detectorBasedeporte = detectArucos(frame, cameraParams)
    # detectorBasedeporte.detect()
    detectorBasedeporte.showImage()

    # computePosBase = computePosition(mujocoOn)


    # cv2.imshow("cam", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()