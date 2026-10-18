
import time
import threading
import queue
from pathlib import Path

import cv2
import numpy as np
import mujoco

from computePosition import computePosition
from tools.detectArucos import detectArucos


# Configuration
MUJOCO_ON = True
CAMERA_ON = False
CAMERA_ID = 1
CAMERA_WIDTH = 3840
CAMERA_HEIGHT = 2160

def setup_camera():
    camera_params_path = Path(__file__).parent / "tools" / "cameraParams.npz"
    camera_params = np.load(camera_params_path)

    cam = cv2.VideoCapture(CAMERA_ID)

    if not cam.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la caméra {CAMERA_ID}")

    cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    return cam, camera_params

if CAMERA_ON:
    cam, camera_params = setup_camera()

compute_pos = computePosition(MUJOCO_ON)

detector = None
t0 = time.monotonic()

try:
    while True:
        if CAMERA_ON:
            ret, frame = cam.read()

            if not ret:
                print("Erreur lors de la capture de l'image.")
                break

            detector = detectArucos(camera_params)
            detector.changeImage(frame)
            detector.detect()
            detector.showImage()

            position = compute_pos.detectToRobotPosition(
                detector.detectedArucos,
                detector.detected_ids
            )
            print(position)

        if MUJOCO_ON:
            if not compute_pos.viewer.is_running():
                break

            mujoco.mj_step(compute_pos.model, compute_pos.data)
            compute_pos.viewer.sync()
            t1 = time.monotonic()
            print(t1 - t0)
            t0 = t1

except KeyboardInterrupt:
    pass

finally:
    cam.release()
    cv2.destroyAllWindows()

    if MUJOCO_ON:
        compute_pos.viewer.close()
