
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
mujoco_viewer_f = 60

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
tStep = tFrame = time.monotonic()

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

            t0 = time.monotonic()
            if t0 - tStep > compute_pos.model.opt.timestep:
                mujoco.mj_step(compute_pos.model, compute_pos.data)
                tStep = time.monotonic()
            if t0 - tFrame > 1/mujoco_viewer_f:
                compute_pos.viewer.sync()
                tFrame = time.monotonic()

except KeyboardInterrupt:
    pass

finally:
    if CAMERA_ON:
        cam.release()
        cv2.destroyAllWindows()

    if MUJOCO_ON:
        compute_pos.viewer.close()
