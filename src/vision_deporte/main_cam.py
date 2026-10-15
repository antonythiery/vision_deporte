
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
CAMERA_ID = 1
CAMERA_WIDTH = 3840
CAMERA_HEIGHT = 2160

# Charger les paramètres de caméra
camera_params_path = Path(__file__).parent / "tools" / "cameraParams.npz"
camera_params = np.load(camera_params_path)

# Initialiser la caméra
cam = cv2.VideoCapture(CAMERA_ID)

if not cam.isOpened():
    raise RuntimeError(f"Impossible d'ouvrir la caméra {CAMERA_ID}")

cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

# Initialiser MuJoCo une seule fois
compute_pos = computePosition(MUJOCO_ON)

# File des commandes du terminal
commands = queue.Queue()


def read_commands():
    """Lit les commandes sans bloquer la boucle caméra."""
    while True:
        try:
            command = input(
                "\nEntrée : continuer | q : quitter\n"
            ).strip().lower()
        except EOFError:
            commands.put("q")
            break

        commands.put(command)

        if command == "q":
            break


threading.Thread(target=read_commands, daemon=True).start()


# Initialiser la fenêtre OpenCV une seule fois
cv2.namedWindow("Camera - detection ArUco", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Camera - detection ArUco", 1280, 720)

detector = None

try:
    while True:
        # Vérifier les commandes du terminal
        should_quit = False

        while not commands.empty():
            command = commands.get()

            if command == "q":
                should_quit = True
                break

        if should_quit:
            break

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

        # Actualiser MuJoCo
        if MUJOCO_ON:
            if not compute_pos.viewer.is_running():
                break

            mujoco.mj_step(compute_pos.model, compute_pos.data)
            compute_pos.viewer.sync()

        # Traiter les événements OpenCV et permettre de quitter avec q
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

except KeyboardInterrupt:
    pass

finally:
    cam.release()
    cv2.destroyAllWindows()

    if MUJOCO_ON:
        compute_pos.viewer.close()
