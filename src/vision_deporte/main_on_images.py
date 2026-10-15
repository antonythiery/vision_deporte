import numpy as np
import time
import mujoco
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import threading
import queue

from computePosition import computePosition
from tools.detectArucos import detectArucos

mujocoOn = True
imageId = 7

def get_datas() : 
    image_dir = "datas/exemple_images"
    image_dir = Path(__file__).resolve().parents[2] / "datas/exemple_images"

    images = [
        cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        for f in image_dir.iterdir()
        if f.suffix.lower() in (".png", ".jpg")
    ]
    return images

images = get_datas()



cameraParams = np.load(Path(__file__).parent / "tools/cameraParams.npz")

computePosBase = computePosition(mujocoOn)
detectorBasedeporte = detectArucos(cameraParams)


def process_image():
    detectorBasedeporte.changeImage(images[imageId])
    detectorBasedeporte.detect()
    detectorBasedeporte.showImage()

    print(
        computePosBase.detectToRobotPosition(
            detectorBasedeporte.detectedArucos,
            detectorBasedeporte.detected_ids
        )
    )

commands = queue.Queue()
def read_commands():
    while True:
        command = input(
            "Image suivante [n], précédente [p], quitter [q] : "
        ).lower().strip()
        commands.put(command)
        if command == "q":
            break

threading.Thread(target=read_commands, daemon=True).start()

# viewer.opt.frame = mujoco.mjtFrame.mjFRAME_BODY
try:
    # process_image()

    while computePosBase.viewer.is_running():
        while not commands.empty():
            command = commands.get()

            if command == "q":
                break
            elif command == "n":
                imageId = (imageId + 1) % len(images)
                process_image()
            elif command == "p":
                imageId = (imageId - 1) % len(images)
                process_image()

        else:
            mujoco.mj_step(computePosBase.model, computePosBase.data)
            computePosBase.viewer.sync()
            time.sleep(computePosBase.model.opt.timestep)
            continue

        break

except KeyboardInterrupt:
    pass
finally:
    if mujocoOn:
        computePosBase.viewer.close()
    cv2.destroyAllWindows()