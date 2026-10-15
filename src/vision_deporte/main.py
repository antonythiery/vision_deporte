import numpy as np
import os
import time
import mujoco
import mujoco.viewer
import cv2
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from pathlib import Path
import xml.etree.ElementTree as ET

from computePosition import computePosition
from tools.detectArucos import detectArucos
from tools.utils import mean_robust, rotation_kabsch

imageId = 7


def get_datas() : 
    image_dir = "datas/exemple_images"
    image_dir = Path(__file__).resolve().parents[2] / "datas/exemple_images"

    images = [
        f
        for f in image_dir.iterdir()
        if f.suffix.lower() in (".png", ".jpg")
    ]
    return images

images = get_datas()

cameraParams = np.load(Path(__file__).parent / "tools/cameraParams.npz")
detectorBasedeporte = detectArucos(images[imageId], cameraParams)
detectorBasedeporte.detect()
detectorBasedeporte.showImage()

computePosBase = computePosition()

mujocoOn = True
if(mujocoOn):
    computePosBase.setup_mujoco()
    computePosBase.find_robot_mujoco()
    print(computePosBase.idArucoRobot)
    print(computePosBase.detectToRobotPosition(detectorBasedeporte.detectedArucos, detectorBasedeporte.detected_ids))
    computePosBase.update_mujoco()
else :
    print(computePosBase.detectToRobotPosition())

fig = plt.gcf()
def on_key(event):
    global imageId
    if(event.key=="right" or event.key=="left"):
        if(event.key=="right"):
            imageId+=1
        else:
            imageId-=1
        imageId = imageId % (len(images))
        t0 = time.perf_counter()
        detectorBasedeporte.changeImage(images[imageId])
        detectorBasedeporte.detect()
        detectorBasedeporte.showImage()
        fig.canvas.draw_idle()
        print(computePosBase.detectToRobotPosition(detectorBasedeporte.detectedArucos, detectorBasedeporte.detected_ids))
        if(mujocoOn):
            computePosBase.update_mujoco()
        t1 = time.perf_counter()
        print(f"temps exec : {t1-t0:6f}")

manager = plt.get_current_fig_manager()
manager.window.setGeometry(0, 0, 960, 1080)
fig.canvas.mpl_connect("key_press_event", on_key)
plt.show(block=False)

# viewer.opt.frame = mujoco.mjtFrame.mjFRAME_BODY
try :
    while computePosBase.viewer.is_running():
        mujoco.mj_step(computePosBase.model, computePosBase.data)
        computePosBase.viewer.sync()
        time.sleep(computePosBase.model.opt.timestep)
        fig.canvas.flush_events()
except KeyboardInterrupt: 
    pass
finally :
    computePosBase.viewer.close()
    plt.close("all")