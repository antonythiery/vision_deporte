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

from tools.createScene import build_xml
from tools.detectArucos import detectArucos

posCam = [0.0, 0.0, 1.8]
T_world_cam = np.array([
    [-0.98464269, 0.02605037, 0.17262722],
    [-0.05371072, 0.89564419, -0.4415163 ],
    [-0.16611423, -0.44400773, -0.88049032]
])


## detect arucos irl

image_dir = "datas/calibration_images"

images = [
    os.path.join(image_dir, f)
    for f in os.listdir(image_dir)
    if f.endswith(".png") or f.endswith(".jpg")
]

cameraParams = np.load("src/vision_deporte/tools/cameraParams.npz")
detectorBasedeporte = detectArucos(images[3], cameraParams)

detection1 = detectorBasedeporte.detect()

detectorBasedeporte.showImage()

## setup 3D env

xml = build_xml("src/vision_deporte/models/robot.xml")

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

mujoco.mj_kinematics(model, data)

viewer = mujoco.viewer.launch_passive(
    model = model,
    data = data,
    show_left_ui=False,
    show_right_ui=False,
)

## runtime

idArucoTable = []
idArucoTable.append(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,"arucoTable0"))
idArucoTable.append(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,"arucoTable1"))
idArucoTable.append(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,"arucoTable2"))
idArucoTable.append(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,"arucoTable3"))

for k in range(len(detection1)):
    posAruco = detection1[k]
    posAruco = np.array(posAruco[1:]).reshape(2, 3)
    R = cv2.Rodrigues(posAruco[0])[0]
    qposAruco = Rotation.from_matrix(T_world_cam @ R).as_quat(scalar_first=True)
    posArucoCam = posAruco[1] / 1000.0
    model.body_pos[idArucoTable[k]] = T_world_cam @ posArucoCam + posCam
    model.body_quat[idArucoTable[k]] = qposAruco
mujoco.mj_forward(model, data)

manager = plt.get_current_fig_manager()
manager.window.setGeometry(0, 0, 960, 1080)

try :
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)

        plt.show()
except KeyboardInterrupt: 
    pass
finally :
    viewer.close()