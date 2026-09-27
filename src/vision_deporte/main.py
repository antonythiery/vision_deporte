import numpy as np
import os
import time
import mujoco
import mujoco.viewer

from tools.createScene import build_xml
from tools.detectArucos import detectArucos

## detect arucos irl

image_dir = "datas/calibration_images"

images = [
    os.path.join(image_dir, f)
    for f in os.listdir(image_dir)
    if f.endswith(".png") or f.endswith(".jpg")
]

cameraParams = np.load("src/vision_deporte/tools/cameraParams.npz")
detectorBasedeporte = detectArucos(images[7], cameraParams)

detection1 = detectorBasedeporte.detect()

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
    
    model.body_pos[idArucoTable[k]] = posAruco[1] / 1000.0
    # model.body_quat[idArucoTable[k]] = posAruco[0]
mujoco.mj_forward(model, data)

try :
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)
except KeyboardInterrupt: 
    pass
finally :
    viewer.close()