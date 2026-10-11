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

from tools.createScene import build_xml
from tools.detectArucos import detectArucos

posCam = [-0.1, -0.5, 1.8]
T_world_cam = np.array([
    [-0.98464269, 0.02605037, 0.17262722],
    [-0.05371072, 0.89564419, -0.4415163 ],
    [-0.16611423, -0.44400773, -0.88049032]
])


Rz = np.array([
    [np.cos(np.pi/2), np.sin(np.pi/2), 0],
    [-np.sin(np.pi/2),  np.cos(np.pi/2), 0],
    [0,                0,               1]
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


models_dir = Path(__file__).parent / "models"
xml = build_xml(models_dir / "scene.xml")
tmp = models_dir / "_scene_runtime.xml"
tmp.write_text(xml)

model = mujoco.MjModel.from_xml_path(str(tmp))
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

idCam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,"cam")

idTable = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,"table_sans_bordure")

posTable = np.array([0.0, 0.0, 0.0])
qposTable = np.array([0.0, 0.0, 0.0, 0.0])
posArucoTable = {
    "20" : [0.5, -0.9, 0.0],
    "21" : [0.5, 0.9, 0.0],
    "22" : [1.4, -0.9, 0.0],
    "23" : [1.4, 0.9, 0.0]    
 } # positon du centre des arucos sur la table ordre (repère STK) [m]
nbArucoTableDetected = 0

for k in range(len(detection1)):
    idAruco = detectorBasedeporte.detected_ids[k]
    posAruco = detection1[k]
    posAruco = np.array(posAruco[1:]).reshape(2, 3)
    R = cv2.Rodrigues(posAruco[0])[0]
    qposAruco = Rotation.from_matrix(T_world_cam @ R @ Rz).as_quat(scalar_first=True)
    posArucoCam = posAruco[1] / 1000.0
    model.body_pos[idArucoTable[k]] = T_world_cam @ posArucoCam @ Rz + posCam

    model.body_quat[idArucoTable[k]] = qposAruco

    if(20<=idAruco<=23):
        nbArucoTableDetected += 1
        posTable += model.body_pos[idArucoTable[k]] - posArucoTable[str(idAruco)]
        qposTable += model.body_quat[idArucoTable[k]]
        print(f"posTable : {posTable}")

posTable /= nbArucoTableDetected
qposTable /= nbArucoTableDetected
print(f"posTablefinal : {posTable}")
model.body_pos[idTable] = posTable
model.body_quat[idTable] = qposTable

model.body_pos[idCam] = posCam

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