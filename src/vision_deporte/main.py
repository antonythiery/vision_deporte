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
from tools.utils import mean_robust

posCam = [-0.1, -0.5, 1.8]
T_world_cam_old = np.array([
    [-0.98464269, 0.02605037, 0.17262722],
    [-0.05371072, 0.89564419, -0.4415163 ],
    [-0.16611423, -0.44400773, -0.88049032]
])

T_world_cam = np.array([
    [-0.97677887,  0.0699096 , -0.2025233 ],
    [ 0.17033678,  0.82675706, -0.53615123],
    [ 0.12995545, -0.55819836, -0.81946701]
])

posArucoRobot = [0.04, 0.06, 0.4504]
posArucoTable = {
    "20" : [-0.9, -0.5, 0.0],
    "21" : [0.9, -0.5, 0.0],
    "22" : [-0.9, -1.4, 0.0],
    "23" : [0.9, -1.4, 0.0]    
} # positon du centre des arucos sur la table ordre (repère STK) [m]

Rzn = Rotation.from_euler('z', -np.pi/2)
Rzp = Rotation.from_euler('z', np.pi/2)

## detect arucos irl

image_dir = "datas/exemple_images"

images = [
    os.path.join(image_dir, f)
    for f in os.listdir(image_dir)
    if f.endswith(".png") or f.endswith(".jpg")
]

cameraParams = np.load("src/vision_deporte/tools/cameraParams.npz")
detectorBasedeporte = detectArucos(images[10], cameraParams)

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

arucoIdtoObjectId = {
    "20" : mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,"arucoTable0"),
    "21" : mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,"arucoTable1"),
    "22" : mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,"arucoTable2"),
    "23" : mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,"arucoTable3"),

    "2" : mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,"arucoRobot1"),
}
idCam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,"cam")
idTable = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,"table_table_sans_bordure")
idRobot = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,"robot_base_metal")

posRobot = np.array([0.0, 0.0, 0.0])
posTable = np.array([0.0, 0.0, 0.0])
qposTable = np.array([0.0, 0.0, 0.0, 0.0])

nbArucoTableDetected = 0
robotDetected = 0
allQuatsTable = []

for k in range(len(detection1)):
    idAruco = detectorBasedeporte.detected_ids[k]
    idObject = arucoIdtoObjectId[str(idAruco)]
    posAruco = detection1[k]
    posAruco = np.array(posAruco[1:]).reshape(2, 3)
    R = cv2.Rodrigues(posAruco[0])[0]
    qposAruco = Rotation.from_matrix(T_world_cam @ R).as_quat(scalar_first=True)
    posArucoCam = posAruco[1] / 1000.0

    model.body_pos[idObject] = T_world_cam @ posArucoCam + posCam
    model.body_quat[idObject] = qposAruco

    if(20<=idAruco<=23):
        nbArucoTableDetected += 1
        posTable += model.body_pos[idObject] - posArucoTable[str(idAruco)]
        allQuatsTable.append(model.body_quat[idObject])
    elif(idAruco==2):
        robotDetected = 1
        posRobot = model.body_pos[idObject] - posArucoRobot
        qposRobot = model.body_quat[idObject]

if(nbArucoTableDetected!=0):
    posTable /= nbArucoTableDetected
    allRotTable = Rotation.from_quat(allQuatsTable, scalar_first=True)
    qposTable = mean_robust(allRotTable).as_quat(scalar_first=True)
    R_table = Rotation.from_quat(qposTable, scalar_first=True)
    qposTable = (R_table * Rzn).as_quat(scalar_first=True) 
    model.body_pos[idTable] = posTable
    model.body_quat[idTable] = qposTable
if(robotDetected!=0):
    R_Robot = Rotation.from_quat(qposRobot, scalar_first=True)
    qposRobot = (R_Robot ).as_quat(scalar_first=True)
    model.body_pos[idRobot] = posRobot
    model.body_quat[idRobot] = qposRobot

model.body_pos[idCam] = posCam

mujoco.mj_forward(model, data)


manager = plt.get_current_fig_manager()
manager.window.setGeometry(0, 0, 960, 1080)

# viewer.opt.frame = mujoco.mjtFrame.mjFRAME_BODY
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