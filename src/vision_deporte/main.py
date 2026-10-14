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
from tools.utils import mean_robust, rotation_kabsch

imageId = 7
posCam = [-0.1, -0.5, 1.8]


idArucoRobot = 2
posArucoTable = {
    "20" : [-0.9, -0.6, 0.0],
    "21" : [0.9, -0.6, 0.0],
    "22" : [-0.9, -1.4, 0.0],
    "23" : [0.9, -1.4, 0.0]    
} # positon du centre des arucos sur la table ordre (repère STK) [m]

posArucoOnRobot = [0.002, 0.0, 0.4504]

Rzn = Rotation.from_euler('z', -np.pi/2)
Rzp = Rotation.from_euler('z', np.pi/2)

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

## detect arucos irl

image_dir = "datas/exemple_images"

images = [
    os.path.join(image_dir, f)
    for f in os.listdir(image_dir)
    if f.endswith(".png") or f.endswith(".jpg")
]

cameraParams = np.load("src/vision_deporte/tools/cameraParams.npz")

detectorBasedeporte = detectArucos(images[imageId], cameraParams)
detectorBasedeporte.detect()
detectorBasedeporte.showImage()

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

model.body_pos[idCam] = posCam

def detectToMujoco(detection, detected_ids):
    '''
    Current : 
    1 : Récupération de la position du robot dans la simu
    2 : Placement de la caméra en fonction de la postion du robot et de la détection aruco
    3 : Placement de la table en fonction de la postion de la caméra

    Amélioration possible : 
    1 : Récupération de la position du robot dans la simu
    2 : Récupération de l'orientation de la table par rapport à la caméra
    3 : Orientation de la caméra pour une table à plat
    2 : Estimation de position et orientation selon l'axe z de l'ensemble {caméra + table} en fonction de la détection du robot
    '''
    posRobot = np.array([0.0, 0.0, 0.0])
    nbArucoTableDetected = 0
    allArucosTable = []
    allPosTable = []
    allQuatsTable = []
    
    model.body_pos[idRobot] = posRobot

    if (idArucoRobot in detected_ids):
        idObject = arucoIdtoObjectId[str(idArucoRobot)]
        idDetectionRobot = np.where(detected_ids==idArucoRobot)[0][0]
        pos_ArucoRobot_Cam = detection[idDetectionRobot]
        pos_ArucoRobot_Cam = np.array(pos_ArucoRobot_Cam[1:]).reshape(2, 3)
        R_cam_arucoRobot = Rotation.from_rotvec(pos_ArucoRobot_Cam[0])

        posRobot = data.xpos[idRobot].copy()
        qposRobot = data.xquat[idRobot].copy()
        R_world_robot = Rotation.from_quat(qposRobot, scalar_first=True)

        qposArucoRobot = qposRobot
        posArucoRobot = posRobot + R_world_robot.apply(posArucoOnRobot)
        model.body_pos[idObject] = posArucoRobot
        model.body_quat[idObject] = qposArucoRobot
        
        R_world_cam = R_world_robot * R_cam_arucoRobot.inv()
        qposCam = R_world_cam.as_quat(scalar_first=True)
        posCam = posArucoRobot - R_world_cam.apply(pos_ArucoRobot_Cam[1]/1000)
        
        model.body_pos[idCam] = posCam
        model.body_quat[idCam] = qposCam

        for k in range(len(detection)):
            if(detected_ids[k]!=idArucoRobot):
                R_cam_arucoRobot = Rotation.from_rotvec(pos_ArucoRobot_Cam[0])
        
                idAruco = detected_ids[k]
                idObject = arucoIdtoObjectId[str(idAruco)]
                posAruco = detection[k]
                posAruco = np.array(posAruco[1:]).reshape(2, 3)
                R_cam_aruco = Rotation.from_rotvec(posAruco[0])
                R_world_aruco = R_world_cam * R_cam_aruco

                qposAruco =  R_world_aruco.as_quat(scalar_first=True)
                posAruco = R_world_cam.apply(posAruco[1] / 1000.0) + posCam

                model.body_pos[idObject] = posAruco
                model.body_quat[idObject] = qposAruco
                if(20<=idAruco<=23):
                    nbArucoTableDetected += 1
                    allArucosTable.append(idAruco)
                    allPosTable.append(model.body_pos[idObject].copy())
                    allQuatsTable.append(model.body_quat[idObject].copy())
            
        if(nbArucoTableDetected!=0):
            allRotTable = Rotation.from_quat(allQuatsTable, scalar_first=True)
            R_world_table, posTable = mean_robust(allRotTable, allPosTable, allArucosTable)
            R_world_table = rotation_kabsch(R_world_table)
            R_world_table = R_world_table * Rzn
            qposTable = (R_world_table).as_quat(scalar_first=True)
            posTable[2] = 0.001
            model.body_pos[idTable] = posTable
            model.body_quat[idTable] = qposTable

        posRobotOnTable = R_world_table.inv().apply(posRobot-posTable)
        angleRobotOnTable = np.rad2deg((R_world_table.inv() * R_world_robot).as_euler("xyz")[2])
        print(f"posRobot : {posRobotOnTable}, angle : {angleRobotOnTable}")

    mujoco.mj_forward(model, data)

detectToMujoco(detectorBasedeporte.detectedArucos, detectorBasedeporte.detected_ids)

fig = plt.gcf()
def on_key(event):
    global imageId
    if(event.key=="right" or event.key=="left"):
        if(event.key=="right"):
            imageId+=1
        else:
            imageId-=1
        imageId = imageId % (len(images))
        detectorBasedeporte.changeImage(images[imageId])
        detectorBasedeporte.detect()
        detectorBasedeporte.showImage()
        fig.canvas.draw_idle()
        detectToMujoco(detectorBasedeporte.detectedArucos, detectorBasedeporte.detected_ids)

manager = plt.get_current_fig_manager()
manager.window.setGeometry(0, 0, 960, 1080)
fig.canvas.mpl_connect("key_press_event", on_key)
plt.show(block=False)

# viewer.opt.frame = mujoco.mjtFrame.mjFRAME_BODY
try :
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)
        fig.canvas.flush_events()
except KeyboardInterrupt: 
    pass
finally :
    viewer.close()
    plt.close("all")