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

from tools.detectArucos import detectArucos
from tools.utils import mean_robust, rotation_kabsch

imageId = 7
idArucoRobot = 2

## setup 3D env

def setup_mujoco():
    models_dir = Path(__file__).parent / "models"
    tree = ET.parse(models_dir / "scene.xml")
    root = tree.getroot()
    xml = ET.tostring(root, encoding="unicode")
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
    arucoIdtoObjectId = {
        "robot" : mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,"robot_base_metal"),
        "table" : mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,"table_table_sans_bordure"),
        "cam" : mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,"cam"),
        
        "20" : mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,"arucoTable0"),
        "21" : mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,"arucoTable1"),
        "22" : mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,"arucoTable2"),
        "23" : mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,"arucoTable3"),

        "2" : mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,"arucoRobot1"),
    }

    return model, data, viewer, arucoIdtoObjectId

model, data, viewer, arucoIdtoObjectId= setup_mujoco()

## detect arucos irl

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

## runtime

def detectToRobotPosition(detection, detected_ids, posRobot, qposRobot):
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

    if (idArucoRobot in detected_ids):
        nbArucoTableDetected = 0
        allArucosTable = []
        allPosTable = []
        allQuatsTable = []
        allPoses = {}
        posArucoOnRobot = [0.002, 0.0, 0.4504]

        idDetectionRobot = np.where(detected_ids==idArucoRobot)[0][0]
        pos_ArucoRobot_Cam = detection[idDetectionRobot]
        pos_ArucoRobot_Cam = np.array(pos_ArucoRobot_Cam[1:]).reshape(2, 3)
        R_cam_arucoRobot = Rotation.from_rotvec(pos_ArucoRobot_Cam[0])

        R_world_robot = Rotation.from_quat(qposRobot, scalar_first=True)

        qposArucoRobot = qposRobot
        posArucoRobot = posRobot + R_world_robot.apply(posArucoOnRobot)

        allPoses[str(idArucoRobot)] = [posArucoRobot, qposArucoRobot]
        
        R_world_cam = R_world_robot * R_cam_arucoRobot.inv()
        qposCam = R_world_cam.as_quat(scalar_first=True)
        posCam = posArucoRobot - R_world_cam.apply(pos_ArucoRobot_Cam[1]/1000)

        allPoses["cam"] = [posCam, qposCam]

        for k in range(len(detection)):
            if(detected_ids[k]!=idArucoRobot):
                R_cam_arucoRobot = Rotation.from_rotvec(pos_ArucoRobot_Cam[0])
        
                idAruco = detected_ids[k]
                posAruco = detection[k]
                posAruco = np.array(posAruco[1:]).reshape(2, 3)
                R_cam_aruco = Rotation.from_rotvec(posAruco[0])
                R_world_aruco = R_world_cam * R_cam_aruco

                qposAruco =  R_world_aruco.as_quat(scalar_first=True)
                posAruco = R_world_cam.apply(posAruco[1] / 1000.0) + posCam

                allPoses[str(idAruco)] = [posAruco, qposAruco]

                if(20<=idAruco<=23):
                    nbArucoTableDetected += 1
                    allArucosTable.append(idAruco)
                    allPosTable.append(posAruco)
                    allQuatsTable.append(qposAruco)
            
        if(nbArucoTableDetected!=0):
            allRotTable = Rotation.from_quat(allQuatsTable, scalar_first=True)
            R_world_table, posTable = mean_robust(allRotTable, allPosTable, allArucosTable)
            R_world_table = rotation_kabsch(R_world_table)
            Rzn = Rotation.from_euler('z', -np.pi/2)
            R_world_table = R_world_table * Rzn
            qposTable = (R_world_table).as_quat(scalar_first=True)
            posTable[2] = 0.001
            allPoses["table"] = [posTable, qposTable]

        posRobotOnTable = R_world_table.inv().apply(posRobot-posTable)
        angleRobotOnTable = np.rad2deg((R_world_table.inv() * R_world_robot).as_euler("xyz")[2])

        return(posRobotOnTable, angleRobotOnTable, allPoses)
    return(-1)

def update_mujoco(model, data, allPoses, arucoIdtoObjectId):
    keys = allPoses.keys()
    for key in keys:
        objectId = arucoIdtoObjectId[key]
        model.body_pos[objectId] = allPoses[key][0]
        model.body_quat[objectId] = allPoses[key][1]

    mujoco.mj_forward(model, data)


mujocoOn = True
if(mujocoOn):
    posRobot = data.xpos[arucoIdtoObjectId["robot"]].copy()
    qposRobot = data.xquat[arucoIdtoObjectId["robot"]].copy()
else:
    posRobot = [0.0, 0.0, 0.0]
    qposRobot = [1.0, 0.0, 0.0, 0.0]
posRobotOnTable, angleRobotOnTable, allPoses = detectToRobotPosition(detectorBasedeporte.detectedArucos, detectorBasedeporte.detected_ids, posRobot, qposRobot)

if(mujocoOn):
    update_mujoco(model, data, allPoses, arucoIdtoObjectId)

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
        posRobotOnTable, angleRobotOnTable, allPoses = detectToRobotPosition(detectorBasedeporte.detectedArucos, detectorBasedeporte.detected_ids, posRobot, qposRobot)
        print(posRobotOnTable, angleRobotOnTable)
        if(mujocoOn):
            update_mujoco(model, data, allPoses, arucoIdtoObjectId)
        t1 = time.perf_counter()
        print(f"temps exec : {t1-t0:6f}")

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