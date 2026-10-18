import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation
from tools.utils import mean_robust, rotation_kabsch, passe_bas_expo

class computePosition():

    def __init__(self, mujocoOn=False):
        self.mujocoOn = mujocoOn
        self.idArucoRobot = 2
        self.posRobot = [0.0, 0.0, 0.0]
        self.qposRobot = [1.0, 0.0, 0.0, 0.0]

        self.posTable = np.array([0.0, 0.0, 0.0])
        self.R_world_table = Rotation.identity()

        if(mujocoOn):
            self.setup_mujoco()
            self.find_robot_mujoco()

    def detectToRobotPosition(self, detection, detected_ids):
        if(self.mujocoOn):
            self.find_robot_mujoco()

        if(detection is None or detected_ids is None):
            return -1
        if(len(detection) != len(detected_ids)):
            return -1
        
        self.detection = detection
        self.detected_ids = detected_ids
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
        if(detected_ids is None):
            return -1
        
        tableDetected = False
        for id in self.detected_ids:
            if id in [20, 21, 22, 23]:
                tableDetected = True

        if (self.idArucoRobot in self.detected_ids and tableDetected):
            nbArucoTableDetected = 0
            allArucosTable = []
            allPosTable = []
            allQuatsTable = []
            self.allPoses = {}
            posArucoOnRobot = [0.002, 0.0, 0.4504]

            idDetectionRobot = np.where(self.detected_ids==self.idArucoRobot)[0][0]
            pos_ArucoRobot_Cam = self.detection[idDetectionRobot]
            pos_ArucoRobot_Cam = np.array(pos_ArucoRobot_Cam[1:]).reshape(2, 3)
            R_cam_arucoRobot = Rotation.from_rotvec(pos_ArucoRobot_Cam[0])

            R_world_robot = Rotation.from_quat(self.qposRobot, scalar_first=True)

            qposArucoRobot = self.qposRobot
            posArucoRobot = self.posRobot + R_world_robot.apply(posArucoOnRobot)

            self.allPoses[str(self.idArucoRobot)] = [posArucoRobot, qposArucoRobot]
            
            R_world_cam = R_world_robot * R_cam_arucoRobot.inv()
            qposCam = R_world_cam.as_quat(scalar_first=True)
            posCam = posArucoRobot - R_world_cam.apply(pos_ArucoRobot_Cam[1]/1000)

            self.allPoses["cam"] = [posCam, qposCam]

            for k in range(len(self.detection)):
                if(self.detected_ids[k]!=self.idArucoRobot):
                    R_cam_arucoRobot = Rotation.from_rotvec(pos_ArucoRobot_Cam[0])
            
                    idAruco = self.detected_ids[k]
                    posAruco = self.detection[k]
                    posAruco = np.array(posAruco[1:]).reshape(2, 3)
                    R_cam_aruco = Rotation.from_rotvec(posAruco[0])
                    R_world_aruco = R_world_cam * R_cam_aruco

                    qposAruco =  R_world_aruco.as_quat(scalar_first=True)
                    posAruco = R_world_cam.apply(posAruco[1] / 1000.0) + posCam

                    self.allPoses[str(idAruco)] = [posAruco, qposAruco]

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
                self.posTable, self.R_world_table = passe_bas_expo(self.posTable, posTable, self.R_world_table, R_world_table, 0.3)
                qposTable = (self.R_world_table).as_quat(scalar_first=True)
                self.posTable[2] = -0.0005
                self.allPoses["table"] = [self.posTable, qposTable]

            posRobotOnTable = R_world_table.inv().apply(self.posRobot-self.posTable)
            angleRobotOnTable = np.rad2deg((self.R_world_table.inv() * R_world_robot).as_euler("xyz")[2])

            if(self.mujocoOn):
                self.update_mujoco()
            return(posRobotOnTable, angleRobotOnTable)
        return(-1)

    def setup_mujoco(self):
        models_dir = Path(__file__).parent / "models"
        tree = ET.parse(models_dir / "scene.xml")
        root = tree.getroot()
        xml = ET.tostring(root, encoding="unicode")
        tmp = models_dir / "_scene_runtime.xml"
        tmp.write_text(xml)

        self.model = mujoco.MjModel.from_xml_path(str(tmp))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_kinematics(self.model, self.data)

        self.viewer = mujoco.viewer.launch_passive(
            model = self.model,
            data = self.data,
            show_left_ui=False,
            show_right_ui=False,
        )
        self.arucoIdtoObjectId = {
            "robot" : mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,"robot_base_metal"),
            "table" : mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,"table_table_sans_bordure"),
            "cam" : mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,"cam"),
            
            "20" : mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,"arucoTable0"),
            "21" : mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,"arucoTable1"),
            "22" : mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,"arucoTable2"),
            "23" : mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,"arucoTable3"),

            "2" : mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,"arucoRobot1"),
        }

    def find_robot_mujoco(self):
        self.posRobot = self.data.xpos[self.arucoIdtoObjectId["robot"]].copy()
        self.qposRobot = self.data.xquat[self.arucoIdtoObjectId["robot"]].copy()

    def update_mujoco(self):
        keys = self.allPoses.keys()
        for key in keys:
            if key in self.arucoIdtoObjectId.keys():
                objectId = self.arucoIdtoObjectId[key]
                self.model.body_pos[objectId] = self.allPoses[key][0]
                self.model.body_quat[objectId] = self.allPoses[key][1]

        mujoco.mj_forward(self.model, self.data)
