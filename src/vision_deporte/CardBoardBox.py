import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

class CardBoardBox():
    posCardboardBox = np.array([1.0, 0.0, 0.0])
    R_world_box = np.zeros([3,3])
    prevPosCardboardBox = np.array([1.0, 0.0, 0.0])
    distCardboardboxRobot = 0.0
    boxVertical = False

    def __init__(self, num, model, data):
        self.num = num
        self.model = model
        self.data = data
        self.idCardboardBox = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cardboard_box"+str(num))
        
    def update(self, posRobot, R_world_robot):
        self.posCardboardBox = self.data.xpos[self.idCardboardBox].copy()
        self.R_world_box = self.data.xmat[self.idCardboardBox].reshape(3,3)
        self.R_robot_box = R_world_robot.T @ self.R_world_box
        self.posBoxRobot = (self.posCardboardBox - posRobot) @ R_world_robot
        self.mov = np.linalg.norm(self.posCardboardBox - self.prevPosCardboardBox)
        self.prevPosCardboardBox = self.posCardboardBox
        self.distCardboardboxRobot = np.linalg.norm(self.posBoxRobot)
        self.yaw_error = np.arctan2(self.R_robot_box[1,0], self.R_robot_box[0,0]) 

    def reset(self, pos_box : np.array, vertical : bool, forbidden_poses: np.array):
        self.boxVertical = vertical
        for forbidden_pos in forbidden_poses:
            if np.linalg.norm(forbidden_pos - pos_box[:2]) <= 0.32:
                pos_box[:2] = [self.idCardboardBox + 5.0, 0.0]
        if vertical:
            pos_box[2] = 0.16
            new_quat = Rotation.from_quat(pos_box[3:7], scalar_first=True) * Rotation.from_euler("xyz", [np.pi/2, 0.0, 0.0])
            pos_box[3:7] = new_quat.as_quat(scalar_first=True)
        else:
            pos_box[2] = 0.055
            pos_box[3:7] = Rotation.from_quat(pos_box[3:7], scalar_first=True).as_quat(scalar_first=True)
        self.prevPosCardboardBox = pos_box[:3]
        self.data.joint("cardboard_box_joint" +str(self.num)).qpos = pos_box[:7]
        forbidden_poses = np.concatenate([forbidden_poses, np.array([pos_box[:2]])])
        return forbidden_poses
