import numpy as np
import mujoco
import mujoco.viewer
import gymnasium as gym
from gymnasium import spaces
from scipy.spatial.transform import Rotation

from tools.CardBoardBox import CardBoardBox

class RobotEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, xml_path, frame_skip=10, max_steps=500, viewer=False):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.frame_skip = frame_skip
        self.max_steps = max_steps
        self.n_steps = 0
        self.sceneTimeStep = self.model.opt.timestep

        self.posRobot = np.array([1.0, 0.0, 0.0])
        self.R_world_robot = np.zeros([3,3])  
        self.posObjectif = np.array([0.0, 0.0, 0.0])
        self.R_world_obj = np.zeros([3,3])
        self.R_robot_obj = np.zeros([3,3])

        self.idRobotModel = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot_base_metal")
        idRobotRightWheelCapsule = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "robot_capsuleWheel")
        self.robotWheelRadius = self.model.geom_size[idRobotRightWheelCapsule][0]
        self.robotRadius = 0.20645 #[m]

        self.idObjectif = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "objectif")

        self.allCardBoardBox = []
        for k in range(1):
            self.allCardBoardBox.append(CardBoardBox(k, self.model, self.data))

        self.robotMaxSpeed = 1 #[m/s]
        self.wheelMaxSpeed = self.robotMaxSpeed / self.robotWheelRadius #[rad/s]

        self.action_space = spaces.Box(-1, 1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(11,), dtype=np.float32)

        self.previousDistObjectif = 1.0 # pas terrible
        self.previousDistAngleObjectif = np.pi/2 # pas terrible

        self.info_reward = {"r_dist" : 0.0, "r_angle_dist": 0.0, "r_terminated": 0.0, "r_boxes_dist" : 0.0}

        if viewer:
            self.viewer = mujoco.viewer.launch_passive(
                model = self.model,
                data = self.data,
                show_left_ui=False,
                show_right_ui=False,
            )

    def _get_obs(self, action):
        self.posRobot = self.data.xpos[self.idRobotModel].copy()
        self.R_world_robot = self.data.xmat[self.idRobotModel].reshape(3,3)
        self.R_robot_obj = self.R_world_robot.T @ self.R_world_obj
        posObjRobot = (self.posObjectif - self.posRobot) @ self.R_world_robot
        self.yaw_error = np.arctan2(self.R_robot_obj[1,0], self.R_robot_obj[0,0])

        observationCardBoard = np.array([])
        minDistCardboardboxRobot = np.inf
        closestCardBoardBox = self.allCardBoardBox[0] 
        for CardBoardBox in self.allCardBoardBox:
            CardBoardBox.update(self.posRobot, self.R_world_robot)
            if CardBoardBox.distCardboardboxRobot <= minDistCardboardboxRobot:
                minDistCardboardboxRobot = CardBoardBox.distCardboardboxRobot
                closestCardBoardBox = CardBoardBox 

        observationCardBoard = np.concatenate([
            observationCardBoard,
            np.array(closestCardBoardBox.posBoxRobot[:2]).reshape(-1),
            np.array([np.cos(closestCardBoardBox.yaw_error), np.sin(closestCardBoardBox.yaw_error)]),
            np.array(closestCardBoardBox.boxVertical).reshape(-1)
        ])
        
        observation =  np.concatenate([
            posObjRobot[:2], #2// Position de l'objectif repère Robot
            np.array([np.cos(self.yaw_error), np.sin(self.yaw_error)]), #2// Différence d'orientation objectif repère Robot
            np.asarray(action, dtype=np.float32).reshape(-1), #2// Actions précédente pour les roues du robot
            observationCardBoard
        ]).astype(np.float32)

        return observation

    def _get_rew(self):
        sigma_dist = 2.0
        sigma_angle_dist = 1.5# 1.5 pour presque 0 à 360
        sigma_boxes_dist = 0.2
        disObjectif = np.linalg.norm(self.posObjectif - self.posRobot)
        reward_dist = np.exp(-(disObjectif**2)/(sigma_dist**2))

        weight_angle_dist = 0.05
        distAngleObjectif = np.linalg.norm(self.yaw_error)
        reward_angle_dist = np.exp(-(distAngleObjectif**2)/(sigma_angle_dist**2)) * weight_angle_dist

        reward_terminated = 0
        if(self._terminated()):
            reward_terminated = 400

        reward_boxes_moved = 0
        reward_boxes_dist = 0
        for CardBoardBox in self.allCardBoardBox:
            if (CardBoardBox.mov >= 0.001):
                reward_boxes_moved = -0.5
            if(CardBoardBox.distCardboardboxRobot <= 2*self.robotRadius):
                reward_boxes_dist -= np.exp(-CardBoardBox.distCardboardboxRobot**2/sigma_boxes_dist**2)

        self.previousDisObjectif = disObjectif
        self.info_reward["r_dist"] += reward_dist
        self.info_reward["r_angle_dist"] += reward_angle_dist
        self.info_reward["r_terminated"] += reward_terminated
        self.info_reward["r_boxes_dist"] += reward_boxes_dist

        return reward_dist + reward_angle_dist + reward_terminated


    def _terminated(self, posSeuil=0.05, angleSeuil=5):
        '''posSeuil [m], angleSeuil [deg]'''
        disObjectif = np.linalg.norm(self.posRobot-self.posObjectif)
        distAngleObjectif = np.linalg.norm(self.yaw_error)
        if(disObjectif<=posSeuil and distAngleObjectif<=angleSeuil):
            return True
        return False

    def _rand_pos_in_table(self):
        pos_robot = self.np_random.random(3)
        pos_robot[0] *= (2 - 2*self.robotRadius)
        pos_robot[0] += self.robotRadius
        pos_robot[1] -= 0.5
        pos_robot[1] *= (3 - 2*self.robotRadius)
        pos_robot[2] *= 2*np.pi
        pos_robot = np.concatenate([
            np.array(pos_robot[:2]),
            np.array([0.0]), 
            Rotation.from_euler("xyz", [0.0, 0.0, pos_robot[2]]).as_quat(scalar_first=True), 
            np.array([0.0, 0.0])
        ])
        return pos_robot

    def reset(self, seed=None, options=None):
        mujoco.mj_resetData(self.model, self.data)
        super().reset(seed=seed)

        pos_robot = self._rand_pos_in_table()
        self.data.joint("robot_base_metal_freejoint").qpos = pos_robot[:7]
        self.data.joint("robot_left_wheel").qpos = 0.0
        self.data.joint("robot_right_wheel").qpos = 0.0

        self.idRobotModel

        pos_objectif = self._rand_pos_in_table()
        self.posObjectif = pos_objectif[:3]
        self.R_world_obj = Rotation.from_quat(pos_objectif[3:7], scalar_first=True).as_matrix()
        self.model.body_pos[self.idObjectif] = pos_objectif[:3]
        self.model.body_quat[self.idObjectif] = pos_objectif[3:7]     

        forbidden_poses = np.array([pos_robot[:2], pos_objectif[:2]])
        for CardBoardBox in self.allCardBoardBox:
            pos_box = self._rand_pos_in_table()
            vertical = True if self.np_random.random(1)[0] >= 0.5 else False
            forbidden_poses = CardBoardBox.reset(pos_box, vertical, forbidden_poses)
        
        mujoco.mj_forward(self.model, self.data)
        self.n_steps = 0

        obs = self._get_obs(np.array([0.0, 0.0]))
        self.info_reward = {"r_dist" : 0.0, "r_angle_dist": 0.0, "r_terminated": 0.0, "r_boxes_dist" : 0.0}


        return (obs, {})

    def step(self, action):
        self.n_steps+=1
        self.data.ctrl[:] = action * self.wheelMaxSpeed

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        obs = self._get_obs(action)
        reward = self._get_rew()
        terminated = self._terminated()
        truncated = True if self.n_steps >= self.max_steps else False
        infos = {}
        if terminated or truncated:
            infos["ep_rew"] = self.info_reward

        return obs, reward, terminated, truncated, infos
