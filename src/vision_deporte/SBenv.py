import numpy as np
import mujoco
import mujoco.viewer
import gymnasium as gym
from gymnasium import spaces
from scipy.spatial.transform import Rotation

import time
from pathlib import Path

class RobotEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, xml_path, frame_skip=10, max_steps=500):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.frame_skip = frame_skip
        self.max_steps = max_steps
        self.n_steps = 0
        self.sceneTimeStep = self.model.opt.timestep

        self.posObjectif = np.array([0.0, 2.0, 0.0])
        self.R_world_obj = Rotation.from_euler("xyz", [0.0, 0.0, np.pi/2])

        self.idRobotModel = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot_base_metal")
        idRobotRightWheelCapsule = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "robot_capsuleWheel")
        self.robotWheelRadius = self.model.geom_size[idRobotRightWheelCapsule][0]
        self.robotRadius = 0.20645 #[mm]

        robotMaxSpeed = 1 #[m/s]
        self.wheelMaxSpeed = robotMaxSpeed / self.robotWheelRadius #[rad/s]

        self.action_space = spaces.Box(-1, 1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32)

        self.viewer = mujoco.viewer.launch_passive(
            model = self.model,
            data = self.data,
            show_left_ui=False,
            show_right_ui=False,
        )

    def _get_obs(self, action):
        robot_pos = self.data.xpos[self.idRobotModel].copy()
        robot_qpos = self.data.xquat[self.idRobotModel].copy()
        R_world_robot = Rotation.from_quat(robot_qpos, scalar_first=True)
        R_robot_obj = R_world_robot.inv() * self.R_world_obj
        posObjRobot = R_world_robot.inv().apply(self.posObjectif - robot_pos)

        observation =  np.concatenate([
            posObjRobot[:2],
            np.array([np.cos(R_robot_obj.as_euler("xyz")[2]), np.sin(R_robot_obj.as_euler("xyz")[2])]),
            action
        ]).astype(np.float32)

        return observation

    def _rand_pos_in_table(self):
        pos_robot = np.random.rand(3)
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
        self.data.qpos = pos_robot

        pos_objectif = self._rand_pos_in_table()
        self.posObjectif = pos_objectif[:3]
        self.R_world_obj = Rotation.from_quat(pos_objectif[3:7], scalar_first=True)
        
        mujoco.mj_forward(self.model, self.data)
        self.n_steps = 0
        return self._get_obs

    def step(self, action):
        self.n_steps+=1
        self.data.ctrl[:] = action * self.wheelMaxSpeed

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.observation = self._get_obs(action)


RLenv = RobotEnv(str(Path(__file__).resolve().parent / "models/scene.xml"))

t0 = time.monotonic()
try:
    while(1):
        t1 = time.monotonic()
        if(t1-t0>= (RLenv.sceneTimeStep * RLenv.frame_skip)):
            t0 = t1
            RLenv.step(np.array([0.0, 0.0]))
            if RLenv.n_steps % 1 == 0:
                RLenv.viewer.sync()
                # print(RLenv.observation)
            if RLenv.n_steps % 50 == 0:
                RLenv.reset()
except KeyboardInterrupt:
    RLenv.viewer.close()