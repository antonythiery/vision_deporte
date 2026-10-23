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

        self.posObjectif = np.array([0.0, 2.0, 0.0])
        self.R_world_obj = Rotation.from_euler("xyz", [0.0, 0.0, np.pi/2])

        self.idRobotModel = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot_base_metal")
        idRobotRightWheelCapsule = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "robot_capsuleWheel")
        self.robotWheelRadius = self.model.geom_size[idRobotRightWheelCapsule][0]

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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        raise NotImplementedError

    def step(self, action):
        self.n_steps+=1
        self.data.ctrl[:] = action * self.wheelMaxSpeed

        self.observation = self._get_obs(action)
        
        mujoco.mj_step(self.model, self.data)


RLenv = RobotEnv(str(Path(__file__).resolve().parent / "models/scene.xml"))

t0 = time.monotonic()
try:
    while(1):
        t1 = time.monotonic()
        if(t1-t0>=0.002):
            t0 = t1
            RLenv.step(np.array([0.1,0.0]))
            if RLenv.n_steps % 10 == 0:
                RLenv.viewer.sync()
                print(RLenv.observation)
except KeyboardInterrupt:
    RLenv.viewer.close()