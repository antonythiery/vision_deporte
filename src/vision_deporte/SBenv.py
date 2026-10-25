import numpy as np
import mujoco
import mujoco.viewer
import gymnasium as gym
from gymnasium import spaces
from scipy.spatial.transform import Rotation
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env

import time
from pathlib import Path

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
        self.R_world_robot = Rotation.from_euler("xyz", [0.0, 0.0, 0.0])  
        self.posObjectif = np.array([0.0, 0.0, 0.0])
        self.R_world_obj = Rotation.from_euler("xyz", [0.0, 0.0, 0.0])
        self.R_robot_obj = Rotation.from_euler("xyz", [0.0, 0.0, np.pi]) #pi to avoid early terminated

        self.idRobotModel = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot_base_metal")
        idRobotRightWheelCapsule = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "robot_capsuleWheel")
        self.robotWheelRadius = self.model.geom_size[idRobotRightWheelCapsule][0]
        self.robotRadius = 0.20645 #[mm]

        robotMaxSpeed = 1 #[m/s]
        self.wheelMaxSpeed = robotMaxSpeed / self.robotWheelRadius #[rad/s]

        self.action_space = spaces.Box(-1, 1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32)

        self.previousDisObjectif = 0.0

        if viewer:
            self.viewer = mujoco.viewer.launch_passive(
                model = self.model,
                data = self.data,
                show_left_ui=False,
                show_right_ui=False,
            )

    def _get_obs(self, action):
        self.posRobot = self.data.xpos[self.idRobotModel].copy()
        robot_qpos = self.data.xquat[self.idRobotModel].copy()
        self.R_world_robot = Rotation.from_quat(robot_qpos, scalar_first=True)
        self.R_robot_obj = self.R_world_robot.inv() * self.R_world_obj
        posObjRobot = self.R_world_robot.inv().apply(self.posObjectif - self.posRobot)

        observation =  np.concatenate([
            posObjRobot[:2],
            np.array([np.cos(self.R_robot_obj.as_euler("xyz")[2]), np.sin(self.R_robot_obj.as_euler("xyz")[2])]),
            np.asarray(action, dtype=np.float32).reshape(-1),
        ]).astype(np.float32)

        return observation

    def _get_rew(self):
        sigma_dist = 2.0
        sigma_angle_dist = 1.5
        disObjectif = np.linalg.norm(self.posObjectif - self.posRobot)
        reward_dist = np.exp(-(disObjectif**2)/(sigma_dist**2))

        distAngleObjectif = np.linalg.norm(self.R_robot_obj.as_euler("xyz")[2])
        reward_angle_dist = np.exp(-(distAngleObjectif**2)/(sigma_angle_dist**2))

        reward_terminated = 0
        if(self._terminated()):
            reward_terminated = 250

        self.previousDisObjectif = disObjectif
        return reward_dist + reward_angle_dist + reward_terminated

    def _terminated(self, posSeuil=0.01, angleSeuil=10):
        '''posSeuil [m], angleSeuil [deg]'''
        disObjectif = np.linalg.norm(self.posRobot-self.posObjectif)
        distAngleObjectif = np.linalg.norm(self.R_robot_obj.as_euler("xyz", degrees=True)[2])
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
        self.data.qpos = pos_robot

        pos_objectif = self._rand_pos_in_table()
        self.posObjectif = pos_objectif[:3]
        self.R_world_obj = Rotation.from_quat(pos_objectif[3:7], scalar_first=True)
        
        mujoco.mj_forward(self.model, self.data)
        self.n_steps = 0

        obs = self._get_obs(np.array([0.0, 0.0]))
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
        return obs, reward, terminated, truncated, {}

train = False

if __name__ == "__main__":
    
    if train : 
        SBenv = make_vec_env(
            lambda : RobotEnv(str(Path(__file__).resolve().parent / "models/scene.xml")),
            n_envs=8, 
            vec_env_cls=SubprocVecEnv
        )
        SBenv = VecNormalize(SBenv, norm_obs=True, norm_reward=True)
        model = PPO(
            "MlpPolicy", 
            SBenv, 
            verbose=1, 
            tensorboard_log=str(Path(__file__).resolve().parent / "tensorboard_log"),
            device="cpu"
        )
        base = Path(__file__).resolve().parent
        out = base / "trained_agent"
        out.mkdir(exist_ok=True)
        model.learn(total_timesteps=500_000)
        model.save(str(out / "ppo_robot"))
        SBenv.save(str(out / "vecnormalize.pkl"))

    else:
        t0 = time.monotonic()
        try:
            env = DummyVecEnv([lambda: RobotEnv(str(Path(__file__).resolve().parent / "models/scene.xml"), viewer=True)])
            env = VecNormalize.load(str(Path(__file__).resolve().parent / "trained_agent/vecnormalize.pkl"), env)
            env.training = False
            env.norm_reward = False

            model = PPO.load(str(Path(__file__).resolve().parent / "trained_agent/ppo_robot"))
            obs = env.reset()

            RLenv = env.venv.envs[0]
            while(1):
                t1 = time.monotonic()
                if(t1-t0>= (RLenv.sceneTimeStep * RLenv.frame_skip)):
                    t0 = t1
                    action, _ = model.predict(obs, deterministic=True)
                    obs, _, done, _=env.step(action)
                    RLenv.viewer.sync()
                else:
                    time.sleep(0.001)

        except KeyboardInterrupt:
            RLenv.viewer.close()
