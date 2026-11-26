from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env

import time
from pathlib import Path
import argparse

from tools.Callbacks import TensorboardCallback, TimingCallback
from envs import RobotEnv, agentfile
        
parser = argparse.ArgumentParser(
    description="Train and visualise robot agent"
)
parser.add_argument(
    "-t",
    "--train",
    action="store_true"
)
parser.add_argument(
    "-v",
    "--viewer",
    action="store_true"
)

if __name__ == "__main__":
    args = parser.parse_args()
    
    if args.train : 
        SBenv = make_vec_env(
            lambda : RobotEnv(str(Path(__file__).resolve().parent / "models/scene.xml")),
            n_envs=1, 
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
        out = base / "trained_agent" / agentfile
        out.mkdir(exist_ok=True)
        model.learn(total_timesteps=2_000_000, callback=[TensorboardCallback(), TimingCallback()])
        model.save(str(out / "ppo_robot"))
        SBenv.save(str(out / "vecnormalize.pkl"))

    else:
        t0 = time.monotonic()
        try:
            env = DummyVecEnv([lambda: RobotEnv(str(Path(__file__).resolve().parent / "models/scene.xml"), viewer=args.viewer)])
            env = VecNormalize.load(str(Path(__file__).resolve().parent / "trained_agent" / agentfile / "vecnormalize.pkl"), env)
            env.training = False
            env.norm_reward = False

            model = PPO.load(str(Path(__file__).resolve().parent / "trained_agent" / agentfile / "ppo_robot"))
            obs = env.reset()

            RLenv = env.venv.envs[0]
            while(1):
                t1 = time.monotonic()
                if(t1-t0>= (RLenv.sceneTimeStep * RLenv.frame_skip)):
                    t0 = t1
                    action, _ = model.predict(obs, deterministic=True)
                    obs, _, done, _=env.step(action)
                    if args.viewer:
                        RLenv.viewer.sync()
                else:
                    time.sleep(0.001)

        except KeyboardInterrupt:
            if args.viewer:
                RLenv.viewer.close()
