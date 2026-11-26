
from stable_baselines3.common.callbacks import BaseCallback
import time
import numpy as np

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """
    KEYS = ["r_dist", "r_angle_dist", "r_terminated", "r_boxes_dist"]

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.prev_num_timesteps = 0
        self.buf = {k : [] for k in self.KEYS}

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable) 
        infos = self.locals["infos"]
        for envs in range(len(infos)):
            if "ep_rew" in infos[envs]:
                for key in self.KEYS:
                    self.buf[key].append(infos[envs]["ep_rew"][key])
        return True

    def _on_rollout_end(self):
        n = len(self.buf["r_terminated"])
        if n:
            for key in self.KEYS:
                self.logger.record(f"ep_reward_terms/{key}", float(np.mean(self.buf[key])))
        for v in self.buf.values():
            v.clear()
        return super()._on_rollout_end()

class TimingCallback(BaseCallback):
    def _on_step(self):
        return True

    def _on_rollout_start(self):
        now = time.perf_counter()
        if hasattr(self, "t_end"):
            self.logger.record("timing/train_s", now - self.t_end)
        self.t_start = now

    def on_rollout_end(self):
        self.t_end = time.perf_counter()
        self.logger.record("timing/collect_s", self.t_end - self.t_start)