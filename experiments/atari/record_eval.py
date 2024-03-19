import gymnasium as gym
import wandb
import torch
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder


class RecordEval:

    def __init__(self, env_name, model, block_number=1):
        self.model = model
        self.block_number = block_number
        self.config = {
            "env_name": env_name,
            "feature_extractor_block": block_number,
            "record_video_trigger": 8,
        }

        self.run = wandb.init(
            project="sb3",
            config=self.config,
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
        )

    def make_env(self):
        # Create the environment with monitoring
        env = gym.make(self.config["env_name"])
        env = Monitor(env)
        env = DummyVecEnv([lambda: env])
        env = VecVideoRecorder(
            env,
            f"eval_videos/{self.config['env_name']}_{self.block_number}",
            record_video_trigger=lambda x: x % 4 == 0,  # Record every 8 time-steps
            video_length=150,  # 100 frames
        )
        return env

    def record(self):
        eval_env = self.make_env()
        obs = eval_env.reset()
        with torch.no_grad():
            while True:
                action, _ = self.model.predict(obs)
                obs, _, done, _ = eval_env.step(action)
                if done:
                    break

        eval_env.close()
