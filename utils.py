import time
from typing import Union
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.base_class import BaseAlgorithm
from gym import Env

def render_env_with_model(env: Union[Env, VecEnv], model: BaseAlgorithm, num_steps: int = 1000, fps: int = 30) -> None:
    """
    Render interactions of a model with an environment, supporting both vectorized and non-vectorized environments.

    :param env: The environment for rendering interactions (can be vectorized or non-vectorized).
    :param model: The model used for predicting actions.
    :param num_steps: Total number of steps to render. If the environment is vectorized, this is the total number of steps across all environments.
    :param fps: Frames per second for rendering control.
    """

    print("Rendering model")
    is_vec_env = isinstance(env, VecEnv)
    delay = 1.0 / fps
    env.metadata["render_fps"] = fps

    steps_done = 0

    if is_vec_env:
        print("Rendering vectorized environment")
        obs = env.reset()
        while steps_done < num_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, infos = env.step(action)
            env.render("human")
            time.sleep(delay)
            steps_done += env.num_envs

    else:
        print("Rendering non-vectorized environment")
        obs = env.reset()
        while steps_done < num_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = env.step(action)
            env.render("human")
            time.sleep(delay)
            steps_done += 1

    env.close()
