import os
import time

import torch
from PIL import Image
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.base_class import BaseAlgorithm
from gymnasium import Env

import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Type

import gymnasium as gym
import numpy as np

from stable_baselines3.common import type_aliases
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped, SubprocVecEnv
from typing import Callable

from torchvision.transforms.v2 import ToPILImage

from feature_extraction.wrappers.custom_atari_wrapper import CustomAtariWrapper

to_pil_image = ToPILImage()


from feature_extraction.wrappers.resnet_atari_wrapper import ResnetAtariWrapper


def linear_schedule(initial_value):
    """
    Linear learning rate schedule.
    :param initial_value: (float or str)
    :return: (function)
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress):
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress: (float)
        :return: (float)
        """
        return progress * initial_value

    return func

def render_env_with_model(env: Union[Env, VecEnv], model: BaseAlgorithm, num_steps: int = 1000, fps: int = 30) -> None:
    """
    Render interactions of a model with an environment, supporting both vectorized and non-vectorized environments.
    This is basically a copy of the `evaluate_policy` function from stable-baselines3, but with working rendering.

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


def evaluate_policy(
    model: "type_aliases.PolicyPredictor",
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
    warn: bool = True,
    fps: int = 30,
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    If a vector env is passed in, this divides the episodes to evaluate onto the
    different elements of the vector env. This static division of work is done to
    remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
    details and discussion.

    .. note::
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.

    :param model: The RL agent you want to evaluate. This can be any object
        that implements a `predict` method, such as an RL algorithm (``BaseAlgorithm``)
        or policy (``BasePolicy``).
    :param env: The gym environment or ``VecEnv`` environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to do additional checks,
        called after each step. Gets locals() and globals() passed as parameters.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of rewards and episode lengths
        per episode will be returned instead of the mean.
    :param warn: If True (default), warns user about lack of a Monitor wrapper in the
        evaluation environment.
    :return: Mean reward per episode, std of reward per episode.
        Returns ([float], [int]) when ``return_episode_rewards`` is True, first
        list containing per-episode rewards and second containing per-episode lengths
        (in number of steps).
    """
    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]

    is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )

    print("Running evaluation")
    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []

    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    observations = env.reset()
    states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    while (episode_counts < episode_count_targets).any():
        actions, states = model.predict(
            observations,  # type: ignore[arg-type]
            state=states,
            episode_start=episode_starts,
            deterministic=deterministic,
        )
        new_observations, rewards, dones, infos = env.step(actions)
        current_rewards += rewards
        current_lengths += 1
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:
                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                done = dones[i]
                info = infos[i]
                episode_starts[i] = done

                if callback is not None:
                    callback(locals(), globals())

                if dones[i]:
                    if is_monitor_wrapped:
                        # Atari wrapper can send a "done" signal when
                        # the agent loses a life, but it does not correspond
                        # to the true end of episode
                        if "episode" in info.keys():
                            # Do not trust "done" with episode endings.
                            # Monitor wrapper includes "episode" key in info if environment
                            # has been wrapped with it. Use those rewards instead.
                            episode_rewards.append(info["episode"]["r"])
                            episode_lengths.append(info["episode"]["l"])
                            # Only increment at the real end of an episode
                            episode_counts[i] += 1
                    else:
                        episode_rewards.append(current_rewards[i])
                        episode_lengths.append(current_lengths[i])
                        episode_counts[i] += 1
                    current_rewards[i] = 0
                    current_lengths[i] = 0

        observations = new_observations

        if render:
            env.render("human")
            time.sleep(1.0 / fps) # control the speed of the rendering

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return mean_reward, std_reward


def make_resnet_atari_env(
    env_id: Union[str, Callable[..., gym.Env]],
    n_envs: int = 1,
    seed: Optional[int] = None,
    start_index: int = 0,
    monitor_dir: Optional[str] = None,
    wrapper_kwargs: Optional[Dict[str, Any]] = None,
    env_kwargs: Optional[Dict[str, Any]] = None,
    vec_env_cls: Optional[Union[Type[DummyVecEnv], Type[SubprocVecEnv]]] = None,
    vec_env_kwargs: Optional[Dict[str, Any]] = None,
    monitor_kwargs: Optional[Dict[str, Any]] = None,
) -> VecEnv:
    """
    Create a wrapped, monitored VecEnv for Atari.
    It is a wrapper around ``make_vec_env`` that includes common preprocessing for Atari games.

    :param env_id: either the env ID, the env class or a callable returning an env
    :param n_envs: the number of environments you wish to have in parallel
    :param seed: the initial seed for the random number generator
    :param start_index: start rank index
    :param monitor_dir: Path to a folder where the monitor files will be saved.
        If None, no file will be written, however, the env will still be wrapped
        in a Monitor wrapper to provide additional information about training.
    :param wrapper_kwargs: Optional keyword argument to pass to the ``AtariWrapper``
    :param env_kwargs: Optional keyword argument to pass to the env constructor
    :param vec_env_cls: A custom ``VecEnv`` class constructor. Default: None.
    :param vec_env_kwargs: Keyword arguments to pass to the ``VecEnv`` class constructor.
    :param monitor_kwargs: Keyword arguments to pass to the ``Monitor`` class constructor.
    :return: The wrapped environment
    """
    return make_vec_env(
        env_id,
        n_envs=n_envs,
        seed=seed,
        start_index=start_index,
        monitor_dir=monitor_dir,
        wrapper_class=ResnetAtariWrapper,
        env_kwargs=env_kwargs,
        vec_env_cls=vec_env_cls,
        vec_env_kwargs=vec_env_kwargs,
        monitor_kwargs=monitor_kwargs,
        wrapper_kwargs=wrapper_kwargs,
    )

def make_custom_atari_wrapper(
    env_id: Union[str, Callable[..., gym.Env]],
    n_envs: int = 1,
    seed: Optional[int] = None,
    start_index: int = 0,
    monitor_dir: Optional[str] = None,
    wrapper_kwargs: Optional[Dict[str, Any]] = None,
    env_kwargs: Optional[Dict[str, Any]] = None,
    vec_env_cls: Optional[Union[Type[DummyVecEnv], Type[SubprocVecEnv]]] = None,
    vec_env_kwargs: Optional[Dict[str, Any]] = None,
    monitor_kwargs: Optional[Dict[str, Any]] = None,
) -> VecEnv:
    """
    Create a wrapped, monitored VecEnv for Atari.
    It is a wrapper around ``make_vec_env`` that includes common preprocessing for Atari games.

    :param env_id: either the env ID, the env class or a callable returning an env
    :param n_envs: the number of environments you wish to have in parallel
    :param seed: the initial seed for the random number generator
    :param start_index: start rank index
    :param monitor_dir: Path to a folder where the monitor files will be saved.
        If None, no file will be written, however, the env will still be wrapped
        in a Monitor wrapper to provide additional information about training.
    :param wrapper_kwargs: Optional keyword argument to pass to the ``AtariWrapper``
    :param env_kwargs: Optional keyword argument to pass to the env constructor
    :param vec_env_cls: A custom ``VecEnv`` class constructor. Default: None.
    :param vec_env_kwargs: Keyword arguments to pass to the ``VecEnv`` class constructor.
    :param monitor_kwargs: Keyword arguments to pass to the ``Monitor`` class constructor.
    :return: The wrapped environment
    """
    return make_vec_env(
        env_id,
        n_envs=n_envs,
        seed=seed,
        start_index=start_index,
        monitor_dir=monitor_dir,
        wrapper_class=CustomAtariWrapper,
        env_kwargs=env_kwargs,
        vec_env_cls=vec_env_cls,
        vec_env_kwargs=vec_env_kwargs,
        monitor_kwargs=monitor_kwargs,
        wrapper_kwargs=wrapper_kwargs,
    )

def save_feature_maps(feature_maps, path, num_feature_maps=1):
    print("Shape of feature embeddings: ", feature_maps.shape)
    # Ensure the save directory exists
    os.makedirs(path, exist_ok=True)

    fmps = []
    batch_size, channels, height, width = feature_maps.shape
    for batch_index in range(batch_size):
        for channel_index in range(0, min(channels, num_feature_maps)):
            # Extract the single-channel image
            single_channel_image = feature_maps[batch_index, channel_index]

            # Convert the normalized single-channel image to a PIL image
            pil_image = to_pil_image(single_channel_image.cpu())
            fmps.append(pil_image)

            # Save the PIL image
            image_path = os.path.join(path, f"feature_map_b{batch_index}_c{channel_index}.png")
            pil_image.save(image_path)

    return fmps


def save_np_image(image, image_name, save_path):
    os.makedirs(save_path, exist_ok=True)
    image_path = os.path.join(save_path, image_name)
    # Transpose the image from (Channels, Height, Width) to (Height, Width, Channels)
    image_transposed = np.transpose(image, (1, 2, 0))
    # Convert to PIL Image and save
    pil_image = Image.fromarray(image_transposed.astype(np.uint8))  # Ensure it's uint8

    pil_image.save(image_path)
    print("Original image saved.")
    return pil_image

def save_tensor_image(tensor, image_name, save_path):
    print("Tensor image shape: ", tensor.shape)
    os.makedirs(save_path, exist_ok=True)
    image_path = os.path.join(save_path, image_name)

    # Ensure tensor is in CPU and convert to NumPy
    image_np = tensor.cpu().numpy()

    # Transpose from (Channels, Height, Width) to (Height, Width, Channels) and ensure RGB
    image_transposed = np.transpose(image_np, (1, 2, 0))

    # Convert to PIL Image and save
    pil_image = Image.fromarray(image_transposed.astype(np.uint8))
    pil_image.save(image_path)
    print("Tensor image saved.")
    return pil_image


def save_reduced_feature_map(reduced_features, save_dir, output_dim):
    features = reduced_features.detach().cpu().numpy().squeeze()
    # Normalize the features to [0, 1]
    features = (features - features.min()) / (features.max() - features.min())

    # Calculate the dimensions of the image to be as square as possible
    num_features = output_dim[1]
    op_height = int(np.sqrt(num_features))
    op_width = int(np.ceil(num_features / op_height))

    # Ensure the total number of pixels matches the number of features
    # If not perfectly square, adjust to the closest possible rectangle
    if op_height * op_width < num_features:
        op_height += 1

    # Reshape and pad if necessary
    padded_features = np.zeros(op_height * op_width)
    padded_features[:num_features] = features
    image_data = np.reshape(padded_features, (op_height, op_width))

    # Convert to tensor and add a channel dimension to fit (C, H, W)
    image_data_tensor = torch.tensor(image_data).float().unsqueeze(0)
    img = to_pil_image(image_data_tensor)

    # Save the image
    img.save(os.path.join(save_dir, 'reduced_feature_map.png'))
    return img


