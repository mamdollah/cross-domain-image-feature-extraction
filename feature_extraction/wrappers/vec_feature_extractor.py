import numpy as np
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper
from typing import Optional, List, Any, Type
import gymnasium as gym
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn

from feature_extraction.feature_extractors.base_feature_extractor import BaseFeatureExtractor


class VecFeatureExtractor(VecEnvWrapper):
    """
    Vectorized environment wrapper for applying a feature extractor to observations
    from a VecEnv. This wrapper allows preprocessing of observations across all
    environments in the vector using a custom feature extractor.

    :param venv: The vectorized environment to wrap.
    :param feature_extractor: An instance of BaseFeatureExtractor for extracting features from observations.
    """

    def __init__(self, venv: VecEnv, feature_extractor: BaseFeatureExtractor, n_stacks: int = 1):
        super(VecFeatureExtractor, self).__init__(venv)
        self.feature_extractor = feature_extractor
        self.n_stacks = n_stacks
        adjusted_output_dim = (feature_extractor.output_dim[0], feature_extractor.output_dim[1] * n_stacks)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=adjusted_output_dim, dtype=np.float32
        )

    def reset(self) -> np.ndarray:
        """
        Reset all environments in the vector and preprocess initial observations.

        Returns:
            The initial observations after preprocessing.
        """
        obs = self.venv.reset()
        return self.observation(obs)

    def step_wait(self) -> VecEnvStepReturn:
        """
        Step the environments synchronously, process observations through the feature extractor,
        and return the processed observations along with rewards, dones, and infos.

        Returns:
            The tuple of (observations, rewards, dones, infos) after stepping all environments and preprocessing the observations.
        """
        obs, rewards, dones, infos = self.venv.step_wait()
        return self.observation(obs), rewards, dones, infos

    def observation(self, obs: np.ndarray) -> np.ndarray:
        """
        Apply the feature extraction process to the batch of observations using np.split,
        and then restack the features to form a single vector per environment.

        Args:
            obs: The original batch of observations from the environments, shape (8, 3 * n_stacks, 224, 224).

        Returns:
            The processed batch of observations after feature extraction, shape (8, 1, 1024).
        """

        processed_obs_all_envs = []
        #Comments assume block 1 is used
        #(8, 12, 224, 224)
        # Iterate through each environment's observations
        for env_obs in obs:
            #(12,224,224)
            # Use np.split to divide the 12-channel observation into 4 RGB images
            individual_rgb_images = np.split(env_obs, self.n_stacks, axis=0)
            #(4, 3, 224,224)

            # Process each RGB image through the feature extractor
            processed_env_obs = [self.feature_extractor.extract_features(img).cpu().numpy() for img in individual_rgb_images]
            #(4, 1, 256)

            # Concatenate the processed observations for the current environment into a single vector
            concatenated_obs = np.concatenate(processed_env_obs, axis=-1)
            #(1, 1024)

            # Add the concatenated observations to the list
            processed_obs_all_envs.append(concatenated_obs)

        # Convert the list to a NumPy array and ensure it has the desired shape (8, 1, 1024)
        processed_obs_all_envs = np.array(processed_obs_all_envs).reshape(len(obs), 1, -1)
        #(8, 1, 1024)
        return processed_obs_all_envs

