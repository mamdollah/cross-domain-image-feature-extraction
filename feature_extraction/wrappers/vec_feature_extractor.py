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

    def __init__(self, venv: VecEnv, feature_extractor: BaseFeatureExtractor):
        super(VecFeatureExtractor, self).__init__(venv)
        self.feature_extractor = feature_extractor
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=feature_extractor.output_dim, dtype=np.float32
        )

    def reset(self) -> np.ndarray:
        """
        Reset all environments in the vector and preprocess initial observations.

        Returns:
            The initial observations after preprocessing.
        """
        print("VecFeatureExtractor reset")
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
        Apply the feature extraction process to the batch of observations.

        Args:
            obs: The original batch of observations from the environments.

        Returns:
            The processed batch of observations after feature extraction.
        """
        return np.array([self.feature_extractor.extract_features_stack(o) for o in obs])
