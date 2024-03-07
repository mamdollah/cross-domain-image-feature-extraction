from typing import Dict, Type
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper
from feature_extraction.feature_extractors.feature_extractor import FeatureExtractor

class ObsDictWrapperWithFeatures(VecEnvWrapper):
    """
    Wrapper for a VecEnv which not only supports dict observations for Hindsight Experience Replay but also
    integrates feature extraction into the observation processing pipeline.

    :param env: The vectorized environment to wrap.
    :param feature_extractor: A class type for the feature extractor to use for processing observations.
    """

    def __init__(self, venv: VecEnv, feature_extractor: Type[FeatureExtractor]):
        super(ObsDictWrapperWithFeatures, self).__init__(venv, venv.observation_space, venv.action_space)

        self.venv = venv
        self.feature_extractor = feature_extractor()

        # Assuming the feature extractor is correctly initialized and has an `output_dim` attribute.
        self.observation_space = self.feature_extractor.output_dim

        self.spaces = list(venv.observation_space.spaces.values())

        # Initialize observation and goal dimensions based on the original space or feature extractor's output.
        self.init_dimensions()

    def init_dimensions(self):
        # This example assumes that the feature extractor already provides a suitable output dimension.
        # Adjust the logic here if you need to dynamically adjust based on the specific environment spaces.
        pass  # Placeholder for any initialization logic based on the environment and feature extractor.

    def reset(self):
        obs = self.venv.reset()
        return self.process_observation(obs)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        return self.process_observation(obs), rewards, dones, infos

    def process_observation(self, observation):
        # Extend or modify this method to process the observation through the feature extractor.
        return self.feature_extractor.extract_features(observation)

    @staticmethod
    def convert_dict(observation_dict: Dict[str, np.ndarray], observation_key: str = "observation", goal_key: str = "desired_goal") -> np.ndarray:
        """
        Concatenate observation and (desired) goal of observation dict, possibly followed by feature extraction.

        :param observation_dict: Dictionary with observation.
        :param observation_key: Key of observation in dictionary.
        :param goal_key: Key of (desired) goal in dictionary.
        :return: Concatenated observation, possibly with extracted features.
        """
        # Adjust this method if you need to apply feature extraction to individual parts before concatenation.
        concatenated_obs = np.concatenate([observation_dict[observation_key], observation_dict[goal_key]], axis=-1)
        return concatenated_obs  # Placeholder for actual feature extraction logic.
