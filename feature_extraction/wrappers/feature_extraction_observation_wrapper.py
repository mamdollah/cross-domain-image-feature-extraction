import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box


from feature_extraction.feature_extractors.stage_feature_extractor import StageFeatureExtractor


class FeatureExtractionObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(FeatureExtractionObservationWrapper, self).__init__(env)

        self.feature_extractor = (model_name, stage)
        self.observation_space = self.feature_extractor.output_dim

    def observation(self, obs):
        # Extract features from the observation
        return self.feature_extractor.extract_features(obs)