import gymnasium as gym
import numpy as np

from feature_extraction.feature_extractors.base_feature_extractor import BaseFeatureExtractor


class FeatureExtractionObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, feature_extractor: BaseFeatureExtractor):
        super(FeatureExtractionObservationWrapper, self).__init__(env)
        self.feature_extractor = feature_extractor
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=feature_extractor.output_dim, dtype=np.uint8
        )
    def observation(self, obs):
        # Extract features from the observation
        return self.feature_extractor.extract_features(obs)