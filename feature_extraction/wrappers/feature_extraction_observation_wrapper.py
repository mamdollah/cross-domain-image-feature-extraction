import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box


from feature_extraction.feature_extractors.stage_feature_extractor import StageFeatureExtractor


class FeatureExtractionObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(FeatureExtractionObservationWrapper, self).__init__(env)
        model_name = "microsoft/resnet-50"
        stage = 1  # Example stage, adjust based on your model and requirements
        self.feature_extractor = StageFeatureExtractor(model_name, stage)
        print("Old observation space: ", self.observation_space)
        self.observation_space = self.feature_extractor.output_dim

    def observation(self, obs):
        # Extract features from the observation
        return self.feature_extractor.extract_features(obs)