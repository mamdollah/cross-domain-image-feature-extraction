import gymnasium as gym

from feature_extraction.feature_extractors.base_feature_extractor import BaseFeatureExtractor


class FeatureExtractionObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, feature_extractor: BaseFeatureExtractor):
        super(FeatureExtractionObservationWrapper, self).__init__(env)
        self.feature_extractor = feature_extractor
        self.observation_space = self.feature_extractor.output_dim

    def observation(self, obs):
        # Extract features from the observation
        return self.feature_extractor.extract_features(obs)