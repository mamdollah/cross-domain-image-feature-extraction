import gymnasium as gym


class FeatureExtractionObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, feature_extractor: BaseFeatureExtractor):
        super(FeatureExtractionObservationWrapper, self).__init__(env)
        self.feature_extractor = feature_extractor
        self.observation_space = self.feature_extractor.output_dim

    def observation(self, obs):
        # Extract features from the observation
        return self.feature_extractor.extract_features(obs)