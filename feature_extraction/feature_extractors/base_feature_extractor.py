from abc import ABC, abstractmethod

class BaseFeatureExtractor(ABC):
    @property
    @abstractmethod
    def output_dim(self):
        """Return the output dimension of the feature extractor."""
        pass

    @abstractmethod
    def extract_features(self, observation):
        """Extract features from data."""
        pass
