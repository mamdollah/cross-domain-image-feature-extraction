from abc import ABC, abstractmethod

class BaseFeatureExtractor(ABC):
    def __init__(self):
        self._output_dim = None
    @property
    @abstractmethod
    def output_dim(self):
        """Return the output dimension of the feature extractor."""
        return self._output_dim

    @abstractmethod
    def extract_features(self, observation):
        """Extract features from data."""
        pass
