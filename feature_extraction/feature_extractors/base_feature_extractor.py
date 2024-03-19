from torch import nn
from abc import ABC, abstractmethod

class BaseFeatureExtractor(ABC):
    def __init__(self):
        super().__init__()

    @property
    @abstractmethod
    def process_image(self):
        """Subclasses should provide their own image processor."""
        pass

    @property
    @abstractmethod
    def output_dim(self):
        """Subclasses should return the output dimension of the feature extractor."""
        pass

    @abstractmethod
    def extract_features(self, observation):
        """Subclasses must implement this method to extract features from data."""
        pass

    def reduce_dim(self, features):
        """Reduce the dimensionality of the features using average pooling."""
        adaptive_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Pooling layer to reduce dimensions
        reduced_features = adaptive_avg_pool(features)
        return reduced_features.view(features.size(0), -1)
