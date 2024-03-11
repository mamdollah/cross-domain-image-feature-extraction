import torch
from torch import nn
from abc import ABC, abstractmethod

from transformers import ResNetConfig, ResNetModel, AutoImageProcessor



class FeatureExtractor(ABC):
    def __init__(self, model_name: str):
        super().__init__()
        # List of allowed model names
        allowed_model_names = [
            "microsoft/resnet-18",
            "microsoft/resnet-26",
            "microsoft/resnet-34",
            "microsoft/resnet-50",
            "microsoft/resnet-101",
            "microsoft/resnet-152"
        ]

        # Check if the provided model_name is in the list of allowed model names
        if model_name not in allowed_model_names:
            raise ValueError(f"model_name '{model_name}' is not supported. Please choose from {allowed_model_names}")

        # Load the ResNet model and image processor
        config = ResNetConfig.from_pretrained(model_name, output_hidden_states=True)
        self.model = ResNetModel.from_pretrained(model_name, config=config)  # Assuming this is a PyTorch model
        self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Pooling layer to reduce dimensions

    @property
    @abstractmethod
    def output_dim(self):
        """Subclasses should return the output dimension of the feature extractor."""
        pass

    @abstractmethod
    def extract_features(self, observation):
        """Subclasses must implement this method to extract features from data using the ResNet model."""
        pass

    def reduce_dim(self, features):
        """Reduce the dimensionality of the features using average pooling."""
        # Applying the adaptive average pooling to reduce each feature map (channel) to a single value
        reduced_features = self.adaptive_avg_pool(features)
        # Flattening the features to remove the 1x1 spatial dimension, resulting in a shape of [batch_size, num_channels]
        return reduced_features.view(features.size(0), -1)

