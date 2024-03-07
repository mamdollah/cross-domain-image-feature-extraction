from transformers import ResNetConfig, ResNetModel, AutoImageProcessor
from typing import Type
import torch

from feature_extractor import FeatureExtractor

class StageFeatureExtractor(FeatureExtractor):
    def __init__(self, model_name: str, stage: int):
        super().__init__(model_name)  # Properly initialize the base class
        self.stage = stage

    @property
    def output_dim(self):
        self.model.config.hidden_sizes[self.stage]  # Return the output dimension of the desired stage

    def extract_features(self, observation):
        inputs = self.image_processor(observation, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        features = self.reduce_dim(outputs.hidden_states[self.stage])  # Reduce the dimensionality of the features

        return features




