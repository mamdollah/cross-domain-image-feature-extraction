import numpy as np
from transformers import ResNetConfig, ResNetModel, AutoImageProcessor
from typing import Type
import torch
import gymnasium as gym
from PIL import Image

from feature_extractor import FeatureExtractor

class StageFeatureExtractor(FeatureExtractor):
    def __init__(self, model_name: str, stage: int):
        super().__init__(model_name)  # Properly initialize the base class
        self.stage = stage
        self.image_processor = AutoImageProcessor.from_pretrained(model_name)


    @property
    def output_dim(self):
        size = self.model.config.hidden_sizes[self.stage - 1]  # Return the output dimension of the desired stage # 512
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(size,), dtype=np.float32)

    def process_image(self, image):
        return self.image_processor(image, return_tensors="pt")

    def extract_features(self, image):
        inputs = self.process_image(image)
        print("Test")

        with torch.no_grad():
            outputs = self.model(**inputs)
        features = self.reduce_dim(outputs.hidden_states[self.stage])  # Reduce the dimensionality of the features

        return features