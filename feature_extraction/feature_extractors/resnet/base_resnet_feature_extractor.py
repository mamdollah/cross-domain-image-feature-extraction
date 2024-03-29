import torch
import torch.nn as nn
from abc import abstractmethod

from torchinfo import torchinfo
from torchvision.transforms import transforms

from feature_extraction.feature_extractors.base_feature_extractor import BaseFeatureExtractor


class BaseResnetFeatureExtractor(BaseFeatureExtractor, nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.freeze_params()

    def freeze_params(self):
        for param in self.parameters():
            param.requires_grad = False

    @property
    def output_dim(self):
        dummy_input = torch.rand(1, 3, 224, 224)
        output = self.extract_features(dummy_input)
        return output.shape

    @staticmethod
    def process_image(image):
        image_processor = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return image_processor(image)

    def forward(self, x):
        with torch.no_grad():
            for layer in self.model:
                if layer:
                    x = layer(x)
        return x

    @staticmethod
    def reduce_dim(features):
        return nn.AdaptiveAvgPool2d((1, 1))(features).view(features.size(0), -1)

    def extract_features(self, image):
        processed_image = self.process_image(image)
        feature_embeddings = self.forward(processed_image)
        reduced_dim = self.reduce_dim(feature_embeddings)
        return reduced_dim

