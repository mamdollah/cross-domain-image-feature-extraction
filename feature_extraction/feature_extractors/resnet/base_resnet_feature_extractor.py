import torch
import torch.nn as nn
from abc import abstractmethod
from torchvision.transforms import transforms

from feature_extraction.feature_extractors.base_feature_extractor import BaseFeatureExtractor


class BaseResnetFeatureExtractor(BaseFeatureExtractor, nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.stem, self.stages = self._split_model()
        self.sequentials = self._get_features()
        self._configure_model()
        self.model = None

    def _split_model(self):
        children = list(self.model.children())
        stem, stages = children[:4], children[4:8]
        return stem, stages

    def _configure_model(self):
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

    @abstractmethod
    def _get_features(self):
        pass

    def forward(self, x):
        with torch.no_grad():
            x = self.stem_forward(x)
            for seq in self.sequentials:
                if seq:
                    x = seq(x)
        return x

    def stem_forward(self, x):
        for layer in self.stem:
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

