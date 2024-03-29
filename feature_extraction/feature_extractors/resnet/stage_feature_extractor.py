import torch.nn as nn
from feature_extraction.feature_extractors.resnet.base_resnet_feature_extractor import BaseResnetFeatureExtractor


class StageFeatureExtractor(BaseResnetFeatureExtractor):
    def __init__(self, model, num_stages=1):
        if num_stages < 1 or num_stages > 4 or not isinstance(num_stages, int):
            raise ValueError("Number of stages must be between 1 and 4.")

        children = list(model.children())
        model = children[:4+num_stages]
        super().__init__(model)
