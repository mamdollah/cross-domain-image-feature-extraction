import torch.nn as nn
from feature_extraction.feature_extractors.resnet.base_resnet_feature_extractor import BaseResnetFeatureExtractor


class StageFeatureExtractor(BaseResnetFeatureExtractor):
    def __init__(self, model, num_stages=1):
        self.num_stages = num_stages
        super().__init__(model)

    def _get_features(self):
        stages = [stage if i < self.num_stages else None for i, stage in enumerate(self.stages)]
        return stages



if __name__ == "__main__":
    # pass
    from torchinfo import torchinfo
    import torchvision.models as models
    from torchvision.models import ResNet50_Weights

    import torch

    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    rand_input = torch.rand(1, 3, 224, 224)
    bfe = StageFeatureExtractor(model, num_stages=1)
    print(bfe.extract_features(rand_input).shape)
    torchinfo.summary(bfe)