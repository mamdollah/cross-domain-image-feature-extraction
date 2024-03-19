import torch.nn as nn

from feature_extraction.feature_extractors.resnet.base_resnet_feature_extractor import BaseResnetFeatureExtractor


class StageFeatureExtractor(BaseResnetFeatureExtractor):
    def __init__(self, model, num_stages=1):
        self.num_stages = num_stages
        super().__init__(model)

    def _get_features(self):
        stages = [self.stages[i] if i < self.num_stages else None for i in range(len(self.stages))]
        return stages



if __name__ == "__main__":
    pass
    # import torchvision.models as models
    # import torch
    # model = models.resnet50(weights='DEFAULT')
    # rand_input = torch.rand(1, 3, 224, 224)
    # bfe = StageFeatureExtractor(model, num_stages=1)
    # print(bfe.extract_features(rand_input).shape)
