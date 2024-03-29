import torch.nn as nn
from feature_extraction.feature_extractors.resnet.base_resnet_feature_extractor import BaseResnetFeatureExtractor

class BlockFeatureExtractor(BaseResnetFeatureExtractor):
    def __init__(self, model, num_blocks=2):
        self.num_blocks = num_blocks
        super().__init__(model)

    def _get_features(self):
        sequentials = [None] * 4
        block_count = 0
        for i, stage in enumerate(self.stages):
            blocks_in_stage = len(list(stage.children()))
            if (block_count + blocks_in_stage) < self.num_blocks:
                sequentials[i] = stage
                block_count += blocks_in_stage
            else:
                sequentials[i] = nn.Sequential(*list(stage.children())[:self.num_blocks - block_count])
                break
        return sequentials

if __name__ == "__main__":
    # pass
    import torchvision.models as models
    import torchinfo
    import torch
    model = models.resnet50(weights='DEFAULT')
    rand_input = torch.rand(1, 3, 224, 224)
    bfe = BlockFeatureExtractor(model, num_blocks=16)
    print(bfe.extract_features(rand_input).shape)
    torchinfo.summary(bfe)

