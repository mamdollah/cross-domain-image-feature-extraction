import torch.nn as nn
from feature_extraction.feature_extractors.resnet.base_resnet_feature_extractor import BaseResnetFeatureExtractor

class BlockFeatureExtractor(BaseResnetFeatureExtractor):
    def __init__(self, model, num_blocks=3):
        if num_blocks < 1 or num_blocks > 16 or not isinstance(num_blocks, int):
            raise ValueError("Number of blocks must be greater than 0.")

        children = list(model.children())
        blocks_per_stage = [len(list(stage.children())) for stage in children[4:8]]
        num_stage = next((i+1 for i, blocks in enumerate(blocks_per_stage) if sum(blocks_per_stage[:i+1]) >= num_blocks), None)

        model = children[:4+num_stage]
        model[-1] = model[-1][:num_blocks]

        super().__init__(model)
