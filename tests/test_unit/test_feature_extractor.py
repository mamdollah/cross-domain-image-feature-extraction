import unittest

import numpy as np
import torch
from torch import nn
from torchvision import models
from torchvision.models import ResNet50_Weights

from feature_extraction.feature_extractors.resnet.block_feature_extractor import BlockFeatureExtractor
from feature_extraction.feature_extractors.resnet.stage_feature_extractor import StageFeatureExtractor


def custom_resnet(model, stages=1, **kwargs):
    if stages < 1 or stages > 4 or not isinstance(stages, int):
        raise ValueError("Number of stages must be between 1 and 4.")

    layers = {
        1: [3, 0, 0, 0],
        2: [3, 4, 0, 0],
        3: [3, 4, 6, 0],
        4: [3, 4, 6, 3]
    }

    custom_model = models.resnet.ResNet(models.resnet.Bottleneck, layers[stages], **kwargs)
    custom_model.fc = nn.Identity()

    # Ensure that layers beyond the specified stages are set to nn.Identity()
    for i, (name, module) in enumerate(custom_model.named_children()):
        if i >= stages:
            setattr(custom_model, name, nn.Identity())

    # Load weights from the original ResNet50 model
    own_state = custom_model.state_dict()
    pretrained_state = model.state_dict()

    for name, param in pretrained_state.items():
        if name in own_state and param.shape == own_state[name].shape:
            own_state[name].copy_(param)

    return custom_model


class TestFeatureExtractors(unittest.TestCase):
    def setUp(self):
        self.num_stage = 1
        self.num_blocks = 3
        self.dummy_input = torch.rand(1, 3, 224, 224)
        self.model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.stage_model = custom_resnet(self.model, stages=self.num_stage)


    def test_stage_feature_extractor(self):
        stage_extractor = StageFeatureExtractor(self.model, num_stages=self.num_stage)
        output = stage_extractor.extract_features(self.dummy_input)
        custom_resnet_output = self.stage_model(self.dummy_input)
        self.assertEqual(output.shape, custom_resnet_output.shape)
        np.testing.assert_allclose(output, custom_resnet_output, rtol=1e-5, atol=1e-5)

    def test_stage_equal_block(self, stage=1, block=3):
        block_extractor = BlockFeatureExtractor(self.model, num_blocks=block)
        stage_extractor = StageFeatureExtractor(self.model, num_stages=stage)
        stage_output = stage_extractor.extract_features(self.dummy_input)
        block_output = block_extractor.extract_features(self.dummy_input)
        self.assertEqual(block_output.shape, stage_output.shape)
        np.testing.assert_allclose(block_output, stage_output, rtol=1e-5, atol=1e-5)

    def test_stages_blocks_equal(self):
        for stage, block in zip(range(1, 5), [3, 7, 13, 16]):
            self.test_stage_equal_block(stage, block)




if __name__ == '__main__':
    unittest.main()
