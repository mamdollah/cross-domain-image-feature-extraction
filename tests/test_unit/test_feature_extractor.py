import unittest

import numpy as np
import torch
from torch import nn
from torchvision import models
from torchvision.models import ResNet50_Weights

from feature_extraction.feature_extractors.resnet.block_feature_extractor import BlockFeatureExtractor
from feature_extraction.feature_extractors.resnet.stage_feature_extractor import StageFeatureExtractor


def custom_resnet(model, stages=1, blocks=3, **kwargs):
    if stages < 1 or stages > 4 or not isinstance(stages, int):
        raise ValueError("Number of stages must be between 1 and 4.")

    layers = {
        1: [3, 0, 0, 0],
        2: [3, 4, 0, 0],
        3: [3, 4, 6, 0],
        4: [3, 4, 6, 3]
    }
    if blocks > 0:
        layers[stages][stages - 1] = blocks

    custom_model = models.resnet.ResNet(models.resnet.Bottleneck, layers[stages], **kwargs)
    custom_model.fc = nn.Identity()

    if stages < 4:
        custom_model.layer4 = nn.Identity()
        if stages < 3:
            custom_model.layer3 = nn.Identity()
            if stages < 2:
                custom_model.layer2 = nn.Identity()


    own_state = custom_model.state_dict()
    pretrained_state = model.state_dict()

    for name, param in pretrained_state.items():
        if name.startswith('layer'):
            if name not in own_state:
                continue
            own_param = own_state[name]
            if isinstance(own_param, nn.Identity):
                continue
            own_param.copy_(param)

    return custom_model


class TestFeatureExtractors(unittest.TestCase):
    def setUp(self):
        self.num_stage = 2
        self.num_blocks = 7
        self.dummy_input = torch.rand(1, 3, 224, 224)
        self.model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)


    def test_block_feature_extractor(self, stage=1, block=3):
        block_extractor = BlockFeatureExtractor(self.model, num_blocks=block)
        custom_block_fe = custom_resnet(self.model, stages=stage)
        stage_output = custom_block_fe.forward(self.dummy_input).detach().numpy()
        block_output = block_extractor.extract_features(self.dummy_input)
        self.assertEqual(block_output.shape, stage_output.shape)
        np.testing.assert_allclose(block_output, stage_output, rtol=1e-5, atol=1e-5)

    def test_stage_feature_extractor(self):
        stage_extractor = StageFeatureExtractor(self.model, num_stages=self.num_stage)
        output = stage_extractor.extract_features(self.dummy_input)
        self.assertEqual(output.shape, self.stage_model.forward(self.dummy_input).shape)

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
