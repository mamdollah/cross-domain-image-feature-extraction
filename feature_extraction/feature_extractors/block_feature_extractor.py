import torch
import torch.nn as nn
from torchinfo import torchinfo
from torchvision import models
from torchvision.transforms import transforms

from feature_extraction.feature_extractors.feature_extractor import FeatureExtractor


class BlockFeatureExtractor(nn.Module, BaseFeatureExtractor):
    def __init__(self, model, num_blocks=0, num_stages=0):
        # super(BlockFeatureExtractor, self).__init__()
        nn.Module.__init__(self)
        self.num_blocks = num_blocks
        self.num_stages = num_stages

        # just for torchinfo.summary to show correct model architecture
        self.Conv2d, self.BatchNorm2d, self.ReLU, self.MaxPool2d = list(model.children())[:4]
        self.stages = self._get_block_features()
        self.stage1, self.stage2, self.stage3, self.stage4 = self.stages

        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.output_dim = self.output_dim()
        self.freeze_params()


    def freeze_params(self):
        for param in self.parameters():
            param.requires_grad = False

    def output_dim(self):
        dummy_input = torch.rand(1, 3, 224, 224)
        output = self.extract_features(dummy_input)
        return output.shape

    def process_image(self, image):
        image_processor = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return image_processor(image)

    def _get_block_features(self):
        seqs = [None] * 4
        sequentials = nn.Sequential(*list(model.children())[4:-6+self.num_stages])
        last_stage_blocks = list(sequentials[-1].children())
        num_blocks = self.num_blocks if self.num_blocks else len(last_stage_blocks)
        sequentials[-1] = nn.Sequential(*last_stage_blocks[:num_blocks])
        for idx, seq in enumerate(sequentials):
            seqs[idx] = seq
        return seqs

    def forward(self, x):
        with torch.no_grad():
            x = self.Conv2d(x)
            x = self.BatchNorm2d(x)
            x = self.ReLU(x)
            x = self.MaxPool2d(x)
            for stage in self.stages[:self.num_stages]:
                x = stage(x)
        return x

    def reduce_dim(self, features):
        reduced_features = self.adaptive_avg_pool(features)
        return reduced_features.view(features.size(0), -1)

    def extract_features(self, image):
        processed_image = self.process_image(image)
        feature_embeddings = self.forward(processed_image)
        reduced_dim = self.reduce_dim(feature_embeddings)
        return reduced_dim




if __name__ == "__main__":
    model = models.resnet50(weights='DEFAULT')
    rand_input = torch.rand(1, 3, 224, 224)

    custom_model = nn.Sequential(*list(model.children())[:-5]) # first stage
    feature_extractor = BlockFeatureExtractor(model, num_stages=1, num_blocks=1)

    output1 = feature_extractor.extract_features(rand_input)
    output2 = feature_extractor.reduce_dim(custom_model.forward(feature_extractor.process_image(rand_input)))
    # output2 = feature_extractor.extract_features(torch.rand(1, 3, 224, 224))

    print(output1.shape, output2.shape)
    print(feature_extractor.output_dim)
    torchinfo.summary(feature_extractor)

    # Check if the outputs are the same
    if torch.allclose(output1, output2):
        print("Outputs of the models are the same.")
    else:
        print("Outputs of the models are different.")
