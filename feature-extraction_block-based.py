import torch
import torch.nn as nn
import torchvision.models as models
import torchinfo


class Custom_Model(nn.Module):
    def __init__(self, num_blocks: int):
        super(Custom_Model, self).__init__()
        rn50 = models.resnet50(pretrained=True)

        self.conv1 = rn50.conv1
        self.bn1 = rn50.bn1
        self.relu = rn50.relu
        self.maxpool = rn50.maxpool

        layers = [getattr(rn50, f"layer{i}") for i in range(1, 5)]
        self.get_blocks(num_blocks, layers)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.freeze_params()
        rn50 = None

    def get_blocks(self, num_blocks, layers):
        for i, layer in enumerate(layers, 1):
            if not num_blocks: break
            bottleneck_blocks = []
            for bottleneck in layer:
                if num_blocks:
                    bottleneck_blocks.append(bottleneck)
                    num_blocks -= 1
                else:
                    break
            setattr(self, f"layer{i}", nn.Sequential(*bottleneck_blocks))

    def freeze_params(self):
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        for seq in self.sequential_blocks:
            x = seq(x)
        return self.avgpool(x)


num_blocks = 5
custom_model = Custom_Model(num_blocks)

print(torchinfo.summary(custom_model))
