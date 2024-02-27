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
        rm50 = None

        custom_layers = []
        try:
            for nb, layer in zip([3, 4, 6, 3], [rn50.layer1, rn50.layer2, rn50.layer3, rn50.layer4]):
                if num_blocks > nb:
                    custom_layers.extend(list(layer)[:nb])
                    num_blocks -= nb
                elif num_blocks > 0:
                    custom_layers.extend(list(layer)[:num_blocks])
                    num_blocks = 0
        except ValueError:
            print("num_blocks must be in [1,...,16]")

        self.layers = nn.ModuleList(custom_layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.freeze_params()

    def freeze_params(self):
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        for layer in self.layers:
            x = layer(x)
        return self.avgpool(x)


num_blocks = 3
custom_model = Custom_Model(num_blocks)

print(torchinfo.summary(custom_model))
custom_model(torch.randn(1, 3, 224, 224)).shape