import torch
import torch.nn as nn
import cv2
import numpy as np
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights


class ResNetFeatureExtractor(nn.Module):
    def __init__(self):
        super(ResNetFeatureExtractor, self).__init__()
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.features = nn.Sequential(*list(resnet.children())[:-1])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        self._output_dim = self._calculate_output_dim()

    def extract_features(self, x):
        with torch.no_grad():
            x = self.process_image(x)
            return self.features(x)

    def process_image(self, image):
        # Transpose the image from (channels, height, width) to (height, width, channels) for cv2 operations
        #image = image.transpose((1, 2, 0))

        # Resize the image
        #image = cv2.resize(image.cpu().numpy(), (224, 224), interpolation=cv2.INTER_LINEAR)

        # Normalize the image
        image = image.cpu().numpy() / 255.0
        image -= np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
        image /= np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3 )

        # If needed, transpose back to (channels, height, width) for PyTorch or similar frameworks
        #image = image.transpose((2, 0, 1))

        return image

    def process_image1(self, image):
        # Convert image to float and scale to range [0, 1]
        image = image.cpu().numpy() / 255.0

        # Subtract mean RGB values
        mean_rgb = np.array([0.485, 0.456, 0.406])
        image -= mean_rgb.reshape(1, 1, 3)

        # Normalize by standard deviation (optional)
        std_rgb = np.array([0.229, 0.224, 0.225])
        image /= std_rgb.reshape(1, 1, 3)

        return image


    def _calculate_output_dim(self):
        # Logic to calculate output dimensions, similar to the previous example
        dummy_input = torch.rand(1, 3, 224, 224).to(self.device)
        with torch.no_grad():
            output = self.extract_features(dummy_input)
            output = self.reduce_feature_map(output)
        return output.shape[0], output.shape[1]


    @property
    def output_dim(self):
        return self._output_dim


