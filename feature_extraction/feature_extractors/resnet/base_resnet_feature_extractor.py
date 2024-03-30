import numpy as np
import torch
import torch.nn as nn
from abc import abstractmethod

from PIL import Image
from torchinfo import torchinfo
from torchvision.transforms import transforms

from feature_extraction.feature_extractors.base_feature_extractor import BaseFeatureExtractor


class BaseResnetFeatureExtractor(BaseFeatureExtractor, nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.freeze_params()
        self.image_processor = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
            transforms.Resize((224, 224)),  # Resize to 224x224
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
        ])

    def freeze_params(self):
        for param in self.parameters():
            param.requires_grad = False

    @property
    def output_dim(self):
        dummy_input = torch.rand(1, 3, 224, 224)
        output = self.forward(dummy_input)
        output = self.reduce_dim(output)
        return output.shape[0], output.shape[1]

    def process_image(self, image_np):
        # Ensure image_np has no extra leading dimensions (e.g., shape (1, H, W))
        if image_np.ndim > 2 and image_np.shape[0] == 1:  # Assuming the shape is (1, H, W)
            image_np = image_np.squeeze(0)  # Now shape is (H, W)

        # Check if the image is grayscale (H, W) and convert it to RGB (H, W, C) by repeating the channels
        if image_np.ndim == 2:  # Grayscale image, needs to be converted to RGB
            image_np = np.repeat(image_np[:, :, np.newaxis], 3, axis=2)  # Now shape is (H, W, C)

        # Convert the NumPy array to a PIL Image
        image_pil = Image.fromarray(image_np.astype('uint8'), 'RGB')  # Ensure the type is uint8 and mode is RGB

        return self.image_processor(image_pil).unsqueeze(0)

    def forward(self, x):
        with torch.no_grad():
            for layer in self.model:
                if layer:
                    x = layer(x)
        return x

    @staticmethod
    def reduce_dim(features):
        return nn.AdaptiveAvgPool2d((1, 1))(features).view(features.size(0), -1)

    def extract_features(self, image):
        processed_image = self.process_image(image)
        feature_embeddings = self.forward(processed_image)
        reduced_dim = self.reduce_dim(feature_embeddings)
        return reduced_dim






