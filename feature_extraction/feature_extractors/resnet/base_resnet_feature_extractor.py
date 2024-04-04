import os

import numpy as np
import torch
import torch.nn as nn
from abc import abstractmethod

from PIL import Image
from torchinfo import torchinfo
from torchvision.transforms import transforms
from torchvision.transforms.v2 import ToPILImage

from feature_extraction.feature_extractors.base_feature_extractor import BaseFeatureExtractor


class BaseResnetFeatureExtractor(BaseFeatureExtractor, nn.Module):
    def __init__(self, model, log_dir=None, log_to_wandb=False):
        nn.Module.__init__(self)  # Ensure nn.Module is initialized first
        super(BaseFeatureExtractor, self).__init__()
        self.model = model
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "images")
        self.feature_embeddings_path = os.path.join(self.save_path, "feature_embeddings")
        self.original_image_name = "original_image.png"
        self.reduced_dim_image_name = "reduced_dim_image.png"


        # Needs to be defined before calling _calculate_output_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._output_dim = self._calculate_output_dim()
        self.frame_number = 0
        self.to_pil_image = ToPILImage()

    def _calculate_output_dim(self):
        # Logic to calculate output dimensions, similar to the previous example
        dummy_input = torch.rand(1, 3, 224, 224).to(self.device)
        with torch.no_grad():
            output = self.forward(dummy_input)
            output = self.reduce_dim(output)
        return output.shape[0], output.shape[1]


    @property
    def output_dim(self):
        return self._output_dim

    def forward(self, x):
        with torch.no_grad():
            for layer in self.model:
                if layer:
                    x = layer(x)
        return x

    @staticmethod
    def reduce_dim(features):
        return nn.AdaptiveAvgPool2d((1, 1))(features).view(features.size(0), -1)

    def save_feature_embeddings(self, feature_embeddings):
        print("Shape of feature embeddings: ", feature_embeddings.shape)
        # Ensure the save directory exists
        os.makedirs(self.feature_embeddings_path, exist_ok=True)

        batch_size, channels, height, width = feature_embeddings.shape
        for batch_index in range(batch_size):
            for channel_index in range(channels):
                # Extract the single-channel image
                single_channel_image = feature_embeddings[batch_index, channel_index]

                # Normalize the single-channel image
                #image_min = single_channel_image.min()
                #image_max = single_channel_image.max()
                #normalized_image = (single_channel_image - image_min) / (image_max - image_min)

                # Convert the normalized single-channel image to a PIL image
                pil_image = self.to_pil_image(single_channel_image.cpu())

                # Save the PIL image
                image_path = os.path.join(self.feature_embeddings_path,
                                          f"embedding_b{batch_index}_c{channel_index}.png")

                pil_image.save(image_path)

    def save_np_image(self, image, image_name):
        os.makedirs(self.save_path, exist_ok=True)
        image_path = os.path.join(self.save_path, image_name)
        # Transpose the image from (Channels, Height, Width) to (Height, Width, Channels)
        image_transposed = np.transpose(image, (1, 2, 0))

        print("Image_transposed shape: ", image_transposed.shape)

        # Convert to PIL Image and save
        pil_image = Image.fromarray(image_transposed.astype(np.uint8))  # Ensure it's uint8
        pil_image.save(image_path)
        print("Original image saved.")

    def save_tensor_image(self, tensor, image_name):
        print("Tensor image shape: ", tensor.shape)
        os.makedirs(self.save_path, exist_ok=True)
        image_path = os.path.join(self.save_path, image_name)

        # Ensure tensor is in CPU and convert to NumPy
        image_np = tensor.cpu().numpy()

        # Transpose from (Channels, Height, Width) to (Height, Width, Channels) and ensure RGB
        image_transposed = np.transpose(image_np, (1, 2, 0))

        # Convert to PIL Image and save
        pil_image = Image.fromarray(image_transposed.astype(np.uint8))
        pil_image.save(image_path)
        print("Tensor image saved.")


    def extract_features(self, image):

        self.frame_number += 1
        image_tensor = torch.tensor(image).unsqueeze(0).to(self.device).float()

        feature_embeddings = self.forward(image_tensor)
        reduced_dim = self.reduce_dim(feature_embeddings)

        if self.frame_number == 100 and self.log_dir:
            print("Saving original image...")
            self.save_np_image(image, self.original_image_name)
            self.save_tensor_image(image_tensor[0], "tensor_image.png")

            print("Saving tensor_0")
            pil_image = self.to_pil_image(image_tensor[0])
            pil_image.save(os.path.join(self.save_path, "image_tensor_0.png"))
            # save image_tensor_0 as image


            print("Original image saved.")

            print("Saving feature embeddings...")
            self.save_feature_embeddings(feature_embeddings)
            print("Feature embeddings saved.")


        return reduced_dim.cpu()