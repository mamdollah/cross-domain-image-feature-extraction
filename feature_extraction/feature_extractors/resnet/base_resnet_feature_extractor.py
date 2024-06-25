import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import wandb
from torchvision.transforms.v2 import ToPILImage
from feature_extraction.feature_extractors.base_feature_extractor import BaseFeatureExtractor
from utils import save_np_image, save_tensor_image, save_feature_maps, save_reduced_feature_map


class BaseResnetFeatureExtractor(BaseFeatureExtractor, nn.Module):
    def __init__(self, model, log_dir=None, log_to_wandb=False):
        nn.Module.__init__(self)  # Ensure nn.Module is initialized first
        super(BaseFeatureExtractor, self).__init__()
        self.input_width = 224
        self.input_height = 224
        self.model = model
        self.log_dir = log_dir
        self.log_to_wandb = log_to_wandb
        self.save_path = os.path.join(log_dir, "images")
        self.feature_embeddings_path = os.path.join(self.save_path, "feature_embeddings")
        self.reduced_dim_image_name = "reduced_dim_image.png"
        # Needs to be defined before calling _calculate_output_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Resnet Device is: " + str(self.device))


        self._output_dim = self._calculate_output_dim()
        self.frame_number = 0
        self.to_pil_image = ToPILImage()

    def _calculate_output_dim(self):
        # Logic to calculate output dimensions, similar to the previous example
        dummy_input = torch.rand(1, 3, 224, 224).to(self.device)
        with torch.no_grad():
            output = self.forward(dummy_input)
            output = self.reduce_feature_map(output)
        return output.shape[0], output.shape[1]


    @property
    def output_dim(self):
        return self._output_dim

    def forward(self, x):
        with torch.no_grad():
            for layer in self.model:
                if layer:
                    x = layer(x.to(self.device).float())
        return x

    @staticmethod
    def reduce_feature_map(feature_maps):
        return nn.AdaptiveAvgPool2d((1, 1))(feature_maps).view(feature_maps.size(0), -1)

#    @staticmethod
#    def reduce_feature_map(feature_map):
#         pooled_features = torch.mean(feature_map, dim=1)
#         flattened_features = pooled_features.view(pooled_features.size(0), -1)
#         return flattened_features  # shape (batch_size, height, width)

    def process_image(self, image):
        # Transpose the image from (channels, height, width) to (height, width, channels) for cv2 operations
        image = image.transpose((1, 2, 0))

        # Resize the image
        #image = cv2.resize(image, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)

        # Normalize the image
        image = image / 255.0
        image -= np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
        image /= np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)

        # If needed, transpose back to (channels, height, width) for PyTorch or similar frameworks
        image = image.transpose((2, 0, 1))

        return image

    def extract_features(self, image):
        self.frame_number += 1

        processed_image = self.process_image(image)

        image_tensor = torch.tensor(processed_image).unsqueeze(0)

        feature_maps = self.forward(image_tensor)
        reduced_dim = self.reduce_feature_map(feature_maps)

        if self.frame_number == 100 and self.log_dir:
            print(f"------- IMAGE PRE PROCESSING FOR FRAME NUMBER {self.frame_number} ------")
            print("Image shape is: " + str(image.shape))
            print("Image type is: " + str(type(image)))
            print("Image dtype is: " + str(image.dtype))
            print("Image max value", np.max(image))
            print("Image min value", np.min(image))

            print("------- IMAGE POST PROCESSING ------")
            print("Image shape is: " + str(processed_image.shape))
            print("Image type is: " + str(type(processed_image)))
            print("Image dtype is: " + str(processed_image.dtype))
            print("Image max value", np.max(processed_image))
            print("Image min value", np.min(processed_image))


            print("------- SAVING IMAGES LOCALLY ------")

            wandb_images = []
            print("Saving original image...")
            np_image = save_np_image(image, "original_image.png", self.save_path)
            print("Original image saved.")

            print("Saving processed image...")
            processed_image = save_np_image(processed_image,"processed_image.png", self.save_path)
            print("Processed image saved.")

            print("Saving feature embeddings...")
            feature_maps = save_feature_maps(feature_maps, self.feature_embeddings_path, 3)
            print("Feature embeddings saved.")

            print("------- IMAGES SAVED LOCALLY -------")

            reduced_feature_map_image = save_reduced_feature_map(reduced_dim, self.save_path, self.output_dim)

            if self.log_to_wandb:
                print("------- SAVING IMAGES TO WANDB -------")
                wandb_images.append(wandb.Image(np_image, caption="original_image"))
                wandb_images.append(wandb.Image(processed_image, caption="processed_image"))
                for idx, image in enumerate(feature_maps):
                    wandb_images.append(wandb.Image(image, caption=f"feature_map_{idx+1}"))
                wandb_images.append(wandb.Image(reduced_feature_map_image, caption="reduced_feature_map"))
                wandb.log({"Images": wandb_images})
                print("------- IMAGES SAVED TO WANDB -------")

        return reduced_dim.cpu()
