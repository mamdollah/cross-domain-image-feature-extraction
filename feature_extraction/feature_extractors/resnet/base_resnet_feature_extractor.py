import os
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
        self.model = model
        self.log_dir = log_dir
        self.log_to_wandb = log_to_wandb
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
            output = self.reduce_feature_map(output)
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
    def reduce_feature_map(feature_maps):
        return nn.AdaptiveAvgPool2d((1, 1))(feature_maps).view(feature_maps.size(0), -1)

    # @staticmethod
    # def reduce_feature_map(feature_map):
    #     pooled_features = torch.mean(feature_map, dim=1)
    #     flattened_features = pooled_features.view(pooled_features.size(0), -1)
    #     return flattened_features  # shape (batch_size, height, width)

    def extract_features(self, image):

        self.frame_number += 1
        image_tensor = torch.tensor(image).unsqueeze(0).to(self.device).float()

        feature_maps = self.forward(image_tensor)
        reduced_dim = self.reduce_feature_map(feature_maps)

        if self.frame_number == 100 and self.log_dir:
            wandb_images = []
            print("Saving original image...")
            np_image = save_np_image(image, self.original_image_name, self.save_path)
            tensor_image = save_tensor_image(image_tensor[0], "tensor_image.png", self.save_path)


            print("Saving tensor_0")
            pil_image = self.to_pil_image(image_tensor[0])
            pil_image.save(os.path.join(self.save_path, "image_tensor_0.png"))


            print("Original image saved.")

            print("Saving feature embeddings...")
            feature_maps = save_feature_maps(feature_maps, self.feature_embeddings_path, 3)
            print("Feature embeddings saved.")

            reduced_feature_map_image = save_reduced_feature_map(reduced_dim, self.save_path, self.output_dim)

            if self.log_to_wandb:
                wandb_images.append(wandb.Image(np_image, caption="original_image_np"))
                for idx, image in enumerate(feature_maps):
                    wandb_images.append(wandb.Image(image, caption=f"feature_map_{idx+1}"))
                wandb_images.append(wandb.Image(reduced_feature_map_image, caption="reduced_feature_map"))
                wandb.log({"Images": wandb_images})



        return reduced_dim.cpu()
