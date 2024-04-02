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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to_pil = ToPILImage()

        self.frame_number = 0

    def freeze_params(self):
        for param in self.parameters():
            param.requires_grad = False

    @property
    def output_dim(self):
        dummy_input = torch.rand(4, 3, 224, 224).to(self.device)
        output = self.forward(dummy_input)
        output = self.reduce_dim(output)
        return output.shape[0], output.shape[1]

    def process_image(self, image_np):
        self.frame_number += 1

        #Saving preprocessed image


        if image_np.ndim > 2 and image_np.shape[0] == 1:  # Assuming the shape is (1, H, W)
            image_np = image_np.squeeze(0)  # Now shape is (H, W)

        if self.frame_number == 100:
            # Save the image as a PNG file
            print(f"Saving original_frame_{self.frame_number}.png")
            image_pil = Image.fromarray(image_np.astype('uint8'), 'L')  # 'L' mode for grayscale
            image_pil.save(f"frame_{self.frame_number}.png")

        # Check if the image is grayscale (H, W) and convert it to RGB (H, W, C) by repeating the channels
        if image_np.ndim == 2:  # Grayscale image, needs to be converted to RGB
            image_np = np.repeat(image_np[:, :, np.newaxis], 3, axis=2)  # Now shape is (H, W, C)

        # Convert the NumPy array to a PIL Image
        image_pil = Image.fromarray(image_np.astype('uint8'), 'RGB')

        # Apply the image processing transformations
        processed_tensor = self.image_processor(image_pil)

        if self.frame_number == 100:
            # Convert the processed tensor back to a PIL image for saving
            processed_image_pil = self.to_pil(processed_tensor)
            processed_image_pil.save(f"processed_frame_{self.frame_number}.png")

        return processed_tensor.unsqueeze(0)

    def process_image_stack(self, image_stack_np):
        # Shape should be (4, 84, 84) for atari imgaes. with stack = 4
        processed_images = []

        # Iterate through each image in the batch
        for i in range(image_stack_np.shape[0]):
            image_np = image_stack_np[i]  # Extract the i-th grayscale image
            # Process the image
            processed_image = self.process_image(image_np)
            processed_images.append(processed_image)

        # Concatenate all processed images along the batch dimension to form a batch tensor
        batch_tensor = torch.cat(processed_images, dim=0)

        return batch_tensor.to(self.device)

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

        return reduced_dim.cpu()

    def extract_features_stack(self, image_stack):
        (4, 84, 84)
        processed_image = self.process_image_stack(image_stack)  # Process image stack
        (4, 3, 256, 256)
        feature_embeddings = self.forward(processed_image)  # Get feature embeddings
        (4, 256, (56*56))
        reduced_dim = self.reduce_dim(feature_embeddings)  # Reduce dimensions
        (4, 256)

        save_dir = "feature_embeddings_images"
        os.makedirs(save_dir, exist_ok=True)

        if self.frame_number == 100:

            for batch_index in range(feature_embeddings.shape[0]):  # Loop through the batch
                for channel_index in range(feature_embeddings.shape[1]):  # Loop through each channel
                    # Extract the single feature map
                    feature_map = feature_embeddings[batch_index, channel_index, :, :]

                    # Convert to PIL image
                    img = self.to_pil(feature_map.cpu())  # Convert tensor to PIL Image

                    # Construct the filename for each feature map
                    filename = f"batch{batch_index}_channel{channel_index}.png"

                    # Save the image
                    img.save(os.path.join(save_dir, filename))

            reduced_dir = "reduced_dim_images"
            os.makedirs(reduced_dir, exist_ok=True)

            # Convert each batch's reduced feature set to an image
            for i in range(reduced_dim.shape[0]):  # Loop through the batch
                # Normalize the features to [0, 1] for better visualization
                features = reduced_dim[i].detach().cpu().numpy()
                min_val, max_val = features.min(), features.max()
                features = (features - min_val) / (max_val - min_val)

                # Reshape or repeat features to make them visually interpretable
                image_data = np.reshape(features, (16, 16))

                # Convert numpy array to tensor
                image_data_tensor = torch.tensor(image_data).float().unsqueeze(0)  # Add channel dimension
                img = self.to_pil(image_data_tensor)  # Convert tensor to PIL Image

                # Save the image
                img.save(os.path.join(reduced_dir, f'reduced_dim_batch_{i}.png'))

        return reduced_dim



