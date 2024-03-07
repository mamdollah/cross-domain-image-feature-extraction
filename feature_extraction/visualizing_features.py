import os
import torchvision.models as models
from torchvision import transforms
from PIL import Image

# Loading pre-trained ResNet50 model
resnet50 = models.resnet50(pretrained=True)

# a hook function to store the intermediate feature maps
feature_maps = []


def hook(module, input, output):
    feature_maps.append(output)


# Register the hook to the desired layers
resnet50.layer1.register_forward_hook(hook)
resnet50.layer2.register_forward_hook(hook)
resnet50.layer3.register_forward_hook(hook)
resnet50.layer4.register_forward_hook(hook)


# Input image
input_image_path = "../images/shapes.png"
input_image = Image.open(input_image_path).convert("RGB")
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)  # Add batch dimension

# Forward pass to get feature maps
output = resnet50(input_batch)

# Access the feature maps from the list
feature_maps_layer1 = feature_maps[0]
feature_maps_layer2 = feature_maps[1]
feature_maps_layer3 = feature_maps[2]
feature_maps_layer4 = feature_maps[3]

# Save feature maps as common images
save_dir = os.getcwd()+"/resnet50-test"

# Ensure the directory exists, create it if not
os.makedirs(save_dir, exist_ok=True)

# Define a transformation to convert tensors to PIL images
to_pil = transforms.ToPILImage()

print("feature maps", len(feature_maps))

for layer, fmap in enumerate(feature_maps):
    layer_dir = save_dir + f"/layer-{layer}"
    os.makedirs(layer_dir, exist_ok=True)

    print(f"dimensions of layer {layer} features: ", fmap.shape)

    # Save feature maps as images
    for f in range(fmap.shape[1]):  # Iterate over channels
        # Extract individual channel
        channel_image = fmap[0, f, :, :].unsqueeze(0)

        # Convert tensor to PIL image
        pil_image = to_pil(channel_image)

        # Save as image file
        save_path = os.path.join(layer_dir, f"channel{f + 1}.png")
        pil_image.save(save_path)

    print(f"{f + 1} feature maps saved in {save_dir}")

print(f"\n Feature maps saved as images successfully.")


if __name__ == "__main__":
    pass