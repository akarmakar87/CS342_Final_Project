import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# New occlusion patch size and stride to match the 512x512 image
patch_size = 8  # Size of the patch
stride = 8  # Stride for sliding the patch

# Assuming the tensor is already preprocessed and ready to be fed into the model
# For the sake of demonstration, we'll create a dummy tensor with the correct shape
# input_tensor = torch.rand((1, 3, 512, 512))  # [batch_size, channels, height, width]

# Load the image
image_path = "sampled_images/mannerism_late_renaissance.jpg"  # Replace with the path to your image
image = Image.open(image_path).convert('RGB')

# Define transformations
class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = max([w, h])
        hp = int((max_wh - w) // 2)
        vp = int((max_wh - h) // 2)
        padding = (hp, hp, vp, vp)
        if hp * 2 + w < 256:
            padding = (hp, hp + 1, vp, vp)
        if vp * 2 + h < 256:
            padding = (hp, hp, vp, vp + 1)
            
        image_tensor = transforms.functional.pil_to_tensor(image)
#         image_tensor = torch.tensor(np.array(image), dtype=torch.float).permute(2,0,1)
        padded_tensor = F.pad(image_tensor, padding, mode='replicate')

        return padded_tensor


def resize_larger_dimension(image, size):
    width, height = image.size
    aspect_ratio = width / height
    if width > height:
        new_width = size
        new_height = int(size / aspect_ratio)
    else:
        new_width = int(size * aspect_ratio)
        new_height = size

    return image.resize((new_width, new_height))

transform = transforms.Compose([
    transforms.Lambda(lambda x: resize_larger_dimension(x, 512)),  # Resize the larger dimension to 256 while preserving aspect ratio
    SquarePad(),
#     transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
])

# Apply transformations
image_tensor = transform(image)

# If needed, add an extra dimension to represent batch size
input_tensor = image_tensor.unsqueeze(0)

# Create an empty heatmap
heatmap = np.zeros((512 // stride, 512 // stride))

# Slide the occlusion patch over the image and create the heatmap
for y in range(0, 512, stride):
    for x in range(0, 512, stride):
        # Create a copy of the input tensor to modify it
        occluded = input_tensor.clone()
        
        # Apply the occlusion patch
        # We set the pixels to 0 (black) within the occluded area
        occluded[:, :, y:y+patch_size, x:x+patch_size] = 0
        
        # Forward pass with the occluded image
        with torch.no_grad():
            occluded_output = model(occluded)
        
        # Use the probability of the correct class for the heatmap
        # Assuming class 0 is the correct class for demonstration
        heatmap[y // stride, x // stride] = occluded_output[0, 0].item()

# Normalize the heatmap
heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))

# Display the heatmap
plt.imshow(heatmap, cmap='viridis', extent=[0, 512, 512, 0])
plt.colorbar()
plt.show()

transform_resize_pad = transforms.Compose([
    transforms.Lambda(lambda x: resize_larger_dimension(x, 512)),  # Resize the larger dimension to 256 while preserving aspect ratio
    SquarePad(),
#     transforms.ToTensor(),
])

# Apply transformations
resize_pad_image_tensor = transform_resize_pad(image)

# print(resize_pad_image_tensor.shape)

# Convert tensor to a PIL Image
original_image = transforms.ToPILImage()(resize_pad_image_tensor)

# Display or save the PIL Image
original_image.show()
