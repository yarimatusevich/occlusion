import torch
from torchvision.transforms import v2
from torchvision import datasets
from PIL.Image import Image as PILImage

HEIGHT, WIDTH = 128, 128
channel_means = [0.5, 0.5, 0.5]
channel_stds = [0.5, 0.5, 0.5]

train_transforms = v2.Compose([
    v2.RGB(),
    v2.Resize(size=(HEIGHT, WIDTH), antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomRotation(degrees=10),
    v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=channel_means, std=channel_stds)
])

inference_transforms = v2.Compose([
    v2.RGB(),
    v2.Resize(size=(HEIGHT, WIDTH), antialias=True),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=channel_means, std=channel_stds)
])

def transform_training_data(filepath: str):
    return datasets.ImageFolder(filepath, transform=train_transforms)

def transform_image_for_inference(image: PILImage):
    return inference_transforms(image)
