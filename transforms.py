import torchio
import torchvision
import torchio as tio
from torchio.transforms import Crop, Pad, CropOrPad


def resize_normalize(size: int, component: str):
    if component == "image":
        resize_normalize_transform = torchvision.transforms.Compose([
            tio.transforms.Resize(size, image_interpolation="linear"),
            tio.transforms.ZNormalization(),
        ])
    elif component == "mask":
        resize_normalize_transform = torchvision.transforms.Compose([
            tio.transforms.Resize(64, image_interpolation="nearest"),
        ])
    else:
        raise ValueError(f"Invalid component '{component}'. Input component must be 'image' or 'mask'.")
    return resize_normalize_transform
