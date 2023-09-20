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


class OverlapCrop(object):
    def __init__(self, patch_size=(64, 64, 64), patch_overlap=(32, 32, 32), padding=None):
        """
        Initialize the OverlapCrop transform.

        Args:
            patch_size (tuple): Size of each patch (depth, height, width).
            patch_overlap (tuple): Amount of overlap between patches (depth, height, width).
            padding (tuple or None): Padding to apply to the input data, if needed.
        """
        super().__init__()
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.padding = padding

    def __call__(self, subject):
        if not isinstance(subject, tio.Subject):
            raise ValueError("Input must be a TorchIO Subject.")

        sampler = tio.GridSampler(
            subject,
            patch_size=self.patch_size,
            patch_overlap=self.patch_overlap,
            padding_mode=self.padding,
        )
        patches = sampler(subject)

        return patches
