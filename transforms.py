import torchio
import torchvision
import torchio as tio
from torchio.transforms import Crop, CropOrPad

resize_transform_pet = torchvision.transforms.Compose([
    tio.transforms.Resize(64, image_interpolation="linear"),
    tio.transforms.ZNormalization(),
])

resize_transform_suv = torchvision.transforms.Compose([
    tio.transforms.Resize(64, image_interpolation="linear"),
    tio.transforms.ZNormalization(),
])

resize_transform_seg = torchvision.transforms.Compose([
    tio.transforms.Resize(64, image_interpolation="nearest"),
])


def resize_normalize_transform():
    torchvision.transforms.Compose([
        # insert overlap crop transform
        tio.transforms.ZNormalization(),
    ])


class OverlapCrop(object):
    """
    Generate overlapping cropped blocks from volume sample.
    """

    def __init__(self, overlap, output_size):
        assert isinstance(overlap, float)
        if 0 < overlap <
        pass

    def __call__(self, sample_volume):
        pass


