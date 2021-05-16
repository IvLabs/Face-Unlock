from typing import Callable, Optional
from torchvision import transforms
from  torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

# ref: https://pytorch.org/vision/stable/_modules/torchvision/datasets/folder.html#ImageFolder


class AttDataset(ImageFolder):

    def __init__(
        self,
        path: str,
        transform: Optional[Callable] = transforms.ToTensor()
    ):
        super(AttDataset, self).__init__(path,transform)

    process = transforms.Compose([
        transforms.Resize((224,224), interpolation= transforms.InterpolationMode.BICUBIC),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.4422, std=0.1931),
    ])


def is_valid(s):
    return True
# A hack to get over with extension chechinkg of DatsetFolder class
# "Supported extensions are: .jpg,.jpeg,.png,.ppm,.bmp,.pgm,.tif,.tiff,.webp"

class YaleDataset(ImageFolder):
    # Ref : http://vision.ucsd.edu/content/yale-face-database
    def __init__(
        self,
        path: str,
        transform: Optional[Callable] = transforms.ToTensor()
    ):
        super(YaleDataset, self).__init__(path,transform, is_valid_file = is_valid)

    process = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])