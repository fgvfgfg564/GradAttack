from torch.utils.data import Dataset
from torchvision.transforms import *
import glob
from PIL import Image
import random
import os

class SingleImage(Dataset):
    def __init__(self, img, length) -> None:
        super().__init__()
        self.length = length
        self.img = img
    
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.img


class Vimeo90KRandom(Dataset):
    NUM_SEQUENCES = 95936
    ROOT = os.path.expanduser('~/dataset/vimeo_septuplet/sequences/')

    def __init__(self, patch_size) -> None:
        self.patch_size = patch_size

    def __len__(self):
        return self.NUM_SEQUENCES * 7

    def random_sample(self):
        while True:
            seq_num = random.randrange(0, self.NUM_SEQUENCES - 1)
            num1 = seq_num // 1000 + 1
            num2 = seq_num % 1000 + 1
            img_num = random.randrange(1, 7)
            img_path = os.path.join(self.ROOT, f"{num1:05d}", f"{num2:04d}", f"im{img_num}.png")
            if os.path.isfile(img_path):
                return img_path
    
    def __getitem__(self, _):
        img = Image.open(self.random_sample())
        img = ToTensor()(img)
        if self.patch_size is not None:
            img = RandomCrop(self.patch_size)(img)
        return img

class LIU4KPatches(Dataset):
    """
    Patch size = 256
    """
    GLOB = os.path.expanduser('~/dataset/LIU4K_patch/*/*.png')

    def __init__(self):
        self.image_list = glob.glob(self.GLOB)
        self.len_image_list = len(self.image_list)
        self.patch_size = 256
    
    def __len__(self):
        return self.len_image_list
    
    def __getitem__(self, idx):
        filename = self.image_list[idx]
        img = Image.open(filename)
        img = ToTensor()(img)
        return img

class Kodak(Dataset):
    """
    Randomized image patchifier with a buffer
    If stable: return image centers
    """

    def __init__(self, patch_size=512) -> None:
        super().__init__()
        dataset_glob = os.path.expanduser('~/dataset/kodak/*.png')
        self.image_list = glob.glob(dataset_glob)
        self.len_image_list = len(self.image_list)
        self.patch_size = patch_size

    def __len__(self):
        return self.len_image_list

    def __getitem__(self, index):
        img = Image.open(self.image_list[index])
        img = ToTensor()(img)

        if self.patch_size is not None:
            cropper = CenterCrop(self.patch_size)
            img = cropper(img)
        return img