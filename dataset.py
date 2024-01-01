from torch.utils.data import Dataset
from torchvision.transforms import *
from queue import Queue
import glob
from PIL import Image
import random
import os

class Vimeo90KRandom(Dataset):
    """
    Randomized image patchifier with a buffer
    """

    NUM_SEQUENCES = 95936

    def __init__(self, patch_size, patch_per_img=20) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.root = '/home/xyhang/dataset/vimeo_septuplet/sequences/'
        self.patch_per_img = patch_per_img
        self.q = Queue()

    def __len__(self):
        return self.NUM_SEQUENCES * 7 * self.patch_per_img
    
    def random_sample(self):
        while True:
            seq_num = random.randrange(0, self.NUM_SEQUENCES - 1)
            num1 = seq_num // 1000 + 1
            num2 = seq_num % 1000 + 1
            img_num = random.randrange(1, 7)
            img_path = os.path.join(self.root, f"{num1:05d}", f"{num2:04d}", f"im{img_num}.png")
            if os.path.isfile(img_path):
                return img_path

    def __getitem__(self, _):
        if self.q.qsize() < 256:
            img = Image.open(self.random_sample())
            img = ToTensor()(img)

            cropper = RandomCrop(self.patch_size)
            for i in range(self.patch_per_img):
                patch = cropper(img)
                self.q.put(patch)
        patch = self.q.get()
        return patch

class Kodak(Dataset):
    """
    Randomized image patchifier with a buffer
    If stable: return image centers
    """

    def __init__(self, patch_size=512) -> None:
        super().__init__()
        dataset_glob = "/home/xyhang/dataset/kodak/*.png"
        self.image_list = glob.glob(dataset_glob)
        self.image_list.sort()
        self.len_image_list = len(self.image_list)
        self.patch_size = patch_size

    def __len__(self):
        return self.len_image_list

    def __getitem__(self, index):
        img = Image.open(self.image_list[index])
        img = ToTensor()(img)

        if self.patch_size is not None:
            cropper = CenterCrop(self.patch_size)
            return cropper(img)
        else:
            return img