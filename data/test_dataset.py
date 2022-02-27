from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.datasets.folder import pil_loader
import numpy as np

class TestDataset(Dataset):

    def __init__(self, root_dir):
        self.root_dir = root_dir
        images_path = os.path.join(root_dir, 'images')
        masks_path = os.path.join(root_dir, 'masks')
        self.images = os.listdir(images_path)
        self.masks = os.listdir(masks_path)

        self.testsize = 256

        # self.transform_image = transforms.Compose([
        #     transforms.Resize((self.testsize, self.testsize)),
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # ])

        # self.transform_mask = transforms.Compose([
        #     transforms.Resize((self.testsize, self.testsize)),
        #     transforms.ToTensor()
        #     # transforms.Normalize((0.5,), (0.5,))
        # ])

        self.mean=(0.5, 0.5, 0.5)
        self.std= (0.5, 0.5, 0.5)
        self.transform = A.Compose([
            A.Resize(256,256),
            A.Normalize(mean=self.mean, std=self.std),
            ToTensorV2()
        ])


        assert len(self.images) == len(self.masks), 'Test images and mask number must match'
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = np.asarray(pil_loader(os.path.join(self.root_dir, 'images', self.images[idx])))
        mask = cv2.imread(os.path.join(self.root_dir, 'masks',self.masks[idx]), cv2.IMREAD_GRAYSCALE)//255

        data_transformed = self.transform(image=image, mask=mask)
        return data_transformed['image'], data_transformed['mask'][None]
        # return self.transform_image(image), self.transform_mask(mask)
