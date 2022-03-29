from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
from torchvision.datasets.folder import pil_loader
import numpy as np

class ValDataset(Dataset):

    def __init__(self, root_dir, crop_size):
        self.root_dir = root_dir
        images_path = os.path.join(root_dir, 'valB')
        masks_path = os.path.join(root_dir, 'valB_seg')
        synth_masks_path = os.path.join(root_dir, 'valA_seg')
        synth_path = os.path.join(root_dir, 'valA')
        self.images = os.listdir(images_path)
        self.masks = os.listdir(masks_path)
        self.synth = os.listdir(synth_path)
        self.synth_masks = os.listdir(synth_masks_path)

        # self.mean=(0.485, 0.456, 0.406)
        # self.std=(0.229, 0.224, 0.225)
        self.mean=(0.5,0.5,0.5)
        self.std=(0.5,0.5,0.5)
        # It is not clear which mean and std we should use in validation and test, it seems that netS trains with perfectly normalized images... 
        self.transform = A.Compose([
            A.Resize(crop_size,crop_size),
            A.Normalize(mean=self.mean, std=self.std),
            ToTensorV2()
        ])
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # ToTensor
        # self.image2tensor = transforms.PILToTensor()
        assert len(self.images) == len(self.masks), 'Validation images and mask number must match'
        assert len(self.synth) == len(self.images), 'Validation images and synth number must match'

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        # A_img = cv2.imread(os.path.join(self.root_dir, 'valB', self.images[idx]))
        # A_img = cv2.cvtColor(A_img, cv2.COLOR_BGR2RGB)
        B_img = np.asarray(pil_loader(os.path.join(self.root_dir, 'valB', self.images[idx])))
        
        A_img = np.asarray(pil_loader(os.path.join(self.root_dir, 'valA', self.synth[idx])))

        B_seg_img = cv2.imread(os.path.join(self.root_dir, 'valB_seg', self.masks[idx]), cv2.IMREAD_GRAYSCALE)//255
        
        A_seg_img = cv2.imread(os.path.join(self.root_dir, 'valA_seg', self.synth_masks[idx]), cv2.IMREAD_GRAYSCALE)//255


        A_transformed = self.transform(image=A_img, mask=A_seg_img)
        A = A_transformed['image']
        A_seg = A_transformed['mask'][None]

        B_transformed = self.transform(image=B_img, mask=B_seg_img)
        B = B_transformed['image']
        B_seg = B_transformed['mask'][None]
        
        return A, A_seg, B, B_seg



