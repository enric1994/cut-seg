from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

class ValDataset(Dataset):

    def __init__(self, root_dir):
        self.root_dir = root_dir
        images_path = os.path.join(root_dir, 'valB')
        masks_path = os.path.join(root_dir, 'valB_seg')
        self.images = os.listdir(images_path)
        self.masks = os.listdir(masks_path)


        self.transform = A.Compose([
            A.Resize(256,256),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # ToTensor
        # self.image2tensor = transforms.PILToTensor()

        assert len(self.images) == len(self.masks), 'Validation images and mask number must match'
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        A_img = cv2.imread(os.path.join(self.root_dir, 'valB', self.images[idx]))
        # A_img = cv2.cvtColor(A_img, cv2.COLOR_BGR2RGB)

        A_seg_img = cv2.imread(os.path.join(self.root_dir, 'valB_seg', self.masks[idx]), cv2.IMREAD_GRAYSCALE)//255


        A_transformed = self.transform(image=A_img, mask=A_seg_img)
        A = A_transformed['image']
        A_seg = A_transformed['mask'][None]

        return A, A_seg
