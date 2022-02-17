from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image

class TestDataset(Dataset):

    def __init__(self, root_dir):
        self.root_dir = root_dir
        images_path = os.path.join(root_dir, 'images')
        masks_path = os.path.join(root_dir, 'masks')
        self.images = os.listdir(images_path)
        self.masks = os.listdir(masks_path)

        self.testsize = 256

        self.transform_image = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.transform_mask = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor()
            # transforms.Normalize((0.5,), (0.5,))
        ])

        assert len(self.images) == len(self.masks), 'Test images and mask number must match'
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.root_dir, 'images', self.images[idx])).convert('RGB')
        mask = Image.open(os.path.join(self.root_dir, 'masks',self.masks[idx])).convert('L')

        return self.transform_image(image), self.transform_mask(mask)
