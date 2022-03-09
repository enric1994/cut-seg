import os.path
from data.base_dataset import BaseDataset, get_transform, get_transform_mask
from data.image_folder import make_dataset
from PIL import Image
import random
import util.util as util
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
from torchvision.datasets.folder import pil_loader
import numpy as np
from IPython import embed
from tqdm import tqdm


class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_A_seg = os.path.join(opt.dataroot, opt.phase + 'A_seg')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        if opt.phase == "test" and not os.path.exists(self.dir_A) \
           and os.path.exists(os.path.join(opt.dataroot, "valA")):
            self.dir_A = os.path.join(opt.dataroot, "valA")
            self.dir_B = os.path.join(opt.dataroot, "valB")

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.A_seg_paths = sorted(make_dataset(self.dir_A_seg, opt.max_dataset_size))   # load images from '/path/to/data/trainA_seg'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.A_seg_size = len(self.A_seg_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B


        self.mean=(0.5,0.5,0.5)
        self.std=(0.5,0.5,0.5)
        self.transform = A.Compose([
            A.Resize(286,286),
            A.RandomCrop(width=256, height=256),
            A.HorizontalFlip(p=0.5),
            # A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=self.mean, std=self.std),
            ToTensorV2()
        ])
        # self.transform = A.Compose([
        #     A.Resize(286,286),
        #     A.HorizontalFlip(p=0.5),
        #     A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
        #     A.PadIfNeeded(min_height=256, min_width=256, always_apply=True, border_mode=0),
        #     A.RandomCrop(width=256, height=256),
        #     A.Perspective(p=0.5),   
        #     A.Normalize(mean=self.mean, std=self.std),
        #     ToTensorV2()
        # ])


    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        A_seg_path = self.A_seg_paths[index % self.A_seg_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]


        # A_img = cv2.imread(A_path)
        # A_img = cv2.cvtColor(A_img, cv2.COLOR_BGR2RGB)
        A_img = np.asarray(pil_loader(A_path)) # PIL loads in RGB, the pixels are between 1 and 255, and after converting it to an array the shape is (h, w, 3)

        A_seg_img = cv2.imread(A_seg_path, cv2.IMREAD_GRAYSCALE)//255

        # B_img = cv2.imread(B_path)
        # B_img = cv2.cvtColor(B_img, cv2.COLOR_BGR2RGB)
        B_img = np.asarray(pil_loader(B_path))


        # Apply image transformation
        # For FastCUT mode, if in finetuning phase (learning rate is decaying),
        # do not perform resize-crop data augmentation of CycleGAN.
#        print('current_epoch', self.current_epoch)
        is_finetuning = self.opt.isTrain and self.current_epoch > self.opt.n_epochs
        modified_opt = util.copyconf(self.opt, load_size=self.opt.crop_size if is_finetuning else self.opt.load_size)
        

        A_transformed = self.transform(image=A_img, mask=A_seg_img)
        A = A_transformed['image']
        A_seg = A_transformed['mask'][None]

        B_transformed = self.transform(image=B_img)
        B = B_transformed['image']

        return {'A': A, 'A_seg': A_seg, 'B': B, 'A_paths': A_path, 'A_seg_paths': A_seg_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)       
        