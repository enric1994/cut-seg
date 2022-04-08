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

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 

import random
import string
import torch


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
        self.B_paths = self.B_paths[:opt.real_size]
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.A_seg_size = len(self.A_seg_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

        self.mean=(0.5,0.5,0.5)
        self.std=(0.5,0.5,0.5)
        self.transform = A.Compose([
            A.Resize(opt.crop_size + 64 ,opt.crop_size + 64),
            A.RandomCrop(width=opt.crop_size, height=opt.crop_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(always_apply=False, p=1.0, limit=(-360, 360), interpolation=0, border_mode=0, value=(0, 0, 0), mask_value=None),
            # A.CoarseDropout(always_apply=False, p=1.0, max_holes=3, max_height=59, max_width=60, min_holes=1, min_height=49, min_width=47),
            A.Normalize(mean=self.mean, std=self.std),
            ToTensorV2()
        ])

        if opt.DAtrainB:
            self.transform_B = self.transform
        else:
            self.transform_B = A.Compose([
                A.Resize(opt.crop_size + 64 ,opt.crop_size + 64),
                A.CenterCrop(width=opt.crop_size, height=opt.crop_size),
                A.Normalize(mean=self.mean, std=self.std),
                ToTensorV2()
            ])
        
        if opt.syncPL:
            self.transform_pl = A.Compose([
                A.Resize(opt.crop_size + 64 ,opt.crop_size + 64),
                A.Normalize(mean=self.mean, std=self.std),
                ToTensorV2()
            ])
            self.pl_real_B = torch.zeros(size = (self.B_size, 1, opt.crop_size + 64, opt.crop_size + 64))


        self.serial_batches_pl = self.opt.serial_batches

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
        
        if self.opt.serial_batches or self.serial_batches_pl:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]


        A_PIL = pil_loader(A_path) # PIL loads in RGB, the pixels are between 1 and 255, and after converting it to an array the shape is (h, w, 3)

        A_img = np.asarray(A_PIL)

        A_seg_img = cv2.imread(A_seg_path, cv2.IMREAD_GRAYSCALE)//255

        B_PIL = pil_loader(B_path)

        B_img = np.asarray(B_PIL)


        is_finetuning = self.opt.isTrain and self.current_epoch > self.opt.n_epochs
        modified_opt = util.copyconf(self.opt, load_size=self.opt.crop_size if is_finetuning else self.opt.load_size)
        

        A_transformed = self.transform(image=A_img, mask=A_seg_img)
        A = A_transformed['image']
        A_seg = A_transformed['mask'][None]

        if self.opt.syncPL:
            B_seg_pl = np.asarray(self.pl_real_B[index_B])[0]
            B_transformed = self.transform_B(image=B_img, mask=B_seg_pl)
            B = B_transformed['image']
            B_pl =  B_transformed['mask'][None]
            return {'A': A, 'A_seg': A_seg, 'B': B, 'A_paths': A_path, 'A_seg_paths': A_seg_path, \
                    'B_paths': B_path, 'B_pl':B_pl, 'indexes': index_B}
        else:
            B_transformed = self.transform_B(image=B_img)
            B = B_transformed['image']

        return {'A': A, 'A_seg': A_seg, 'B': B, 'A_paths': A_path, 'A_seg_paths': A_seg_path, 'B_paths': B_path, 'indexes': index_B}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)       
    
    def add_random_text(self, image):
        draw = ImageDraw.Draw(image)
        random_font_name = random.choice([x for x in os.listdir('/cut/fonts/') if '.ttf' in x])
        random_color = random.choice([(0,0,0),(255,255,255)])

        for i in range(random.randint(0,4)):
            random_font_size = random.randint(5, 34)
            random_font = ImageFont.truetype(os.path.join('/cut', 'fonts', random_font_name), random_font_size)
            x_pos = random.randint(0, 256)
            y_pos = random.randint(0, 256)
            random_length = random.randint(0,16)
            random_text = ''.join(random.choices(string.ascii_letters + string.digits, k=random_length))
            draw.text((x_pos, y_pos),random_text,random_color,font=random_font)
        
        return image