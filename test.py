import torch
import os
from data.test_dataset import TestDataset
from torch.utils.data import DataLoader
from util.util import dice_coef
import segmentation_models_pytorch as smp

experiment_name = 'cvc300.test'

base_path = "/polyp-data/TestDataset"
dataset_names = ["CVC-300", "CVC-ClinicDB", "CVC-ColonDB", "ETIS-LaribPolypDB", "Kvasir"]


iou = smp.utils.metrics.IoU()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load segmentation model
model_path = '/cut/checkpoints/{}/S_best.pth'.format(experiment_name)
model = torch.load(model_path)
model = model.to(device)
model.eval()

for dataset_name in dataset_names:
    target_dataset = os.path.join(base_path, dataset_name)
    test_data = TestDataset(target_dataset)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=8)
    
    total = 0
    total_iou = 0
    total_dice = 0
    with torch.no_grad():

        for image, mask in test_dataloader:
            image = image.to(device)
            mask = mask.to(device)
            if 'reversed' in experiment_name:
                image = generator(image)
            pred = model(image)
            dice = dice_coef(mask.cpu(),pred.cpu())
            l = iou(pred,mask).item()
            total_iou+= l
            total_dice+=dice
            total+=1

        print(dataset_name, ' - Mean DICE:', total_dice/total, ' - Mean IOU:', total_iou/total)
