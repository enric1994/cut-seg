import torch
import os
from data.test_dataset import TestDataset
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

experiment_name = 'synth_polyp_V11.11'

base_path = "/polyp-data/TestDataset"
dataset_names = ["CVC-300", "CVC-ClinicDB", "CVC-ColonDB", "ETIS-LaribPolypDB", "Kvasir"]


iou = smp.utils.metrics.IoU()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_path = '/cut/checkpoints/{}/_best.pth'.format(experiment_name)
model = torch.load(model_path)
model = model.to(device)
model.eval()

for dataset_name in dataset_names:
    target_dataset = os.path.join(base_path, dataset_name)
    test_data = TestDataset(target_dataset)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=8)


    
    total = 0
    total_iou = 0
    with torch.no_grad():

        for image, mask in test_dataloader:
            image = image.to(device)
            mask = mask.to(device)
            pred = model(image)
            l = iou(pred,mask).item()
            total_iou+= l
            total+=1

        print(dataset_name, ' - Mean IOU:', total_iou/total)
