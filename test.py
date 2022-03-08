import torch
import os
from data.test_dataset import TestDataset
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

experiment_name = 'cut_all_200_reversed.7'

base_path = "/polyp-data/TestDataset"
dataset_names = ["CVC-300", "CVC-ClinicDB", "CVC-ColonDB", "ETIS-LaribPolypDB", "Kvasir"]


iou = smp.utils.metrics.IoU()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load segmentation model
model_path = '/cut/checkpoints/{}/S_best.pth'.format(experiment_name)
model = torch.load(model_path)
model = model.to(device)
model.eval()

if 'reversed' in experiment_name:
    # Load generator
    # TODO create G net
    # self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
    generator_path = '/cut/checkpoints/{}/G_best.pth'.format(experiment_name)
    generator = torch.load(generator_path)
    generator = generator.to(device)
    generator.eval()

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
            if 'reversed' in experiment_name:
                image = generator(image)
            pred = model(image)
            l = iou(pred,mask).item()
            total_iou+= l
            total+=1

        print(dataset_name, ' - Mean IOU:', total_iou/total)
