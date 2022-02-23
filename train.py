import time
import os
import torch
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from util.util import tensor2im
from torchvision.utils import save_image

from torchvision import transforms
from data.val_dataset import ValDataset
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from IPython import embed
import numpy as np
from PIL import Image

def save_not_normalized_image(image, path, mean, std):
    no_norm_image = image.detach().cpu()*np.asarray(std)[:,None, None] + np.asarray(mean)[:, None, None]
    save_image(no_norm_image, path)

def save_image_custom(image_numpy, image_path):
    # this is directly adapted from the function "save_image" in utils/utils.py
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)    


if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.

    val_data = ValDataset(opt.dataroot)
    # opt.batch_size
    val_dataloader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=opt.num_threads)
    # dice_loss=smp.losses.DiceLoss(mode='binary', log_loss=True, ignore_index=-1)
    iou = smp.utils.metrics.IoU()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # image2tensor = transforms.PILToTensor()

    model = create_model(opt)      # create a model given opt.model and other options
    print('The number of training images = %d' % dataset_size)

    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    opt.visualizer = visualizer
    total_iters = 0                # the total number of training iterations

    optimize_time = 0.1

    best_iou = 0.0

    times = []
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch

        dataset.set_epoch(epoch)
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            batch_size = data["A"].size(0)
            total_iters += batch_size
            epoch_iter += batch_size
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            optimize_start_time = time.time()
            if epoch == opt.epoch_count and i == 0:
                model.data_dependent_initialize(data)
                model.setup(opt)               # regular setup: load and print networks; create schedulers
                model.parallelize()
            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            optimize_time = (time.time() - optimize_start_time) / batch_size * 0.005 + 0.995 * optimize_time

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                visualizer.print_current_losses(epoch, epoch_iter, losses, optimize_time, t_data)

            iter_data_time = time.time()

        model.compute_visuals()
        visuals = model.get_current_visuals()
        os.makedirs('/cut/checkpoints/{}/train/epoch_{}'.format(opt.name, epoch), exist_ok=True)
        for image_type in visuals.keys():
            for i, img in enumerate(visuals[image_type]):
                img_path = '/cut/checkpoints/{}/train/epoch_{}/{}_{}.png'.format(opt.name, epoch, image_type, i)
                if img.shape[0]==3:
                    if image_type == "fake_B":
                        save_image_custom(tensor2im(img[None]), img_path)
                    else:
                        save_not_normalized_image(img, img_path, dataset.dataset.mean, dataset.dataset.std)
                    # save_image(img, img_path)
                else:
                    save_image(img.repeat(3,1,1).float(), img_path)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.

    
        # Val dataset
        os.makedirs('/cut/checkpoints/{}/val/epoch_{}'.format(opt.name, epoch), exist_ok=True)
        
        model.netS.eval()
        total = 0
        total_iou = 0
        with torch.no_grad():

            for image, mask in val_dataloader:
                image = image.to(device)
                mask = mask.to(device)
                pred = model.netS(image)
                l = iou(pred,mask).item()
                total_iou+= l
                total+=1

                if total < 10:
                    pred_path = '/cut/checkpoints/{}/val/epoch_{}/pred_{}.png'.format(opt.name, epoch, total)
                    save_image(pred[0], pred_path)

                    image_path = '/cut/checkpoints/{}/val/epoch_{}/image_{}.png'.format(opt.name, epoch, total)
                    # save_image(image[0], image_path)
                    # save_not_normalized_image(image[0], image_path, val_dataloader.dataset.mean, val_dataloader.dataset.std)
                    save_image_custom(tensor2im(image[0][None]), image_path)
            current_iou = total_iou/total
            if current_iou >= best_iou:
                print('Overwrite best model')
                torch.save(model.netS, os.path.join('/cut/checkpoints/', opt.name, '_best.pth'))
                best_iou = current_iou
            print('Mean IOU:', current_iou)
        model.netS.train()


# TODO
# fix channels albumentations
# apply to validation
# check if red images

# save GAN image in val

# data augemantation in synth or fake_real?

# print the train IoU (even if not used as a loss)