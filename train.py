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
from tqdm import tqdm

import wandb

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

    wandb.init(
    config=opt,
    # tags=[opt.CUT_mode, opt.dataroot, "all", "reversed"],
    project="cut-seg"
    )

    wandb.run.name = opt.name

    # wandb.log(dict(vars(opt)))
    opt = wandb.config

    val_data = ValDataset(opt.dataroot)
    # opt.batch_size
    val_dataloader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=opt.num_threads)
    # dice_loss=smp.losses.DiceLoss(mode='binary', log_loss=True, ignore_index=-1)
    iou = smp.utils.metrics.IoU()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # image2tensor = transforms.PILToTensor()

    model = create_model(opt)      # create a model given opt.model and other options
    # print('The number of training images = %d' % dataset_size)

    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    opt.visualizer = visualizer
    total_iters = 0                # the total number of training iterations

    optimize_time = 0.1

    best_iou = 0.0
    # val_log_path = os.path.join(opt.checkpoints_dir, opt.name, 'val_log.txt')
    # with open(val_log_path, "a") as log_file:
    #     now = time.strftime("%c")
    #     log_file.write('================ Validation metrics (%s) ================\n' % now)    

    times = []
    for epoch in tqdm(range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1)):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch

        dataset.set_epoch(epoch)
        model.current_epoch = epoch
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
                wandb.log({"train_loss_G_GAN": losses['G_GAN'], "epoch": epoch})
                wandb.log({"train_loss_D_real": losses['D_real'], "epoch": epoch})
                wandb.log({"train_loss_D_fake": losses['D_fake'], "epoch": epoch})
                wandb.log({"train_loss_G": losses['G'], "epoch": epoch})
                wandb.log({"train_loss_NCE": losses['NCE'], "epoch": epoch})
                wandb.log({"train_loss_S": losses['S'], "epoch": epoch})
                wandb.log({"train_loss_NCE_Y": losses['NCE_Y'], "epoch": epoch})
                
            iter_data_time = time.time()
        
        model.save_networks(epoch)
        model.load_networks(epoch)
        model.compute_visuals()
        visuals = model.get_current_visuals()
        os.makedirs('/cut/checkpoints/{}/train/epoch_{}'.format(opt.name, epoch), exist_ok=True)
        wandb_images = {}
        for image_type in visuals.keys():
            wandb_images[image_type] = []
            for i, img in enumerate(visuals[image_type]):
                img_path = '/cut/checkpoints/{}/train/epoch_{}/{}_{}.png'.format(opt.name, epoch, image_type, i)
                if img.shape[0]==3:
                    if image_type == "fake_B":
                        save_image_custom(tensor2im(img[None]), img_path)
                        wandb_images[image_type].append(wandb.Image(Image.fromarray(tensor2im(img[None]))))
                    else:
                        save_not_normalized_image(img, img_path, dataset.dataset.mean, dataset.dataset.std)

                        no_norm_image = img.detach().cpu()*np.asarray(dataset.dataset.std)[:,None, None] + np.asarray(dataset.dataset.mean)[:, None, None]
                        wandb_images[image_type].append(wandb.Image(no_norm_image))
                    # save_image(img, img_path)
                else:
                    save_image(img.repeat(3,1,1).float(), img_path)
                    wandb_images[image_type].append(wandb.Image(img.repeat(3,1,1).float()))
        
            wandb.log({"{}_train_{}".format(str(epoch).zfill(5), image_type): wandb_images[image_type]})
        
        # print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        lr = model.update_learning_rate()                     # update learning rates at the end of every epoch.
        wandb.log({"learning_rate": lr, "epoch": epoch})
    
        # Val dataset
        os.makedirs('/cut/checkpoints/{}/val/epoch_{}'.format(opt.name, epoch), exist_ok=True)
        
        model.netS.eval()
        model.netG.eval()
        total = 0
        total_iou = 0
        wandb_images_pred = []
        wandb_images_fake = []
        wandb_images_input = []
        wandb_images_synth = []
        with torch.no_grad():

            for synth, synth_mask, image, mask in val_dataloader:
                image = image.to(device)
                mask = mask.to(device)
                synth = synth.to(device)
                synth_mask = synth_mask.to(device)
                if opt.reversed:
                    fake = model.netG(synth)
                    pred = model.netS(fake)
                    l = iou(pred,synth_mask).item()
                else:
                    fake = model.netG(synth)
                    pred = model.netS(image)
                    l = iou(pred,mask).item()
                # wandb.log({"val_IOU": l})
                total_iou+= l
                total+=1

                if total <= 4:
                    pred_path = '/cut/checkpoints/{}/val/epoch_{}/pred_{}.png'.format(opt.name, epoch, total)
                    save_image(pred[0], pred_path)
                    wandb_images_pred.append(wandb.Image(pred[0].repeat(3,1,1).float()))

                    fake_path = '/cut/checkpoints/{}/val/epoch_{}/fake_{}.png'.format(opt.name, epoch, total)
                    save_image_custom(tensor2im(fake[0][None]), fake_path)
                    wandb_images_fake.append(wandb.Image(Image.fromarray(tensor2im(fake[0][None]))))

                    image_path = '/cut/checkpoints/{}/val/epoch_{}/image_{}.png'.format(opt.name, epoch, total)
                    # save_image(image[0], image_path)
                    # save_not_normalized_image(image[0], image_path, val_dataloader.dataset.mean, val_dataloader.dataset.std)
                    save_image_custom(tensor2im(image[0][None]), image_path)
                    wandb_images_input.append(wandb.Image(Image.fromarray(tensor2im(image[0][None]))))

                    synth_path = '/cut/checkpoints/{}/val/epoch_{}/synth_{}.png'.format(opt.name, epoch, total)
                    # save_image(image[0], image_path)
                    # save_not_normalized_image(image[0], image_path, val_dataloader.dataset.mean, val_dataloader.dataset.std)
                    save_image_custom(tensor2im(synth[0][None]), synth_path)
                    wandb_images_synth.append(wandb.Image(Image.fromarray(tensor2im(synth[0][None]))))
            
            wandb.log({"{}_val_pred".format(str(epoch).zfill(5)): wandb_images_pred})
            wandb.log({"{}_val_fake".format(str(epoch).zfill(5)): wandb_images_fake})
            wandb.log({"{}_val_input".format(str(epoch).zfill(5)): wandb_images_input})
            wandb.log({"{}_val_synth".format(str(epoch).zfill(5)): wandb_images_synth})

            current_iou = total_iou/total
            wandb.log({"val_mIOU": current_iou, "epoch": epoch})
            if current_iou >= best_iou:
                # print('Overwrite best model')
                torch.save(model.netS, os.path.join('/cut/checkpoints/', opt.name, 'S_best.pth'))
                wandb.save(os.path.join('/cut/checkpoints/', opt.name, 'S_best.pth'), base_path='/cut')

                if opt.reversed:
                    torch.save(model.netG.state_dict(), os.path.join('/cut/checkpoints/', opt.name, 'G_best.pth'))
                    wandb.save(os.path.join('/cut/checkpoints/', opt.name, 'G_best.pth'), base_path='/cut')

                best_iou = current_iou
                wandb.log({"best_val_mIOU": best_iou, "epoch": epoch})
            # print('Mean IOU:', current_iou)
            # with open(val_log_path, "a") as log_file:
            #     log_file.write('Epoch {}, mean IOU: {}\n'.format(epoch, current_iou))  # save the message
        model.netS.train()
        model.netG.train()


# TODO
# fix channels albumentations
# apply to validation
# check if red images

# save GAN image in val

# data augemantation in synth or fake_real?

# print the train IoU (even if not used as a loss)