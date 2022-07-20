
import torch
from torch import nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
import os
from time import localtime, strftime
import yaml
import argparse
import time
import copy

from model import *
from data import *

parser = argparse.ArgumentParser(description='CS7643 Final Project GAN')
parser.add_argument('--config', default='config.yaml')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, args):
    epoch += 1
    if epoch <= args.warmup:
        lr = args.learning_rate * epoch / args.warmup
    elif epoch > args.steps[1]:
        lr = args.learning_rate * 0.01
    elif epoch > args.steps[0]:
        lr = args.learning_rate * 0.1
    else:
        lr = args.learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



def train(device, epoch, data_loader, Generater, Discriminator, Doptimizer, Goptimizer, criterion1, criterion2):
    """
    criterion1 = BCELoss
    criterion2 = MSELoss
    """
    global coeff_adv, coeff_pw

    iter_time = AverageMeter()
    Dis_losses = AverageMeter()
    Gen_losses = AverageMeter()

    for idx, data in enumerate(data_loader):
        start = time.time()
        sampled, target, mean, std = data[0], data[1], data[2], data[3]
        # if torch.cuda.is_available():
        sampled = sampled.to(device)
        target = target.to(device)

        fake_img = Generater(sampled).to(device)

        # train discrimiator
        d_real_logit = Discriminator(target)
        real_loss = criterion1(d_real_logit, torch.ones(d_real_logit.shape).to(device))  # label 1 as real
        d_fake_logit = Discriminator(fake_img)
        fake_loss = criterion1(d_fake_logit, torch.zeros(d_fake_logit.shape).to(device)) # label 0 as fake

        Disloss = real_loss + fake_loss
        Doptimizer.zero_grad()
        Disloss.backward()
        Doptimizer.step()

        # train generator
        g_fake_logit = Discriminator(fake_img)
        gen_adv_loss = criterion1(g_fake_logit, torch.ones(g_fake_logit.shape).to(device))
        gen_pw_loss = criterion2(fake_img, target)  # pixel wise loss
        Genloss = gen_adv_loss * coeff_adv + gen_pw_loss * coeff_pw
        Goptimizer.zero_grad()
        Genloss.backward()
        Goptimizer.step()

        Dis_losses.update(Disloss, sampled.shape[0])
        Gen_losses.update(Genloss, sampled.shape[0])

        iter_time.update(time.time() - start)
        if idx % 10 == 0:
            print(('Epoch: [{0}][{1}/{2}]\t'
                   'Time {iter_time.val:.3f} ({iter_time.avg:.3f})\t'
                   'Discriminator Loss {Dis_losses.val:.4f} ({Dis_losses.avg:.4f})\t'
                   'Generator Loss {Gen_losses.val:.4f} ({Gen_losses.avg:.4f})\t')
                  .format(epoch, idx, len(data_loader), iter_time=iter_time, Dis_losses=Dis_losses, Gen_losses=Gen_losses))

    

# bce_loss = nn.BCELoss(reduction='mean').to(device)
# mcs_loss = nn.MSELoss(reduction='mean').to(device)
# check_point_dir = os.path.join(f'check_point/checkpoint.{localtime}')
# if not os.path.exists(check_point_dir):
#     os.makedirs(check_point_dir)

def main():
    global args
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = mri_data.SliceDataset(
        root=pathlib.Path('./dataset/singlecoil_train/'),
        transform=UnetDataTransform(which_challenge="singlecoil"),
        challenge='singlecoil'
    )
    val_dataset = mri_data.SliceDataset(
        root=pathlib.Path('./dataset/singlecoil_val/'),
        transform=UnetDataTransform(which_challenge="singlecoil"),
        challenge='singlecoil'
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

    Generater_model = Generater().to(device)
    Discriminator_model = Discriminator().to(device)
    


    print(args.batch_size)

if __name__ == '__main__':
    main()