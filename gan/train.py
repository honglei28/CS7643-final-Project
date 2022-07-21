
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



def train(device, epoch, data_loader, Generater, Discriminator, Goptimizer, Doptimizer, criterion1, criterion2):
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
        sampled = sampled.unsqueeze(0).to(device)
        target = target.unsqueeze(0).to(device)

        fake_img = Generater(sampled).to(device)

        # train discrimiator
        d_real_logit = Discriminator(target)
        real_loss = criterion1(d_real_logit, torch.ones(d_real_logit.shape).to(device))  # label 1 as real
        d_fake_logit = Discriminator(fake_img)
        fake_loss = criterion1(d_fake_logit, torch.zeros(d_fake_logit.shape).to(device)) # label 0 as fake

        Disloss = real_loss + fake_loss
        Doptimizer.zero_grad()
        Disloss.backward(retain_graph=True)
        Doptimizer.step()

        # train generator
        g_fake_logit = Discriminator(fake_img)
        gen_adv_loss = criterion1(g_fake_logit, torch.ones(g_fake_logit.shape).to(device))
        gen_pw_loss = criterion2(fake_img, target)  # pixel wise loss
        Genloss = gen_adv_loss * coeff_adv + gen_pw_loss * coeff_pw
        Goptimizer.zero_grad()
        Genloss.backward(retain_graph=True)
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

def validate(device, epoch, data_loader, Generator, Discriminator, criterion1, criterion2):
    """
    criterion1 = BCELoss
    criterion2 = MSELoss
    """
    global coeff_adv, coeff_pw

    iter_time = AverageMeter()
    Dis_losses = AverageMeter()
    Gen_losses = AverageMeter()

    Generator.eval()
    Discriminator.eval()

    for idx, data in enumerate(data_loader):
        start = time.time()
        sampled, target, mean, std = data[0], data[1], data[2], data[3]
        # if torch.cuda.is_available():
        sampled = sampled.unsqueeze(0).to(device)
        target = target.unsqueeze(0).to(device)
        with torch.no_grad():
            fake_img = Generater(sampled).to(device)
            d_real_logit = Discriminator(target)
            real_loss = criterion1(d_real_logit, torch.ones(d_real_logit.shape).to(device))
            d_fake_logit = Discriminator(fake_img)
            fake_loss = criterion1(d_fake_logit, torch.zeros(d_fake_logit.shape).to(device))
            Disloss = real_loss + fake_loss

            g_fake_logit = Discriminator(fake_img)
            gen_adv_loss = criterion1(g_fake_logit, torch.ones(g_fake_logit.shape).to(device))
            gen_pw_loss = criterion2(fake_img, target)  # pixel wise loss
            Genloss = gen_adv_loss * coeff_adv + gen_pw_loss * coeff_pw

            iter_time.update(time.time() - start)
            if idx % 10 == 0:
                print(('Epoch: [{0}][{1}/{2}]\t'
                    'Time {iter_time.val:.3f} ({iter_time.avg:.3f})\t'
                    'Discriminator Loss {Dis_losses.val:.4f} ({Dis_losses.avg:.4f})\t'
                    'Generator Loss {Gen_losses.val:.4f} ({Gen_losses.avg:.4f})\t')
                    .format(epoch, idx, len(data_loader), iter_time=iter_time, Dis_losses=Dis_losses, Gen_losses=Gen_losses))
            
        Dis_losses.update(Disloss, sampled.shape[0])
        Gen_losses.update(Genloss, sampled.shape[0])

    return Dis_losses.avg, Gen_losses.avg



def main():
    global args, coeff_adv, coeff_pw
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    coeff_adv, coeff_pw = args.coeff_adv, args.coeff_pw

    train_dataset = mri_data.SliceDataset(
        root=pathlib.Path('./data/singlecoil_train/'),
        transform=UnetDataTransform(which_challenge="singlecoil"),
        challenge='singlecoil'
    )
    val_dataset = mri_data.SliceDataset(
        root=pathlib.Path('./data/singlecoil_val/'),
        transform=UnetDataTransform(which_challenge="singlecoil"),
        challenge='singlecoil'
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

    Generater_model = Generater().to(device)
    Discriminator_model = Discriminator().to(device)

    Doptimizer = torch.optim.SGD(Discriminator_model.parameters(), args.d_learning_rate,
                                 momentum=args.d_momentum,
                                 weight_decay=args.d_reg)
    Goptimizer = torch.optim.SGD(Generater_model.parameters(), args.g_learning_rate,
                                 momentum=args.g_momentum,
                                 weight_decay=args.g_reg)

    criterion_bce = nn.BCELoss()
    criterion_mse = nn.MSELoss()

    best_dis_loss = 0.0
    best_gen_loss = 0.0
    best_dis_model = None
    best_gen_model = None
    for epoch in range(args.epochs):
        # adjust_learning_rate(Doptimizer, epoch, args)
        # adjust_learning_rate(Goptimizer, epoch, args)

        # train loop
        train(device, epoch, train_loader, Generater_model, Discriminator_model, Goptimizer, Doptimizer, criterion_bce, criterion_mse)

        # validation loop
        dis_loss, gen_loss = validate(device, epoch, val_loader, Generater_model, Discriminator_model, criterion_bce, criterion_mse)

        if dis_loss < best_dis_loss:
            best_dis_loss = best_dis_loss
            best_dis_model = copy.deepcopy(Discriminator_model)

        if gen_loss < best_gen_loss:
            best_gen_loss = gen_loss
            best_gen_model = copy.deepcopy(Generater_model)

    if args.save_best:
        torch.save(best_gen_model.state_dict(), './checkpoints/' + 'generator' + '.pth')
        torch.save(best_dis_model.state_dict(), './checkpoints/' + 'discriminator' + '.pth')

if __name__ == '__main__':
    main()