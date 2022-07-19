from model import *
import torch
import torch.optim as optim
import torchvision
import os
from time import localtime, strftime
import yaml
import argparse
import time
import copy



def train(device):
    
    curr_time = strftime("%Y_%m_%d_%H_%M_%S", localtime())
    
    check_point_dir = os.path.join(f'check_point/checkpoint.{localtime}')
    if not os.path.exists(check_point_dir):
        os.makedirs(check_point_dir)

    
    
    
    
    pass
