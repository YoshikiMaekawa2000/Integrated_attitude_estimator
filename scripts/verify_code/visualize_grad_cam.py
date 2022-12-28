import os
import cv2
import PIL.Image as PILIMAGE
import math
import numpy as np
import time
import argparse
from numpy.core.fromnumeric import argmin
from torch.utils import data
import yaml
import csv
import random
import itertools
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import urllib
import pickle

import scipy.stats as stats

from sklearn.mixture import GaussianMixture
from collections import OrderedDict

import torch
from torchvision import models
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as nn_functional

#Grad CAM
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('../')
from models import vit
from models import senet

class GradCam:
    def __init__(self, CFG):
        self.CFG = CFG
        self.infer_dataset_top_directory = CFG['infer_dataset_top_directory']

        self.camera_image_directory = CFG['camera_image_directory']
        self.image_num = CFG['image_num']
        self.image_format = CFG['image_format']

        self.weights_top_directory = CFG['weights_top_directory']
        self.weights_file_name = CFG['weights_file_name']
        self.weights_path = os.path.join(self.weights_top_directory, self.weights_file_name)

        self.save_image_top_directory = CFG['save_image_top_directory']
        self.save_image_file_format = CFG['save_image_file_format']

        self.index_csv_path = CFG['index_csv_path']

        self.img_size = int(self.CFG['hyperparameters']['img_size'])
        
        self.do_domain_randomization =str(self.CFG['hyperparameters']["transform_params"]['do_domain_randomization'])
        self.resize = int(CFG["hyperparameters"]["transform_params"]["resize"])

        self.num_classes = int(self.CFG['hyperparameters']['num_classes'])
        self.num_frames = int(self.CFG['hyperparameters']['num_frames'])
        self.deg_threshold = int(self.CFG['hyperparameters']['deg_threshold'])
        self.mean_element = float(self.CFG['hyperparameters']['mean_element'])
        self.std_element = float(self.CFG['hyperparameters']['std_element'])
        self.network_type = self.CFG['hyperparameters']['network_type']

        # TimeSformer params
        self.patch_size = int(CFG["hyperparameters"]["timesformer"]["patch_size"])
        self.attention_type = str(CFG["hyperparameters"]["timesformer"]["attention_type"])
        self.depth = int(CFG["hyperparameters"]["timesformer"]["depth"])
        self.num_heads = int(CFG["hyperparameters"]["timesformer"]["num_heads"])

        # SENet params
        self.resnet_model = str(CFG["hyperparameters"]["senet"]["resnet_model"])

        self.image_cv = np.empty(0)
        self.depth_cv = np.empty(0)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Device: ",self.device)

        self.img_transform = self.getImageTransform(self.mean_element, self.std_element, self.resize)

        self.net = self.getNetwork()

        self.target_layer = self.net.feature_extractor
        self.target_layer_roll = self.net.fully_connected.roll_fc[-1]
        self.target_layer_pitch = self.net.fully_connected.pitch_fc[-1]

        self.gradcam_roll = GradCAM(model = self.net, target_layer = self.target_layer_roll, use_cuda = torch.cuda.is_available())
        self.gradcam_pitch = GradCAM(model = self.net, target_layer = self.target_layer_pitch, use_cuda = torch.cuda.is_available())


        self.value_dict = []

        with open(self.index_csv_path) as fd:
            reader = csv.reader(fd)
            for row in reader:
                num = float(row[0])
                self.value_dict.append(num)

    def getImageTransform(self, mean_element, std_element, resize):

        mean = mean_element
        std = std_element
        size = (resize, resize)

        img_transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize((mean,), (std,))
        ])

        return img_transform

    def getNetwork(self):
        print("Load Network")
        if self.network_type == "TimeSformer":
            net = vit.TimeSformer(self.img_size, self.patch_size, self.num_classes, self.num_frames, self.depth, self.num_heads, self.attention_type, self.weights_path, 'eval')
        elif self.network_type == "SENet":
            net = senet.SENet(model=self.resnet_model, dim_fc_out=self.num_classes, norm_layer=nn.BatchNorm2d, pretrained_model=self.weights_path, time_step=self.num_frames, use_SELayer=True)
        else:
            print("Error: Network type is not defined")
            quit()

        print(net)
        net.to(self.device)
        net.train()

        print("Load state_dict")
        if torch.cuda.is_available:
            state_dict = torch.load(self.weights_path, map_location=lambda storage, loc: storage)
            #print(state_dict.keys())
            new_state_dict = OrderedDict()

            for k, v in state_dict.items():
                if 'module' in k:
                    k = k.replace('module.', '')
                new_state_dict[k] = v

            state_dict = new_state_dict
            print("Load .pth file")
            # print(state_dict.keys())
            # print(state_dict['model.mlp_head_roll.1.weight'].size())
        else:
            state_dict = torch.load(self.weights_path, map_location={"cuda:0": "cpu"})
            print("Load to CPU")

        # net.load_state_dict(state_dict, strict=False)
        net.load_state_dict(state_dict)

        # net_roll = GradCAMNetwork(net.feature_extractor, net.fully_connected.roll_fc)
        # net_pitch = GradCAMNetwork(net.feature_extractor, net.fully_connected.pitch_fc)

        # net_roll.to(self.device)
        # net_pitch.to(self.device)

        #return net, net_roll, net_pitch
        return net

    def spin(self):
        print("aaa")


if __name__ == '__main__':

    parser = argparse.ArgumentParser("./visualize_grad_cam.py")
    parser.add_argument(
        '--config', '-c',
        type=str,
        required=False,
        default='./visualize_grad_cam_config.yaml',
        help='Grad Cam Config'
    )

    FLAGS, unparsed = parser.parse_known_args()

    #Load yaml file
    try:
        print("Opening grad cam config file %s", FLAGS.config)
        CFG = yaml.safe_load(open(FLAGS.config, 'r'))
    except Exception as e:
        print(e)
        print("Error opening grad cam config file %s", FLAGS.config)
        quit()
    
    grad_cam = GradCam(CFG)
    grad_cam.spin()