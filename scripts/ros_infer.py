#!/usr/bin/env python3

import sys, codecs
from tkinter import W
#sys.stdout = codecs.getwriter("utf-8")(sys.stdout)
sys.dont_write_bytecode = True

from random import shuffle
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import datetime
import numpy as np
import random
import cv2

import argparse
import subprocess
import datetime
import yaml
from shutil import copyfile
import os
import shutil
import __init__ as booger
import csv

import torch
from torchvision import models
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as nn_functional
import torch.optim as optim
import torch.backends.cudnn as cudnn

from collections import OrderedDict

from tensorboardX import SummaryWriter

import rospy
from sensor_msgs.msg import Image as ImageMsg
from geometry_msgs.msg import Vector3Stamped
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32
from cv_bridge import CvBridge, CvBridgeError

#Need in running in ROS
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('../')

from models import vit
from models import senet
from common import dataset_mod_Gimbal
from common import dataset_mod_AirSim
from common import make_datalist_mod
from common import data_transform_mod

from integrated_attitude_estimator.msg import *

class IntegratedAttitudeEstimator:
    def __init__(self, FLAGS, CFG):
        self.FLAGS = FLAGS
        self.CFG = CFG

        self.weights_top_directory = self.CFG["weights_top_directory"]
        self.weights_file_name = self.CFG["weights_file_name"]
        self.weights_path = os.path.join(self.weights_top_directory, self.weights_file_name)

        self.save_in_csv = bool(self.CFG["save_in_csv"])
        self.infer_log_top_directory = self.CFG["infer_log_top_directory"]
        self.infer_log_file_name = self.CFG["infer_log_file_name"]
        self.infer_log_file_path = os.path.join(self.infer_log_top_directory, self.infer_log_file_name)

        self.yaml_name = self.CFG["yaml_name"]
        self.yaml_path = os.path.join(self.infer_log_top_directory, self.yaml_name)
        shutil.copy(FLAGS.config, self.yaml_path)

        self.index_dict_path = self.CFG["index_csv_path"]

        self.image_topic_name = self.CFG["ros_params"]["image_topic_name"]
        self.gt_angle_topic_name = self.CFG["ros_params"]["gt_angle_topic_name"]

        self.network_type = str(CFG["hyperparameters"]["network_type"])
        self.img_size = int(self.CFG['hyperparameters']['img_size'])
        self.resize = int(CFG["hyperparameters"]["transform_params"]["resize"])
        self.num_classes = int(self.CFG['hyperparameters']['num_classes'])
        self.num_frames = int(self.CFG['hyperparameters']['num_frames'])
        self.deg_threshold = int(self.CFG['hyperparameters']['deg_threshold'])
        self.mean_element = float(self.CFG['hyperparameters']['mean_element'])
        self.std_element = float(self.CFG['hyperparameters']['std_element'])

        # TimeSformer params
        self.patch_size = int(CFG["hyperparameters"]["timesformer"]["patch_size"])
        self.attention_type = str(CFG["hyperparameters"]["timesformer"]["attention_type"])
        self.depth = int(CFG["hyperparameters"]["timesformer"]["depth"])
        self.num_heads = int(CFG["hyperparameters"]["timesformer"]["num_heads"])

        # SENet params
        self.resnet_model = str(CFG["hyperparameters"]["senet"]["resnet_model"])

        self.value_dict = []
        self.value_dict.append([-1*int(self.deg_threshold)-1, 0])
        with open(self.index_dict_path) as fd:
            reader = csv.reader(fd)
            for row in reader:
                #num = float(row[0])
                tmp_row = [int(row[0]), int(row[1])+1]
                self.value_dict.append(tmp_row)
        self.value_dict.append([int(self.deg_threshold)+1, int(self.num_classes)-1])

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Device: ", self.device)

        # Network params
        self.load_net = False
        self.net = self.getNetwork()

        # Transform Params
        self.img_transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize((self.mean_element,), (self.std_element,))
        ])

        # ROS Image Subscriber
        self.sub_image = rospy.Subscriber(self.image_topic_name, ImageMsg, self.image_callback, queue_size=1)
        self.image_queue = []
        self.image_count = 0
        self.bridge = CvBridge()

        # ROS GT Angle Subscriber
        self.sub_gt_angle = rospy.Subscriber(self.gt_angle_topic_name, EularAngle, self.gt_angle_callback, queue_size=1)
        self.gt_angle = EularAngle()

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
        net.eval()

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
        self.load_net = True
        return net

    def gt_angle_callback(self, msg):
        self.gt_angle = msg

    def image_callback(self, msg):
        try:
            self.color_img_cv = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            if self.image_count < self.num_frames:
                self.image_queue.append(self.color_img_cv)
                self.image_count += 1
            elif self.image_count == self.num_frames:
                self.image_queue.pop(0)
                self.image_queue.append(self.color_img_cv)
                if self.load_net == True:
                    self.network_prediction()
            else:
                print("Error: image_count is out of range")
                quit()
        except CvBridgeError as e:
            print(e)

    def network_prediction(self):
        print("Foo")


if __name__ == '__main__':
    rospy.init_node('ros_infer', anonymous=True)
    parser = argparse.ArgumentParser("./ros_infer.py")

    parser.add_argument(
        "--config", "-c",
        type=str,
        default="../pyyaml/ros_infer_config.yaml",
        required=False,
        help="Infer config file path"
    )

    FLAGS, unparsed = parser.parse_known_args()

    try:
        print("Opening Infer config file %s", FLAGS.config)
        CFG = yaml.safe_load(open(FLAGS.config, 'r'))
    except Exception as e:
        print(e)
        print("Failed to open config file %s", FLAGS.config)
        quit()

    integrated_attitude_estimator = IntegratedAttitudeEstimator(FLAGS, CFG)

    rospy.spin()