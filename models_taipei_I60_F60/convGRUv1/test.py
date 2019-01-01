## import some useful tools
import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import datetime as dt

## import torch module
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, utils

# import our model and dataloader
import sys
sys.path.append("..")
from tools.args_tools import createfolder
from tools.dataset_GRU import ToTensor, Normalize, TyDataset
from tools.loss_function import BMAE, BMSE
from CNNGRU import *


data = torch.randn(4,5,1,72,72).to("cuda")

Net = model(n_encoders=5, n_decoders=20,
        encoder_input=1, encoder_hidden=[2,3,4], encoder_kernel=[3,3,3], encoder_n_layers=3,
        decoder_input=0, decoder_hidden=[4,3,2], decoder_output=1, decoder_kernel=[3,3,3], decoder_n_layers=3,
        padding=True, batch_norm=False).to("cuda")

# print(len(c(data)))
# for i in c(data):
#     print(i.size())

input_frames = 5
output_frames = 20
# Normalize data
mean = [12.834] * input_frames + [3.014] * output_frames
std = [14.14] * input_frames + [6.773] * output_frames
transfrom = transforms.Compose([ToTensor(),Normalize(mean=mean, std=std)])
traindataset = TyDataset(ty_list_file="../../ty_list.xlsx",
                    input_frames=input_frames,
                    output_frames=output_frames,
                    train=True,
                    root_dir="../../01_TY_database/02_wrangled_data_Taipei/",
                    transform = transfrom)
testdataset = TyDataset(ty_list_file="../../ty_list.xlsx",
                    input_frames=input_frames,
                    output_frames=output_frames,
                    train=False,
                    root_dir="../../01_TY_database/02_wrangled_data_Taipei/",
                    transform = transfrom)

# set train and test dataloader
params = {"batch_size":4, "shuffle":True, "num_workers":1}
trainloader = DataLoader(traindataset, **params)
testloader = DataLoader(testdataset, **params)
# for idx, data in enumerate(testloader):
#     print(idx)
#     # rad = data["RAD"].to("cuda",dtype=torch.float)
#     # label = data["QPE"].to("cuda",dtype=torch.float)
#     # label = label.view(label.size(0),-1)
#     # results = Net(rad)
#     # print(idx, results.size())
print(len(trainloader.dataset))
print(len(testloader.dataset))
