## import some useful tools
import os
import numpy as np
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
from tools.args_tools import args, createfolder
from tools.dataset_CNN_v2 import ToTensor, Normalize, TyDataset
from tools.loss_function import BMAE, BMSE
from CNN2D import *

def train(net, trainloader, testloader, results_file, max_epochs=50, loss_function=BMSE,
            optimizer=optim.Adam, device=args.device):
    # create new files to record the results
    f_train = open(results_file,"w")
    results_file_test = results_file[:-4]+"_test.txt"
    f_test = open(results_file_test,"w")

    # set the optimizer
    optimizer = optimizer(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_step = len(trainloader)

    for epoch in range(max_epochs):
        # Training
        net.train()
        for i, data in enumerate(trainloader,0):
            inputs, labels = data["RAD"].to(device, dtype=torch.float), data["QPE"].to(device, dtype=torch.float)

            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = net(inputs)
            n = outputs.size(0) * outputs.size(1)
            outputs = outputs.view(outputs.size(0), -1)
            labels = labels.view(labels.size(0), -1)

            # calculate loss function (divide by batch size and size of output frames)
            loss = loss_function(outputs, labels)/n
            loss.backward()
            optimizer.step()

            if (i+1) % 40 == 0:
                print('CNN2D|  Epoch [{}/{}], Step [{:03}/{}], Loss: {:8.3f}'
                    .format(epoch+1, max_epochs, i+1, total_step, loss.item()))
                f_train.writelines('Epoch [{}/{}], Step [{:03}/{}], Loss: {:8.3f}\n'
                    .format(epoch+1, max_epochs, i+1, total_step, loss.item()))

        test_loss = test(net, testloader=testloader, loss_function=loss_function, device=device)
        print("CNN2D|  Epoch [{}/{}], Test Loss: {:8.3f}".format(epoch+1, max_epochs, test_loss))
        f_test.writelines("Epoch [{}/{}], Test Loss:{:8.3f}\n".format(epoch+1, max_epochs, test_loss))

    total_params = sum(p.numel() for p in net.parameters())
    print("\nCNN2D|  Total_params: {:.2e}".format(total_params))
    f_train.writelines("\nTotal_params: {:.2e}".format(total_params))
    f_train.close()
    f_test.close()
    torch.save(Net.state_dict(), results_file[:-4]+'.ckpt')

def test(net,testloader,loss_function=BMSE,device=args.device):
    net.eval()
    n = 0
    loss = 0
    with torch.no_grad():
        for i, data in enumerate(testloader,0):
            inputs, labels = data["RAD"].to(device, dtype=torch.float), data["QPE"].to(device, dtype=torch.float)
            outputs = net(inputs)
            n += outputs.shape[0]*outputs.shape[1]
            outputs = outputs.reshape(outputs.size(0), -1)
            labels = labels.reshape(labels.size(0), -1)
            loss += loss_function(outputs, labels)

        loss = loss/n

    return loss

# Run exp1
def run(results_file, input_frames=5, output_frames=18, loss_function="BMSE", max_epochs=50, device=args.device):

    if loss_function == "BMSE":
        loss_function = BMSE
    elif loss_function == "BMAE":
        loss_function = BMAE

    mean = [12.834] * input_frames
    std = [14.14] * input_frames
    transfrom = transforms.Compose([ToTensor(),Normalize(mean=mean, std=std)])

    train_dataset = TyDataset(ty_list_file="../../ty_list.xlsx",
                            root_dir="../../01_TY_database/02_wrangled_data_Taipei",
                            input_frames=input_frames,
                            output_frames=output_frames,
                            train=True,
                            transform = transfrom)

    test_dataset = TyDataset(ty_list_file="../../ty_list.xlsx",
                            root_dir="../../01_TY_database/02_wrangled_data_Taipei",
                            input_frames=input_frames,
                            output_frames=output_frames,
                            train=False,
                            transform = transfrom)
    batch_size = 10
    params = {"batch_size":batch_size, "shuffle":True, "num_workers":1}
    trainloader = DataLoader(train_dataset, **params)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Make CNN2D Net
    c = 70
    encoder_input = input_frames
    encoder_hidden = [c,8*c,12*c,16*c]
    encoder_kernel = [5,4,4,4]
    encoder_n_layer = 4
    encoder_stride = [2,2,2,2]
    encoder_padding = [0,0,1,1]

    decoder_input = 16*c
    decoder_hidden = [16*c,16*c,8*c,4*c,24,output_frames]
    decoder_kernel = [1,4,4,4,6,3]
    decoder_n_layer = 6
    decoder_stride = [1,2,2,2,2,1]
    decoder_padding = [0,1,1,0,0,1]


    Net = model(encoder_input, encoder_hidden, encoder_kernel, encoder_n_layer, encoder_stride, encoder_padding,
                decoder_input, decoder_hidden, decoder_kernel, decoder_n_layer, decoder_stride, decoder_padding,
                batch_norm=True).to(device)

    # print(Net)
    # Train process
    info = "| Forecast frames: {:02d}, Input frames: {:02d} |".format(output_frames, input_frames)
    print("="*len(info))
    print(info)
    print("="*len(info))

    train(net=Net, trainloader=trainloader, testloader=testloader, results_file=results_file,
            max_epochs=max_epochs, loss_function=loss_function, device=device)



##--------------------main function--------------------##
device1 = "cuda:01"

def main():
    results_dir = "../01_results/CNN2D"
    createfolder(results_dir)

    for input_frames in [11,9,7,5]:
        results_file = os.path.join(results_dir,"BMSE_f.{:02d}_x.{:02d}.txt".format(18, input_frames))
        print(results_file)
        run(results_file=results_file, input_frames=input_frames, output_frames=18, loss_function="BMSE", max_epochs=100, device=device1)

if __name__ == "__main__":
    main()
