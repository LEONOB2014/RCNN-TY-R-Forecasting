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
from tools.args_tools import args, createfolder
from tools.dataset_CNN import ToTensor, Normalize, TyDataset

from CNN2D import CNN2D


def BMSE(outputs, labels):
    BMSE = 0
    outputs_size = outputs.shape[0]*outputs.shape[1]
    BMSE += torch.sum(1*(outputs[2>outputs]-labels[2>outputs])**2)
    BMSE += torch.sum(2*(outputs[5>outputs]-labels[5>outputs])**2) - torch.sum(2*(outputs[2>outputs]-labels[2>outputs])**2)
    BMSE += torch.sum(5*(outputs[10>outputs]-labels[10>outputs])**2) - torch.sum(5*(outputs[5>outputs]-labels[5>outputs])**2)
    BMSE += torch.sum(10*(outputs[30>outputs]-labels[30>outputs])**2) - torch.sum(10*(outputs[10>outputs]-labels[10>outputs])**2)
    BMSE += torch.sum(30*(outputs[outputs>=30]-labels[outputs>=30])**2)

    return BMSE/outputs_size

def BMAE(outputs, labels):
    BMAE = 0
    outputs_size = outputs.shape[0]*outputs.shape[1]
    BMAE += torch.sum(1*torch.abs(outputs[2>outputs]-labels[2>outputs]))
    BMAE += torch.sum(2*torch.abs(outputs[5>outputs]-labels[5>outputs])) - torch.sum(2*torch.abs(outputs[2>outputs]-labels[2>outputs]))
    BMAE += torch.sum(5*torch.abs(outputs[10>outputs]-labels[10>outputs])) - torch.sum(5*torch.abs(outputs[5>outputs]-labels[5>outputs]))
    BMAE += torch.sum(10*torch.abs(outputs[30>outputs]-labels[30>outputs])) - torch.sum(10*torch.abs(outputs[10>outputs]-labels[10>outputs]))
    BMAE += torch.sum(30*torch.abs(outputs[outputs>=30]-labels[outputs>=30]))

    return BMAE/outputs_size


def train(net, train_loader, test_loader, results_file, max_epochs=50, loss_function=BMSE,
            optimizer=optim.Adam, device=args.device):
    f = open(results_file,"w")
    criterion = loss_function
    optimizer = optimizer(net.parameters())
    total_step = len(train_loader)

    for epoch in range(max_epochs):
        # Training
        for i, data in enumerate(train_loader,0):
            inputs, labels = data["RAD"].to(device, dtype=torch.float), data["QPE"].to(device, dtype=torch.float)

            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = net(inputs)
            labels = labels.view(labels.shape[0], labels.shape[1]*labels.shape[2])

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i+1) % 200 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:>.3f}'
                        .format(epoch+1, max_epochs, i+1, total_step, loss.item()))
                    f.writelines('Epoch [{}/{}], Step [{}/{}], Loss: {:>.3f}\n'
                        .format(epoch+1, max_epochs, i+1, total_step, loss.item()))

        test_loss = test(net, test_loader=test_loader, loss_function=criterion, device=device)
        print("Epoch [{}/{}], Test Loss:{:>.3f}\n".format(epoch+1, max_epochs, loss))
        f.writelines("Epoch [{}/{}], Test Loss:{:>.3f}\n".format(epoch+1, max_epochs, loss))

    total_params = sum(p.numel() for p in Net.parameters())
    print("Total_params: {:d}".format(total_params))
    f.writelines("Total_params: {:d}".format(total_params))
    f.close()

def test(net,test_loader,loss_function=nn.MSELoss(),device=args.device):
    net.eval()
    criterion = loss_function
    loss = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader,0):
            inputs, labels = data["RAD"].to(device, dtype=torch.float), data["QPE"].to(device, dtype=torch.float)
            outputs = net(inputs)

            labels = labels.view(labels.shape[0], labels.shape[1]*labels.shape[2])

            loss += criterion(outputs, labels)
        loss = loss/(i+1)

    return loss

# Run exp1
def run(results_file, x_tsteps=6, forecast_t=1, loss_function="BMSE", max_epochs=50, device=args.device):

    if loss_function == "BMSE":
        loss_function = BMSE
    elif loss_function == "BMAE":
        loss_function = BMAE

    mean = [12.834] * x_tsteps
    std = [14.14] * x_tsteps
    transfrom = transforms.Compose([ToTensor(),Normalize(mean=mean, std=std)])

    train_dataset = TyDataset(ty_list_file="../../ty_list.xlsx",
                            root_dir="../../01_TY_database/02_wrangled_data_Taipei",
                            x_tsteps=x_tsteps,
                            forecast_t=forecast_t,
                            train=True,
                            transform = transfrom)

    test_dataset = TyDataset(ty_list_file="../../ty_list.xlsx",
                            root_dir="../../01_TY_database/02_wrangled_data_Taipei",
                            x_tsteps=x_tsteps,
                            forecast_t=forecast_t,
                            train=False,
                            transform = transfrom)

    params = {"batch_size":4, "shuffle":True, "num_workers":1}
    train_generator = DataLoader(train_dataset, **params)
    test_generator = DataLoader(test_dataset, **params)

    # Make CNN2D Net
    Net = CNN2D(n_input=x_tsteps,n_hidden=[x_tsteps+1,x_tsteps+2,x_tsteps+3],kernel_size=[3,3,3],n_hid_layers=3,
                n_fully=72*72*(x_tsteps+3),n_fully_layers=1,n_out_layer=72*72,batch_norm=True).to(device)
    # print(Net)
    # Train process
    info = "| Forecast time: {:02d}, Inputs time steps: {:02d} |".format(forecast_t,x_tsteps)
    print("="*len(info))
    print(info)
    print("="*len(info))

    train(net=Net, train_loader=train_generator, test_loader=test_generator, results_file=results_file,
            max_epochs=max_epochs, loss_function=loss_function, device=device)
    total_params = torch.sum(p.numel() for p in Net.parameters())
    print(total_params)
    torch.save(Net.state_dict(), results_file[:-4]+'.ckpt')


##--------------------main function--------------------##
device = "cuda:2"

def main():
    results_dir = "../01_results/CNN2D"
    createfolder(results_dir)

    for forecast_t in range(1,4):
        for x_tsteps in range(3,10):
            results_file = os.path.join(results_dir,"BMSE_f.{:02d}_x.{:02d}.txt".format(forecast_t,x_tsteps))

            run(results_file=results_file,x_tsteps=x_tsteps,forecast_t=forecast_t,loss_function="BMSE",max_epochs=1,device=device)

if __name__ == "__main__":
    main()
