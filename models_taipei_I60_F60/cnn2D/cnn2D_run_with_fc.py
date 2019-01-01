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
from tools.loss_function import BMAE, BMSE
from CNN2D import *

def train(net, trainloader, testloader, results_file, max_epochs=50, loss_function=BMSE,
            optimizer=optim.Adam, device=args.device):
    f_train = open(results_file,"w")
    results_file_test = results_file[:-4]+"_test.txt"
    f_test = open(results_file_test,"w")

    optimizer = optimizer(net.parameters(), lr=1e-4)
    total_step = len(trainloader)

    for epoch in range(max_epochs):
        # Training
        net.train()
        for i, data in enumerate(trainloader,0):
            inputs, labels = data["RAD"].to(device, dtype=torch.float), data["QPE"].to(device, dtype=torch.float)

            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = net(inputs)
            outputs = outputs.view(outputs.size(0), -1)
            labels = labels.view(labels.size(0), -1)

            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i+1) % 40 == 0:
                print('Epoch [{}/{}], Step [{:03}/{}], Loss: {:8.3f}'
                    .format(epoch+1, max_epochs, i+1, total_step, loss.item()))
                f_train.writelines('Epoch [{}/{}], Step [{:03}/{}], Loss: {:8.3f}\n'
                    .format(epoch+1, max_epochs, i+1, total_step, loss.item()))

        test_loss = test(net, testloader=testloader, loss_function=loss_function, device=device)
        print("Epoch [{}/{}], Test Loss: {:8.3f}".format(epoch+1, max_epochs, test_loss))
        f_test.writelines("Epoch [{}/{}], Test Loss:{:8.3f}\n".format(epoch+1, max_epochs, test_loss))

    total_params = sum(p.numel() for p in net.parameters())
    print("\nTotal_params: {:.2e}".format(total_params))
    f_train.writelines("\nTotal_params: {:.2e}".format(total_params))
    f_train.close()
    f_test.close()

def test(net,testloader,loss_function=nn.MSELoss(),device=args.device):
    net.eval()

    loss = 0
    with torch.no_grad():
        for i, data in enumerate(testloader,0):
            inputs, labels = data["RAD"].to(device, dtype=torch.float), data["QPE"].to(device, dtype=torch.float)
            outputs = net(inputs)
            outputs = outputs.reshape(outputs.size(0), -1)
            labels = labels.reshape(labels.size(0), -1)

            loss += loss_function(outputs, labels)
        loss = loss/((i+1)*inputs.size(0))

    return loss

# Run exp1
def run(results_file, x_tsteps=6, n_layers=3, forecast_t=1, loss_function="BMSE", max_epochs=50, device=args.device):

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
    batch_size = 10
    params = {"batch_size":batch_size, "shuffle":True, "num_workers":1}
    trainloader = DataLoader(train_dataset, **params)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Make CNN2D Net
    encoder_input = x_tsteps
    encoder_hidden = [x_tsteps+2,x_tsteps+4,x_tsteps+6]
    encoder_kernel = [5,5,5]
    encoder_n_layer = 3
    decoder_input = x_tsteps+6
    decoder_hidden = [x_tsteps+3,x_tsteps-1,3]
    decoder_kernel = [5,5,5]
    decoder_n_layer = 3
    Net = model(encoder_input=encoder_input, encoder_hidden=encoder_hidden, encoder_kernel=encoder_kernel, encoder_n_layer=encoder_n_layer,
                decoder_input=decoder_input, decoder_hidden=decoder_hidden, decoder_kernel=decoder_kernel, decoder_n_layer=decoder_n_layer,
                fully_input = 3*72*72, fully_hidden=[72*72], fully_layers=1,
                padding=True,batch_norm=True).to(device)
    # print(Net)
    # Train process
    info = "| Forecast time: {:02d}, Inputs time steps: {:02d} |".format(forecast_t,x_tsteps)
    print("="*len(info))
    print(info)
    print("="*len(info))

    train(net=Net, trainloader=trainloader, testloader=testloader, results_file=results_file,
            max_epochs=max_epochs, loss_function=loss_function, device=device)

    torch.save(Net.state_dict(), results_file[:-4]+'.ckpt')


##--------------------main function--------------------##
device1 = "cuda:01"
def main():
    results_dir = "../01_results/CNN2D_En_De_Fc"
    createfolder(results_dir)

    for forecast_t in range(1,4):
        for x_tsteps in range(3,7):
            results_file = os.path.join(results_dir,"BMSE_x.{:02d}_f.{:02d}.txt".format(x_tsteps,forecast_t))
            print(results_file)
            run(results_file=results_file, x_tsteps=x_tsteps ,forecast_t=forecast_t, loss_function="BMSE", max_epochs=50, device=device1)

if __name__ == "__main__":
    main()
