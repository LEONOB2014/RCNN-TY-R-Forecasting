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
from tools.dataset_GRU import ToTensor, Normalize, TyDataset
from tools.loss_function import BMAE, BMSE
from convGRU import model

def train(net, trainloader, testloader, results_file, max_epochs=50, loss_function=BMSE,
        optimizer=optim.Adam, device=args.device):
    net.train()
    # open a new file to save results.
    f_train = open(results_file,"w")
    test_file = results_file[:-4]+"_test.txt"
    f_test = open(test_file,"w")

    # Set optimizer
    optimizer = optimizer(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_step = len(trainloader)

    for epoch in range(max_epochs):
        # Training
        for i, data in enumerate(trainloader,0):
            inputs = data["RAD"].to(device, dtype=torch.float)
            labels = data["QPE"].to(device, dtype=torch.float)

            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = net(inputs)
            n = outputs.size(0) * outputs.size(1)
            outputs = outputs.view(outputs.shape[0], -1)
            labels = labels.view(labels.shape[0], -1)

            # calculate loss function
            loss = loss_function(outputs, labels)/n
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=args.clip_max_norm)

            optimizer.step()
            if (i+1) % 40 == 0:
                print('ConvGRUv2|  Epoch [{}/{}], Step [{}/{}], Loss: {:.3f}'
                    .format(epoch+1, max_epochs, i+1, total_step, loss.item()))
                f_train.writelines('Epoch [{}/{}], Step [{}/{}], Loss: {:.3f}\n'
                    .format(epoch+1, max_epochs, i+1, total_step, loss.item()))

        # Save the test loss per epoch
        test_loss = test(net, testloader=testloader, loss_function=loss_function, device=device)

        print("ConvGRUv2|  Epoch [{}/{}], Test Loss: {:8.3f}".format(epoch+1, max_epochs, test_loss))
        f_test.writelines("Epoch [{}/{}], Test Loss: {:8.3f}\n".format(epoch+1, max_epochs, test_loss))

    total_params = sum(p.numel() for p in net.parameters())
    print("\nConvGRUv2|  Total_params: {:.2e}".format(total_params))
    f_train.writelines("\nTotal_params: {:.2e}".format(total_params))
    f_train.close()
    f_test.close()
    torch.save(Net.state_dict(), results_file[:-4]+'.ckpt')

def test(net, testloader, loss_function=nn.MSELoss(), device=args.device):
    net.eval()

    loss = 0
    n = 0
    with torch.no_grad():
        for i, data in enumerate(testloader,0):
            inputs, labels = data["RAD"].to(device, dtype=torch.float), data["QPE"].to(device, dtype=torch.float)
            outputs = net(inputs)
            n += outputs.shape[0]*outputs.shape[1]
            outputs = outputs.view(outputs.shape[0], -1)
            labels = labels.view(labels.shape[0], -1)
            loss += loss_function(outputs, labels)

        loss = loss/n
    return loss

def get_dataloader(input_frames, output_frames):
    # Normalize data
    mean = [12.834] * input_frames
    std = [14.14] * input_frames
    transfrom = transforms.Compose([ToTensor(),Normalize(mean=mean, std=std)])

    # set train and test dataset
    traindataset = TyDataset(ty_list_file="../../ty_list.xlsx",
                        input_frames=input_frames,
                        output_frames=output_frames,
                        input_size = 180,
                        output_size = 60,
                        train=True,
                        root_dir=os.path.join("../../01_Radar_data",args.files_folder),
                        transform = transfrom)
    testdataset = TyDataset(ty_list_file="../../ty_list.xlsx",
                        input_frames=input_frames,
                        output_frames=output_frames,
                        input_size = 180,
                        output_size = 60,
                        train=False,
                        root_dir=os.path.join("../../01_Radar_data",args.files_folder),
                        transform = transfrom)

    # set train and test dataloader
    params = {"batch_size":args.batch_size, "shuffle":True, "num_workers":1}
    trainloader = DataLoader(traindataset, **params)
    testloader = DataLoader(testdataset, batch_size=args.batch_size, shuffle=False)

    return trainloader, testloader


def run(results_file, channel_factor=3, input_frames=5, output_frames=18,
        loss_function="BMSE", max_epochs=50, device=args.device):

    if loss_function == "BMSE":
        loss_function = BMSE
    elif loss_function == "BMAE":
        loss_function = BMAE

    trainloader, testloader = get_dataloader(input_frames=input_frames, output_frames=output_frames)
    c = channel_factor

    # Make ConvGRU Net
    # initialize the parameters of the encoders and decoders
    encoder_input = 1
    encoder_downsample = [2*c,32*c,96*c]
    encoder_crnn = [32*c,96*c,96*c]
    encoder_kernel_downsample = [5,4,4]
    encoder_kernel_crnn = [3,3,3]
    encoder_stride_downsample = [3,2,2]
    encoder_stride_crnn = [1,1,1]
    encoder_padding_downsample = [1,1,1]
    encoder_padding_crnn = [1,1,1]
    encoder_n_layers = 6

    decoder_input=0
    decoder_upsample = [96*c,96*c,4*c]
    decoder_crnn = [96*c,96*c,32*c]
    decoder_kernel_upsample = [4,4,5]
    decoder_kernel_crnn = [3,3,3]
    decoder_stride_upsample = [2,2,3]
    decoder_stride_crnn = [1,1,1]
    decoder_padding_upsample = [1,1,1]
    decoder_padding_crnn = [1,1,1]
    decoder_n_layers = 6

    decoder_output = 1
    decoder_output_kernel = 5
    decoder_output_stride = 3
    decoder_output_padding = 1
    decoder_output_layers = 1

    Net = model(n_encoders=input_frames, n_decoders=output_frames,
                encoder_input=encoder_input, encoder_downsample=encoder_downsample, encoder_crnn=encoder_crnn,
                encoder_kernel_downsample=encoder_kernel_downsample, encoder_kernel_crnn=encoder_kernel_crnn,
                encoder_stride_downsample=encoder_stride_downsample, encoder_stride_crnn=encoder_stride_crnn,
                encoder_padding_downsample=encoder_padding_downsample, encoder_padding_crnn=encoder_padding_crnn,
                encoder_n_layers=encoder_n_layers,
                decoder_input=decoder_input, decoder_upsample=decoder_upsample, decoder_crnn=decoder_crnn,
                decoder_kernel_upsample=decoder_kernel_upsample, decoder_kernel_crnn=decoder_kernel_crnn,
                decoder_stride_upsample=decoder_stride_upsample, decoder_stride_crnn=decoder_stride_crnn,
                decoder_padding_upsample=decoder_padding_upsample, decoder_padding_crnn=decoder_padding_crnn,
                decoder_n_layers=decoder_n_layers, decoder_output=1, decoder_output_kernel= decoder_output_kernel,
                decoder_output_stride=decoder_output_stride, decoder_output_padding=decoder_output_padding,
                decoder_output_layers=decoder_output_layers, batch_norm=False).to(device, dtype=torch.float)
    info = "| Channel factor c: {:02d}, Forecast frames: {:02d}, Input frames: {:02d} |".format(channel_factor, output_frames,input_frames)
    print("="*len(info))
    print(info)
    print("="*len(info))
    train(net=Net, trainloader=trainloader, testloader=testloader, results_file=results_file,
            max_epochs=max_epochs, loss_function=loss_function, device=device)

# set gpu or cpu
def main():
    output_frames = 18
    channel_factor = 2
    input_frames = 10
    results_dir = "../01_results/ConvGRUv2_c.{:d}".format(channel_factor)
    createfolder(results_dir)
    results_name = os.path.join(results_dir,"BMSE_f.{:02d}_x.{:02d}.txt".format(output_frames,input_frames))
    print(results_name)
    run(results_name, channel_factor=channel_factor, input_frames=input_frames, output_frames=output_frames, loss_function="BMSE", max_epochs=100, device=args.device)

if __name__ == "__main__":
    main()
