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
sys.path.append(os.path.abspath('..'))

from tools.args_tools import args, createfolder
from tools.datasetGRU import ToTensor, Normalize, TyDataset
from tools.loss_function import BMAE, BMSE
from convGRU import model

def train(net, trainloader, testloader, result_name, max_epochs=50, loss_function=BMSE,
        optimizer=optim.Adam, device=args.device):
    net.train()

    # Set optimizer
    optimizer = optimizer(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_step = len(trainloader)

    for epoch in range(max_epochs):
        # Training
        # open a new file to save result.
        f_train = open(result_name,"w")
        test_file = result_name[:-4]+"_test.txt"
        f_test = open(test_file,"w")

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
        if (epoch+1) % 10 == 0:
            torch.save(net.state_dict(), result_name[:-4]+'_{:d}.ckpt'.format(epoch+1))
        if (epoch+1) == max_epochs:
            total_params = sum(p.numel() for p in net.parameters())
            print("\nConvGRUv2|  Total_params: {:.2e}".format(total_params))
            f_train.writelines("\nTotal_params: {:.2e}".format(total_params))
        f_train.close()
        f_test.close()


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
    traindataset = TyDataset(ty_list_file=args.ty_list_file,
                        input_frames=input_frames,
                        output_frames=output_frames,
                        train=True,
                        root_dir=args.root_dir,
                        transform = transfrom)
    testdataset = TyDataset(ty_list_file=args.ty_list_file,
                        input_frames=input_frames,
                        output_frames=output_frames,
                        train=False,
                        root_dir=args.root_dir,
                        transform = transfrom)

    # set train and test dataloader
    params = {"batch_size":args.batch_size, "shuffle":True, "num_workers":1}
    trainloader = DataLoader(traindataset, **params)
    testloader = DataLoader(testdataset, batch_size=args.batch_size, shuffle=False)

    return trainloader, testloader


def run(result_name, channel_factor, input_frames, output_frames,
        loss_function=BMSE, max_epochs=50, device=args.device):

    # if loss_function == "BMSE":
    #     loss_function = BMSE
    # elif loss_function == "BMAE":
    #     loss_function = BMAE

    # get dataloader
    trainloader, testloader = get_dataloader(input_frames=input_frames, output_frames=output_frames)

    # set the factor of cnn channels
    c = channel_factor

    # construct convGRU net
    # initialize the parameters of the encoders and decoders
    encoder_input = 1
    encoder_downsample_layer = [2*c,32*c,96*c]
    encoder_crnn_layer = [32*c,96*c,96*c]

    if int(args.I_shape[0]/3) == args.F_shape[0]:
        encoder_downsample_k = [5,4,4]
        encoder_downsample_s = [3,2,2]
        encoder_downsample_p = [1,1,1]
    elif args.I_shape[0] == args.F_shape[0]:
        encoder_downsample_k = [3,4,4]
        encoder_downsample_s = [1,2,2]
        encoder_downsample_p = [1,1,1]


    encoder_crnn_k = [3,3,3]
    encoder_crnn_s = [1,1,1]
    encoder_crnn_p = [1,1,1]
    encoder_n_layers = 6

    decoder_input=0
    decoder_upsample_layer = [96*c,96*c,4*c]
    decoder_crnn_layer = [96*c,96*c,32*c]

    decoder_upsample_k = [4,4,3]
    decoder_upsample_s = [2,2,1]
    decoder_upsample_p = [1,1,1]

    decoder_crnn_k = [3,3,3]
    decoder_crnn_s = [1,1,1]
    decoder_crnn_p = [1,1,1]
    decoder_n_layers = 6

    decoder_output = 1
    decoder_output_k = 3
    decoder_output_s = 1
    decoder_output_p = 1
    decoder_output_layers = 1

    Net = model(n_encoders=input_frames, n_decoders=output_frames,
                encoder_input=encoder_input, encoder_downsample_layer=encoder_downsample_layer, encoder_crnn_layer=encoder_crnn_layer,
                encoder_downsample_k=encoder_downsample_k, encoder_crnn_k=encoder_crnn_k,
                encoder_downsample_s=encoder_downsample_s, encoder_crnn_s=encoder_crnn_s,
                encoder_downsample_p=encoder_downsample_p, encoder_crnn_p=encoder_crnn_p,
                encoder_n_layers=encoder_n_layers,
                decoder_input=decoder_input, decoder_upsample_layer=decoder_upsample_layer, decoder_crnn_layer=decoder_crnn_layer,
                decoder_upsample_k=decoder_upsample_k, decoder_crnn_k=decoder_crnn_k,
                decoder_upsample_s=decoder_upsample_s, decoder_crnn_s=decoder_crnn_s,
                decoder_upsample_p=decoder_upsample_p, decoder_crnn_p=decoder_crnn_p,
                decoder_n_layers=decoder_n_layers, decoder_output=1, decoder_output_k=decoder_output_k,
                decoder_output_s=decoder_output_s, decoder_output_p=decoder_output_p,
                decoder_output_layers=decoder_output_layers, batch_norm=False).to(device, dtype=torch.float)
    info = "| Channel factor c: {:02d}, Forecast frames: {:02d}, Input frames: {:02d} |".format(channel_factor, output_frames,input_frames)
    print("="*len(info))
    print(info)
    print("="*len(info))
    train(net=Net, trainloader=trainloader, testloader=testloader, result_name=result_name,
            max_epochs=max_epochs, loss_function=loss_function, device=device)

def main():
    if args.root_dir == None:
        print("Please set the directory of root data ")
        return
    elif args.ty_list_file == None:
        print("Please set typhoon list file")
        return
    elif args.result_dir == None:
        print("Please set the directory of the result")
        return
    else:
        # set the parameters of the experiment
        output_frames = 18
        channel_factor = 2
        input_frames = 10

        result_dir = os.path.join(args.result_dir,"I{:d}_F{:d}".format(args.I_shape[0],args.F_shape[0]),"convGRU_c.{:d}".format(channel_factor))
        createfolder(result_dir)
        result_name = os.path.join(result_dir,"BMSE_f.{:02d}_x.{:02d}_w{:f}.txt".format(output_frames,input_frames,args.weight_decay))
        print(result_name)
        run(result_name=result_name, channel_factor=channel_factor, input_frames=input_frames, output_frames=output_frames,
            loss_function=BMSE, max_epochs=100, device=args.device)


if __name__ == "__main__":
    main()
