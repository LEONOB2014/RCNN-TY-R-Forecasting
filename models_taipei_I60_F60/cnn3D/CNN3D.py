import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN3D_cell(nn.Module):
    def __init__(self, n_input, n_hidden, kernel_size, padding=True, batch_norm=False):
        super().__init__()

        if padding:
            self.padding = kernel_size // 2
        self.n_input = n_input
        self.n_hidden = n_hidden

        layer_sublist = []
        layer_sublist.append(nn.Conv2d(n_input, n_hidden, kernel_size, padding=self.padding))
        if batch_norm:
            layer_sublist.append(nn.BatchNorm2d(n_hidden))
        layer_sublist.append(nn.ReLU())

        nn.init.orthogonal_(layer_sublist[0].weight)
        nn.init.constant_(layer_sublist[0].bias, 0.)

        self.layer = nn.Sequential(*layer_sublist)

    def forward(self, input_):
        out = self.layer(input_)
        return out


class CNN3D(nn.Module):
    '''
    Generate a 3-D convolutional neural network.
    '''

    def __init__(self, n_input, n_hidden, kernel_size, n_hid_layers, n_fully, n_fully_layers, n_out_layer, padding=True, batch_norm=False):
        '''
        n_input: integral. the channel size of input tensors.
        n_hidden: integer or list. the channel size of hidden layers.
                if integral, the same hidden size is used for all layers.
        n_hid_layers: integral. the number of hidden layers (int)
        kernel_size: integer or list. the kernel size of each hidden layers.
                if integer, the same kernel size is used for all layers.
        padding: boolean, the decision to do padding
        batch_norm = boolean, the decision to do batch normalization
        '''
        super().__init__()

        self.n_input = n_input

        if type(n_hidden) != list:
            self.n_hidden = [n_hidden]*n_hid_layers
        else:
            assert len(n_hidden) == n_hid_layers, '`n_hidden` must have the same length as n_hid_layers'
            self.n_hidden = n_hidden

        if type(kernel_size) != list:
            self.kernel_size = [kernel_size]*n_hid_layers
        else:
            assert len(kernel_size) == n_hid_layers, '`kernel_size` must have the same length as n_hid_layers'
            self.kernel_size = kernel_size

        if type(n_fully) != list:
            self.n_fully = [n_fully]*n_fully_layers
        else:
            assert len(n_fully) == n_fully_layers, '`n_hidden` must have the same length as n_hid_layers'
            self.n_fully = n_fully

        self.n_hid_layers = n_hid_layers
        self.n_fully_layers = n_fully_layers
        self.n_out_layer = n_out_layer

        if padding:
            self.padding = list(np.array(kernel_size) // 2)

        # nn layers
        layers = []
        for layer_idx in range(self.n_hid_layers):
            if layer_idx == 0:
                input_dim = self.n_input
            else:
                input_dim = self.n_hidden[layer_idx-1]

            layer = CNN2D_cell(input_dim, self.n_hidden[layer_idx], self.kernel_size[layer_idx], batch_norm=batch_norm)
            name = 'Conv_' + str(layer_idx).zfill(2)
            setattr(self, name, layer)
            layers.append(getattr(self, name))

        for layer_idx in range(self.n_fully_layers):
            if self.n_fully_layers == 1:
                layer_idx = 0
                break
            if layer_idx == 0:
                input_dim = self.n_fully[layer_idx]
            else:
                input_dim = self.n_fully[layer_idx-1]
            layer = nn.Linear(input_dim, self.n_fully[layer_idx])
            name = 'Fc_' + str(layer_idx).zfill(2)
            setattr(self, name, layer)
            layers.append(getattr(self, name))

        layer = nn.Linear(self.n_fully[layer_idx], self.n_out_layer)
        name = 'Output_' + str(layer_idx).zfill(2)
        setattr(self, name, layer)
        layers.append(getattr(self, name))

        self.layers = layers


    def forward(self, x):
        '''
        Parameters
        ----------
        x : 4D input tensor. (batch, channels, height, width).
        '''

        input_ = x

        for layer_idx in range(self.n_hid_layers):
            layer = self.layers[layer_idx]
            # pass through layers
            out_hidden = layer(input_)
            # update input_ to the last updated hidden layer for next pass
            input_ = out_hidden

        input_ = input_.reshape(input_.size(0), -1)

        for layer_idx in range(self.n_hid_layers, self.n_fully_layers+self.n_hid_layers):
            layer = self.layers[layer_idx]

            # pass through layers
            out_hidden = layer(input_)
            # update input_ to the last updated hidden layer for next pass
            input_ = out_hidden

        # retain tensors in list to allow different hidden sizes
        output = input_
        return output
