from CNNGRU_cell import *

class ConvGRU(nn.Module):

    def __init__(self, channel_input, channel_hidden, kernel_sizes, n_layers):
        '''
        Generates a multi-layer convolutional GRU.
        Preserves spatial dimensions across cells, only altering depth.
        Parameters
        ----------
        channel_input : integer. depth dimension of input tensors.
        channel_hidden : integer or list. depth dimensions of hidden state.
            if integer, the same hidden size is used for all cells.
        kernel_sizes : integer or list. sizes of Conv2d gate kernels.
            if integer, the same kernel size is used for all cells.
        n_layers : integer. number of chained `ConvGRUCell`.
        '''
        super(ConvGRU, self).__init__()

        self.channel_input = channel_input

        if type(channel_hidden) != list:
            self.channel_hidden = [channel_hidden]*n_layers
        else:
            assert len(channel_hidden) == n_layers, '`channel_hidden` must have the same length as n_layers'
            self.channel_hidden = channel_hidden
        if type(kernel_sizes) != list:
            self.kernel_sizes = [kernel_sizes]*n_layers
        else:
            assert len(kernel_sizes) == n_layers, '`kernel_sizes` must have the same length as n_layers'
            self.kernel_sizes = kernel_sizes

        self.n_layers = n_layers

        cells = []
        for i in range(self.n_layers):
            if i == 0:
                input_dim = self.channel_input
            else:
                input_dim = self.channel_hidden[i-1]

            cell = ConvGRUCell(input_dim, self.channel_hidden[i], self.kernel_sizes[i])
            name = 'ConvGRUCell_' + str(i).zfill(2)

            setattr(self, name, cell)
            cells.append(getattr(self, name))

        self.cells = cells


    def forward(self, x, hidden=None):
        '''
        Parameters
        ----------
        x : 4D input tensor. (batch, channels, height, width).
        hidden : list of 4D hidden state representations. (batch, channels, height, width).
        Returns
        -------
        upd_hidden : 5D hidden representation. (layer, batch, channels, height, width).
        '''
        if not hidden:
            hidden = [None]*self.n_layers

        input_ = x

        upd_hidden = []

        for layer_idx in range(self.n_layers):
            cell = self.cells[layer_idx]
            cell_hidden = hidden[layer_idx]

            # pass through layer
            upd_cell_hidden = cell(input_, cell_hidden)
            upd_hidden.append(upd_cell_hidden)
            # update input_ to the last updated hidden layer for next pass
            input_ = upd_cell_hidden

        # retain tensors in list to allow different hidden sizes
        return upd_hidden


class DeconvGRU(nn.Module):

    def __init__(self, channel_input, channel_hidden, channel_output, kernel_sizes, n_layers):
        '''
        Generates a multi-layer convolutional GRU.
        Preserves spatial dimensions across cells, only altering depth.
        Parameters
        ----------
        channel_input : integer. depth dimension of input tensors.
        channel_hidden : integer or list. depth dimensions of hidden state.
            if integer, the same hidden size is used for all cells.
        kernel_sizes : integer or list. sizes of Conv2d gate kernels.
            if integer, the same kernel size is used for all cells.
        n_layers : integer. number of chained `ConvGRUCell`.
        '''

        super(DeconvGRU, self).__init__()

        self.channel_input = channel_input
        self.channel_output = channel_output

        if type(channel_hidden) != list:
            self.channel_hidden = [channel_hidden]*n_layers
        else:
            assert len(channel_hidden) == n_layers, '`channel_hidden` must have the same length as n_layers'
            self.channel_hidden = channel_hidden
        if type(kernel_sizes) != list:
            self.kernel_sizes = [kernel_sizes]*n_layers
        else:
            assert len(kernel_sizes) == n_layers, '`kernel_sizes` must have the same length as n_layers'
            self.kernel_sizes = kernel_sizes

        self.n_layers = n_layers

        cells = []
        for i in range(self.n_layers):
            if i == 0:
                input_dim = self.channel_input
            else:
                input_dim = self.channel_hidden[i-1]

            cell = DeconvGRUCell(input_dim, self.channel_hidden[i], self.kernel_sizes[i])
            name = 'DeconvGRUCell_' + str(i).zfill(2)

            setattr(self, name, cell)
            cells.append(getattr(self, name))

        cell = nn.ConvTranspose2d(self.channel_hidden[i], self.channel_output, 1)
        setattr(self, "OutputDeconv", cell)
        cells.append(cell)
        self.cells = cells

    def forward(self, hidden):
        '''
        Parameters
        ----------
        x : 4D input tensor. (batch, channels, height, width).
        hidden : list of 4D hidden state representations. (batch, channels, height, width).
        Returns
        -------
        upd_hidden : 5D hidden representation. (layer, batch, channels, height, width).
        '''
        input_ = None

        upd_hidden = []

        for layer_idx in range(self.n_layers):
            cell = self.cells[layer_idx]
            cell_hidden = hidden[layer_idx]

            # pass through layer
            upd_cell_hidden = cell(input_, cell_hidden)
            upd_hidden.append(upd_cell_hidden)
            # update input_ to the last updated hidden layer for next pass
            input_ = upd_cell_hidden

        cell = self.cells[self.n_layers]
        output = cell(upd_cell_hidden)

        # retain tensors in list to allow different hidden sizes
        return upd_hidden, output

class model(nn.Module):
    def __init__(self, n_encoders, n_decoders,
                    encoder_input, encoder_hidden, encoder_kernel, encoder_n_layers,
                    decoder_input, decoder_hidden, decoder_output, decoder_kernel, decoder_n_layers,
                    padding=True, batch_norm=False):
        super().__init__()
        self.n_encoders = n_encoders
        self.n_decoders = n_decoders

        models = []
        for i in range(self.n_encoders):
            model = ConvGRU(channel_input=encoder_input, channel_hidden=encoder_hidden,
                            kernel_sizes=encoder_kernel, n_layers=encoder_n_layers)
            name = 'Encoder_' + str(i).zfill(2)
            setattr(self, name, model)
            models.append(getattr(self, name))

        for i in range(self.n_decoders):
            model = DeconvGRU(channel_input=decoder_input, channel_hidden=decoder_hidden, channel_output=decoder_output,
                                kernel_sizes=decoder_kernel, n_layers=decoder_n_layers)
            name = 'Decoder_' + str(i).zfill(2)
            setattr(self, name, model)
            models.append(getattr(self, name))

        self.models = models

    def forward(self, x):
        if x.size()[0] != self.n_encoders:
            assert x.size()[1] == self.n_encoders, '`x` must have the same as n_encoders'

        forecast = []

        for i in range(self.n_encoders):
            if i == 0:
                hidden=None
            model = self.models[i]
            hidden = model(x=x[:,i,:,:,:],hidden=hidden)

        hidden = hidden[::-1]

        for i in range(self.n_encoders,self.n_encoders+self.n_decoders):
            model = self.models[i]
            hidden, output = model(hidden=hidden)
            forecast.append(output)
        forecast = torch.cat(forecast,dim=1)

        return forecast
