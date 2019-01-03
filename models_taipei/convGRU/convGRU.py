from convGRU_cell import *

class ConvGRU(nn.Module):
    def __init__(self, channel_input, channel_downsample, channel_crnn,
                kernel_downsample, kernel_crnn, stride_downsample, stride_crnn,
                padding_downsample, padding_crnn, n_layers):
        '''
        Generates a multi-layer convolutional GRU.
        Preserves spatial dimensions across cells, only altering depth.
        Parameters
        ----------
        channel_input : integer. depth dimension of input tensors.
        channel_downsample : integer or list. depth dimensions of downsample.
        channel_downsample : integer or list. depth dimensions of downsample.
        channel_crnn : integer or list. depth dimensions of crnn.
        stride_downsample: integer or list. the stride size of each downsample layers.
        stride_crnn: integer or list. the stride size of each crnn layers.
        padding_downsample: integer or list. the padding size of each downsample layers.
        padding_crnn: integer or list. the padding size of each crnn layers.
        n_layers : integer. number of chained `ConvGRUCell`.
        '''
        super(ConvGRU, self).__init__()

        self.channel_input = channel_input

        # channel size
        if type(channel_downsample) != list:
            self.channel_downsample = [channel_downsample]*int(n_layers/2)
        else:
            assert len(channel_downsample) == int(n_layers/2), '`channel_downsample` must have the same length as n_layers/2'
            self.channel_downsample = channel_downsample

        if type(channel_crnn) != list:
            self.channel_crnn = [channel_crnn]*int(n_layers/2)
        else:
            assert len(channel_crnn) == int(n_layers/2), '`channel_crnn` must have the same length as n_layers/2'
            self.channel_crnn = channel_crnn

        # kernel size
        if type(kernel_downsample) != list:
            self.kernel_downsample = [kernel_downsample]*int(n_layers/2)
        else:
            assert len(kernel_downsample) == int(n_layers/2), '`kernel_downsample` must have the same length as n_layers/2'
            self.kernel_downsample = kernel_downsample

        if type(kernel_crnn) != list:
            self.kernel_crnn = [kernel_crnn]*int(n_layers/2)
        else:
            assert len(kernel_crnn) == int(n_layers/2), '`kernel_crnn` must have the same length as n_layers/2'
            self.kernel_crnn = kernel_crnn

       # stride size
        if type(stride_downsample) != list:
            self.stride_downsample = [stride_downsample]*int(n_layers/2)
        else:
            assert len(stride_downsample) == int(n_layers/2), '`stride_downsample` must have the same length as n_layers/2'
            self.stride_downsample = stride_downsample

        if type(stride_crnn) != list:
            self.stride_crnn = [stride_crnn]*int(n_layers/2)
        else:
            assert len(stride_crnn) == int(n_layers/2), '`stride_crnn` must have the same length as n_layers/2'
            self.stride_crnn = stride_crnn

        # padding size
        if type(padding_downsample) != list:
            self.padding_downsample = [padding_downsample]*int(n_layers/2)
        else:
            assert len(padding_downsample) == int(n_layers/2), '`padding_downsample` must have the same length as n_layers/2'
            self.padding_downsample = padding_downsample

        if type(padding_crnn) != list:
            self.padding_crnn = [padding_crnn]*int(n_layers/2)
        else:
            assert len(padding_crnn) == int(n_layers/2), '`padding_crnn` must have the same length as n_layers/2'
            self.padding_crnn = padding_crnn

        self.n_layers = n_layers

        cells = []
        for i in range(int(self.n_layers/2)):
            if i == 0:
                cell = nn.Conv2d(self.channel_input, self.channel_downsample[i], self.kernel_downsample[i], self.stride_downsample[i], self.padding_downsample[i])
            else:
                cell = nn.Conv2d(self.channel_crnn[i-1], self.channel_downsample[i], self.kernel_downsample[i], self.stride_downsample[i], self.padding_downsample[i])

            name = 'Downsample_' + str(i).zfill(2)
            setattr(self, name, cell)
            cells.append(getattr(self, name))

            cell = ConvGRUCell(self.channel_downsample[i], self.channel_crnn[i], self.kernel_crnn[i], self.stride_crnn[i], self.padding_crnn[i])
            name = 'ConvGRUCell_' + str(i).zfill(2)
            setattr(self, name, cell)
            cells.append(getattr(self, name))

        self.cells = cells

    def forward(self, x, hidden=None):
        if not hidden:
            hidden = [None]*int(self.n_layers/2)

        input_ = x

        upd_hidden = []

        for i in range(self.n_layers):
            if i % 2 == 0:
                cell = self.cells[i]
                input_ = cell(input_)
            else:
                cell = self.cells[i]
                cell_hidden = hidden[int(i/2)]

                # pass through layer
                upd_cell_hidden = cell(input_, cell_hidden)
                upd_hidden.append(upd_cell_hidden)
                # update input_ to the last updated hidden layer for next pass
                input_ = upd_cell_hidden

        # retain tensors in list to allow different hidden sizes
        return upd_hidden


class DeconvGRU(nn.Module):
    def __init__(self, channel_input, channel_upsample, channel_crnn,
                kernel_upsample, kernel_crnn, stride_upsample, stride_crnn,
                padding_upsample, padding_crnn, n_layers, channel_output=1,
                output_kernel=1, output_stride = 1, output_padding=0, n_output_layers=1):
        '''
        Generates a multi-layer convolutional GRU.
        Preserves spatial dimensions across cells, only altering depth.
        Parameters
        ----------
        channel_input : integer. depth dimension of input tensors.
        channel_upsample : integer or list. depth dimensions of upsample.
        channel_crnn : integer or list. depth dimensions of crnn.
        stride_upsample: integer or list. the stride size of each upsample layers.
        stride_crnn: integer or list. the stride size of each crnn layers.
        padding_upsample: integer or list. the padding size of each upsample layers.
        padding_crnn: integer or list. the padding size of each crnn layers.
        n_layers : integer. number of chained `DeconvGRUCell`.
        '''
        super().__init__()

        self.channel_input = channel_input
        # channel size
        if type(channel_upsample) != list:
            self.channel_upsample = [channel_upsample]*int(n_layers/2)
        else:
            assert len(channel_upsample) == int(n_layers/2), '`channel_upsample` must have the same length as n_layers/2'
            self.channel_upsample = channel_upsample

        if type(channel_crnn) != list:
            self.channel_crnn = [channel_crnn]*int(n_layers/2)
        else:
            assert len(channel_crnn) == int(n_layers/2), '`channel_crnn` must have the same length as n_layers/2'
            self.channel_crnn = channel_crnn

        # kernel size
        if type(kernel_upsample) != list:
            self.kernel_upsample = [kernel_upsample]*int(n_layers/2)
        else:
            assert len(kernel_upsample) == int(n_layers/2), '`kernel_upsample` must have the same length as n_layers/2'
            self.kernel_upsample = kernel_upsample

        if type(kernel_crnn) != list:
            self.kernel_crnn = [kernel_crnn]*int(n_layers/2)
        else:
            assert len(kernel_crnn) == int(n_layers/2), '`kernel_crnn` must have the same length as n_layers/2'
            self.kernel_crnn = kernel_crnn

       # stride size
        if type(stride_upsample) != list:
            self.stride_upsample = [stride_upsample]*int(n_layers/2)
        else:
            assert len(stride_upsample) == int(n_layers/2), '`stride_upsample` must have the same length as n_layers/2'
            self.stride_upsample = stride_upsample

        if type(stride_crnn) != list:
            self.stride_crnn = [stride_crnn]*int(n_layers/2)
        else:
            assert len(stride_crnn) == int(n_layers/2), '`stride_crnn` must have the same length as n_layers/2'
            self.stride_crnn = stride_crnn

        # padding size
        if type(padding_upsample) != list:
            self.padding_upsample = [padding_upsample]*int(n_layers/2)
        else:
            assert len(padding_upsample) == int(n_layers/2), '`padding_upsample` must have the same length as n_layers/2'
            self.padding_upsample = padding_upsample

        if type(padding_crnn) != list:
            self.padding_crnn = [padding_crnn]*int(n_layers/2)
        else:
            assert len(padding_crnn) == int(n_layers/2), '`padding_crnn` must have the same length as n_layers/2'
            self.padding_crnn = padding_crnn

        # output size
        if type(channel_output) != list:
            self.channel_output = [channel_output]*int(n_output_layers)
        else:
            assert len(channel_output) == int(n_output_layers), '`channel_output` must have the same length as n_output_layers'
            self.channel_output = channel_output

        if type(output_kernel) != list:
            self.output_kernel = [output_kernel]*int(n_output_layers)
        else:
            assert len(output_kernel) == int(n_output_layers), '`output_kernel` must have the same length as n_output_layers'
            self.output_kernel = output_kernel

        if type(output_padding) != list:
            self.output_padding = [output_padding]*int(n_output_layers)
        else:
            assert len(output_padding) == int(n_output_layers), '`output_padding` must have the same length as n_output_layers'
            self.output_padding = output_padding

        if type(output_stride) != list:
            self.output_stride = [output_stride]*int(n_output_layers)
        else:
            assert len(output_stride) == int(n_output_layers), '`output_stride` must have the same length as n_output_layers'
            self.output_stride = output_stride

        self.n_output_layers = n_output_layers
        self.n_layers = n_layers

        cells = []
        for i in range(int(self.n_layers/2)):
            if i == 0:
                cell = DeconvGRUCell(self.channel_input, self.channel_crnn[i], self.kernel_crnn[i], self.stride_crnn[i], self.padding_crnn[i])
            else:
                cell = DeconvGRUCell(self.channel_upsample[i-1], self.channel_crnn[i], self.kernel_crnn[i], self.stride_crnn[i], self.padding_crnn[i])

            name = 'DeconvGRUCell_' + str(i).zfill(2)
            setattr(self, name, cell)
            cells.append(getattr(self, name))

            cell = nn.ConvTranspose2d(self.channel_crnn[i], self.channel_upsample[i], self.kernel_upsample[i], self.stride_upsample[i], self.padding_upsample[i])
            name = 'Upsample_' + str(i).zfill(2)
            setattr(self, name, cell)
            cells.append(getattr(self, name))

        for i in range(self.n_output_layers):
            if i == 0:
                cell = nn.Conv2d(self.channel_upsample[-1], self.channel_output[i], self.output_kernel[i], self.output_stride[i], self.output_padding[i])
            else:
                cell = nn.Conv2d(self.channel_output[i-1], self.channel_output[i], self.output_kernel[i], self.output_stride[i], self.output_padding[i])
        name = 'OutputLayer_' + str(i).zfill(2)
        setattr(self, name, cell)
        cells.append(getattr(self, name))
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
        output = 0

        for i in range(self.n_layers):
            if i % 2 == 0:
                cell = self.cells[i]
                cell_hidden = hidden[int(i/2)]
                # pass through layer
                upd_cell_hidden = cell(input_, cell_hidden)
                upd_hidden.append(upd_cell_hidden)
                # update input_ to the last updated hidden layer for next pass
                input_ = upd_cell_hidden
            else:
                cell = self.cells[i]
                input_ = cell(input_)
        cell = self.cells[-1]
        output = cell(input_)

        # retain tensors in list to allow different hidden sizes
        return upd_hidden, output


class model(nn.Module):
    def __init__(self, n_encoders, n_decoders,
                    encoder_input, encoder_downsample_layer, encoder_crnn_layer, encoder_downsample_k, encoder_crnn_k,
                    encoder_downsample_s, encoder_crnn_s, encoder_downsample_p, encoder_crnn_p, encoder_n_layers,
                    decoder_input, decoder_upsample_layer, decoder_crnn_layer, decoder_upsample_k, decoder_crnn_k,
                    decoder_upsample_s, decoder_crnn_s, decoder_upsample_p, decoder_crnn_p, decoder_n_layers,
                    decoder_output=1, decoder_output_k=1, decoder_output_s=1, decoder_output_p=0, decoder_output_layers=1,
                    batch_norm=False):
        super().__init__()
        self.n_encoders = n_encoders
        self.n_decoders = n_decoders

        models = []
        for i in range(self.n_encoders):
            model = ConvGRU(channel_input=encoder_input, channel_downsample=encoder_downsample_layer, channel_crnn=encoder_crnn_layer,
                            kernel_downsample=encoder_downsample_k, kernel_crnn=encoder_crnn_k,
                            stride_downsample=encoder_downsample_s, stride_crnn=encoder_crnn_s,
                            padding_downsample=encoder_downsample_p, padding_crnn=encoder_crnn_p, n_layers=encoder_n_layers)
            name = 'Encoder_' + str(i+1).zfill(2)
            setattr(self, name, model)
            models.append(getattr(self, name))

        for i in range(self.n_decoders):
            model = DeconvGRU(channel_input=decoder_input, channel_upsample=decoder_upsample_layer, channel_crnn=decoder_crnn_layer,
                        kernel_upsample=decoder_upsample_k, kernel_crnn=decoder_crnn_k,
                        stride_upsample=decoder_upsample_s, stride_crnn=decoder_crnn_s,
                        padding_upsample=decoder_upsample_p, padding_crnn=decoder_crnn_p,
                        n_layers=decoder_n_layers, channel_output=decoder_output, output_kernel=decoder_output_kernel,
                        output_stride=decoder_output_stride, output_padding=decoder_output_padding, n_output_layers=decoder_output_layers)
            name = 'Decoder_' + str(i+1).zfill(2)
            setattr(self, name, model)
            models.append(getattr(self, name))

        self.models = models

    def forward(self, x):
        if x.size()[1] != self.n_encoders:
            assert x.size()[1] == self.n_encoders, '`x` must have the same as n_encoders'

        forecast = []

        for i in range(self.n_encoders):
            if i == 0:
                hidden=None
            model = self.models[i]
            hidden = model(x = x[:,i,:,:,:], hidden=hidden)

        hidden = hidden[::-1]

        for i in range(self.n_encoders,self.n_encoders+self.n_decoders):
            model = self.models[i]
            hidden, output = model(hidden=hidden)
            forecast.append(output)
        forecast = torch.cat(forecast, dim=1)

        return forecast
