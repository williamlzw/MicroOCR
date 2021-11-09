import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.functional import adaptive_avg_pool1d


class ConvBNACT(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              groups=groups,
                              bias=True)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class MicroBlock(nn.Module):
    def __init__(self, nh, kernel_size):
        super().__init__()
        self.conv1 = ConvBNACT(nh, nh, kernel_size, groups=nh, padding=1)
        self.conv2 = ConvBNACT(nh, nh, 1)

    def forward(self, x):
        x = x + self.conv1(x)
        x = self.conv2(x)
        return x


class Conv_1D_LSTM_cell(nn.Module):
    """Initialize a basic Conv LSTM cell.
    Args:
      shape: int, length of the hidden states h and c()
      filter_size: int that is the height and width of the filters
      num_features: int thats the num of channels of the states, like hidden_size

    """

    def __init__(self, input_chans, filter_size, num_features):
        super(Conv_1D_LSTM_cell, self).__init__()
        self.input_chans = input_chans
        self.filter_size = filter_size
        self.num_features = num_features
        # in this way the output has the same size
        self.padding = int((filter_size-1)/2)

        self.conv = nn.Conv1d(self.input_chans + self.num_features,
                              4*self.num_features, self.filter_size, 1, self.padding)

    def forward(self, input, hidden_state):
        hidden, c = hidden_state  # hidden and c are images with several channels
        #hidden = hidden.to(input.device)
        #c = c.to(input.device)
        combined = torch.cat((input, hidden), 1)  # concatenate in the channels
        A = self.conv(combined)
        (ai, af, ao, ag) = torch.split(A, self.num_features, dim=1)
        i = torch.sigmoid(ai)
        f = torch.sigmoid(af)
        o = torch.sigmoid(ao)
        g = torch.tanh(ag)

        next_c = f*c+i*g
        next_h = o*torch.tanh(next_c)
        return next_h, next_c

    def init_hidden(self, batch_size, shape):
        return (Variable(torch.zeros(batch_size, self.num_features, shape)).cuda(), Variable(torch.zeros(batch_size, self.num_features, shape)).cuda())


class ConvLSTM(nn.Module):

    """Initialize a basic Conv LSTM cell.
    Args:
      shape: int thats the length of the hidden states h and c()
      filter_size: int that is the height and width of the filters
      num_features: int thats the num of channels of the states, like hidden_size

    """

    def __init__(self, input_chans, filter_sizes, num_features, num_layers):
        super(ConvLSTM, self).__init__()
        self.input_chans = input_chans

        if not isinstance(filter_sizes, list):
            filter_sizes = [filter_sizes for _ in range(num_layers)]
        assert len(filter_sizes) == num_layers
        self.filter_sizes = filter_sizes

        if not isinstance(num_features, list):
            num_features = [num_features for _ in range(num_layers)]
        assert len(num_features) == num_layers
        self.num_features = num_features

        self.num_layers = num_layers
        cell_list = []
        cell_list.append(Conv_1D_LSTM_cell(self.input_chans,
                                           self.filter_sizes[0],
                                           self.num_features[0]).cuda())  # the first
        # one has a different number of input channels

        for idcell in range(1, self.num_layers):
            cell_list.append(Conv_1D_LSTM_cell(self.num_features[idcell-1],
                                               self.filter_sizes[idcell],
                                               self.num_features[idcell]).cuda())
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input, hidden_state):
        """
        args:
            hidden_state:list of tuples, one for every layer, each tuple should be hidden_layer_i,c_layer_i
            input is the tensor of shape Batch,seq_len,Channels,length
        """

        current_input = input.transpose(0, 1)  # now is seq_len,B,C,length
        # current_input=input
        next_hidden = []  # hidden states(h and c)
        seq_len = current_input.size(0)

        for idlayer in range(self.num_layers):  # loop for every layer

            # hidden and c are images with several channels
            hidden_c = hidden_state[idlayer]
            output_inner = []
            for t in range(seq_len):  # loop for every step
                # cell_list is a list with different conv_lstms 1 for every layer
                hidden_c = self.cell_list[idlayer](
                    current_input[t, ...], hidden_c)

                output_inner.append(hidden_c[0])

            next_hidden.append(hidden_c)
            current_input = torch.cat(output_inner, 0).view(current_input.size(0),
                                                            *output_inner[0].size())  # seq_len,B,chans,length

        return current_input, next_hidden

    def init_hidden(self, batch_size, shape):
        init_states = []  # this is a list of tuples
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, shape))
        return init_states

class MicroNet(nn.Module):
    def __init__(self, nh=64, depth=2, nclass=60, img_height=32, use_lstm=True):
        super().__init__()
        assert(nh >= 2)
        self.conv = ConvBNACT(3, nh, 4, 4)
        self.blocks = nn.ModuleList()

        for i in range(depth):
            self.blocks.append(MicroBlock(nh, 3))
        self.use_lstm = use_lstm
        self.conv_lstm = ConvLSTM(8, 1, 8, 1)
        self.flatten = nn.Flatten(start_dim=1, end_dim=2)
        self.dropout = nn.Dropout(0.1)
        linear_in = nh * int((img_height-(4-1)-1)/4 + 1)
        self.fc = nn.Linear(linear_in, nclass)
        

    def forward(self, x):
        x_shape = x.size()
        x = self.conv(x)
        for block in self.blocks:
            x = block(x)
        hidden_state = self.conv_lstm.init_hidden(x_shape[0], int(x_shape[3]/4))
        print(x.shape) # bs, nh, in, width
        x = self.conv_lstm(x, hidden_state)[0]
        print(x.shape)
        x = x.permute(1, 0, 2, 3)
        print(x.shape)
        x = self.flatten(x)
        print(x.shape)
        x = adaptive_avg_pool1d(x, int(x_shape[3]/4))
        print(x.shape)
        x = x.permute(0, 2, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    import time
    x = torch.randn(1, 3, 32, 256).cuda()
    model = MicroNet(32, depth=2, nclass=62, img_height=32, use_lstm=True).cuda()
    t0 = time.time()
    out = model(x)
    t1 = time.time()
    print(out.shape, (t1-t0)*1000)
    #torch.save(model, 'test1.pth')
    # # from torchsummaryX import summary
    # # summary(model, x)
