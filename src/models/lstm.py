import torch
import torch.nn as nn


# class LSTMCell(nn.Module):
#     def __init__(self, feat_len, hidden_size):
#         super(LSTMCell, self).__init__()
#         self.feat_len = feat_len
#         self.hidden_size = hidden_size

#         # matrices containing weights for input, hidden and bias for each of the 4 gates
#         self.W = nn.Parameter(torch.Tensor(
#             self.feat_len, self.hidden_size*4))
#         self.U = nn.Parameter(torch.Tensor(
#             self.hidden_size, self.hidden_size*4))
#         self.bias = nn.Parameter(torch.Tensor(self.hidden_size*4))

#         self.init_weights()

#     def init_weights(self):
#         nn.init.xavier_uniform_(self.W)
#         nn.init.xavier_uniform_(self.U)
#         nn.init.zeros_(self.bias)

#     def forward(self, x_t, h_t, c_t):
#         # x_t (batch_size, feat_len)
#         # batch the computations into a single matrix multiplication
#         # @ is for matrix multiplication
#         gates = x_t@self.W + h_t@self.U + self.bias
#         hsz = self.hidden_size
#         # input gate (batch_size, hidden_size)
#         i_t = torch.sigmoid(gates[:, :hsz])
#         # forget gate (batch_size, hidden_size)
#         f_t = torch.sigmoid(gates[:, hsz:hsz*2])
#         # candidate gate (batch_size, hidden_size)
#         g_t = torch.Sigmoid(gates[:, hsz*2:hsz*3])
#         # output gate (batch_size, hidden_size)
#         o_t = torch.sigmoid(gates[:, hsz*3:])

#         c_t = f_t * c_t + i_t * g_t  # (batch_size, hidden_size)
#         h_t = o_t * torch.Sigmoid(c_t)  # (batch_size, hidden_size)

#         return h_t, c_t


class LSTM(nn.Module):  # short version using matrices
    # https://towardsdatascience.com/building-a-lstm-by-hand-on-pytorch-59c02a4ec091
    def __init__(self, **kwargs):
        """LSTM

        Source: https://towardsdatascience.com/building-a-lstm-by-hand-on-pytorch-59c02a4ec091

        Args:
            seq_len (int): Sequence length
            feat_out_size (int): Output sequence length
            feat_len (int): Input size of the network
            hidden_size (int): Size of the hidden state
            ff_size_features (int): Size of the feed forward layer
            dropout (float): Dropout rate
        """
        super().__init__()
        self.seq_len = kwargs.get("seq_len", 10)
        self.feat_len = kwargs.get("feat_len", 8)
        self.feat_out_size = kwargs.get("feat_out_size", 8)
        self.ff_size_features = kwargs.get("ff_size_features", 16)
        self.hidden_size = kwargs.get("hidden_size", 4)
        self.num_layers = kwargs.get("num_layers", 1)
        self.dropout_p = kwargs.get("dropout", 0.2)
        self.proj_size = kwargs.get("proj_size", 4)

        self.LSTMLayers = nn.LSTM(
            self.feat_len, self.hidden_size, num_layers=self.num_layers, batch_first=True, dropout=self.dropout_p, proj_size=self.proj_size)

        if self.proj_size > 0:
            lstm_out_size = self.proj_size
        else:
            lstm_out_size = self.hidden_size
        self.OutputLayer = nn.Sequential(
            nn.Linear(lstm_out_size, self.feat_out_size), nn.Softmax(dim=2))

        self.dropout = nn.Dropout(self.dropout_p)

        self.init_weights()

    def init_weights(self):
        for layer in self.OutputLayer:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(
                    layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x, return_states=False):
        """LSTM forward pass

        Args:
            x (torch.Tensor): Input torch tensor of shape (batch_size, seq_len, feat_len)
            init_states (torch.Tensor, optional): Initial states for output of the network (h_t) and the long-term memory (c_t). Defaults to None.
            return_states (bool, optional): Returns hidden_state, (h_t, c_t) if set to True, otherwise returns only hidden_state . Defaults to False.

        Returns:
            if return_states is True:
                hidden_state (torch.Tensor), (h_t (torch.Tensor), c_t (torch.Tensor)): Hidden state, network output and long-term memory
            if return_states is False:
                hidden_state (torch.Tensor): Hidden state
        """
        (batch_size, _, _) = x.shape
        # hidden_seq = []

        h_t, c_t = (torch.zeros(batch_size, self.hidden_size).to(x.device, non_blocking=True),
                    torch.zeros(batch_size, self.hidden_size).to(x.device, non_blocking=True))
        # (batch_size, hidden_size), (batch_size, hidden_size)

        out_red, (h_t, c_t) = self.LSTMLayers(x)

        # for t in range(seq_size):
        #     x_t = x[:, t, :]  # (batch_size, feat_len)
        #     h_t, c_t = self.LSTMCell(x_t, (h_t, c_t))

        #     # h_t -->(1, batch_size, hidden_size)
        #     hidden_seq.append(h_t.unsqueeze(0))
        #     # hidden_seq is a list of sequence_length items, each of shape (1, batch_size, hidden_size)

        # # reshape hidden_seq
        # # (sequence_length, batch_size, hidden_size)
        # hidden_seq = torch.cat(hidden_seq, dim=0)
        # hidden_seq = hidden_seq.transpose(0,
        #                                   1).contiguous()  # (batch_size, sequence_length, hidden_size). contiguous returns a tensor contiguous in memory

        # (sequence_length, batch_size, hidden_size)
        out = self.OutputLayer(out_red)

        if return_states:
            return out, out_red, (h_t, c_t)
        else:
            return out, out_red


if __name__ == "__main__":
    batch_size = 32
    seq_len = 10
    feat_len = 8
    feat_out_size = 30
    hidden_size = 6
    x = torch.rand(batch_size, seq_len, feat_len)
    model = LSTM(feat_len=feat_len,
                 hidden_size=hidden_size, seq_len=seq_len, feat_out_size=feat_out_size)
    y, w = model.forward(x)
    print(x.shape)
    print(y.shape)
    print(w.shape)
