import torch
import math


class PositionalEncoding(torch.nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, max_len, dropout=0.1, scale_factor=1.0):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        self.scale_factor = scale_factor

        pe = torch.zeros(max_len, d_model)  # shape (max_len, d_model)
        position = torch.arange(
            0, max_len, dtype=torch.float)  # Shape (max_len)
        position = position.unsqueeze(1)  # Shape (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float(
        ) * (-math.log(10000.0) / d_model))  # Shape (d_model/2)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Insert a new dimension for batch size
        pe = pe.unsqueeze(0)  # Shape (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
   ⁄
        Examples:
            >>> output = pos_encoder(x)
        """
        x = x + self.scale_factor * self.pe

        return self.dropout(x)


class Encoder(torch.nn.Module):
    def __init__(self, d_model, num_heads, ff_size, dropout, num_layers, max_len, mask=False):
        super(Encoder, self).__init__()
        norm_encoder = torch.nn.LayerNorm(d_model)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model, num_heads, ff_size, dropout, activation="gelu", batch_first=True)
        self.Encoder = torch.nn.TransformerEncoder(
            encoder_layer, num_layers, norm_encoder)
        self.square_subsequent_mask = self.generate_square_subsequent_mask(
            max_len) if mask else None

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float(
            '-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src):
        # seq_len x Nxd_model
        out = self.Encoder(src, mask=self.square_subsequent_mask)
        return out


class InputLayer(torch.nn.Module):
    def __init__(self, feat_in_size, d_model, dropout, max_len, pe_scale_factor):
        super(InputLayer, self).__init__()
        self.PositionalEncoding = PositionalEncoding(
            feat_in_size, max_len, dropout, pe_scale_factor)
        self.Linear = torch.nn.Linear(feat_in_size, d_model, bias=True)
        # self.activation = torch.nn.Sigmoid()

    def init_weights(self, initrange=0.1):
        self.Linear.bias.data.zero_()
        self.Linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        x = self.PositionalEncoding(src)
        x = self.Linear(src)
        return x
        # return self.activation(x)


class OutputLayer(torch.nn.Module):
    def __init__(self, output_dim, d_model):
        super(OutputLayer, self).__init__()
        self.Linear = torch.nn.Linear(d_model, output_dim, bias=True)
        # self.activation = torch.nn.Tanh()

    def init_weights(self, initrange=0.1):
        self.Linear.bias.data.zero_()
        self.Linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, decoder_out):
        y = self.Linear(decoder_out)
        return y
        # return self.activation(y)


class TransformerAutoencoder(torch.nn.Module):
    def __init__(self, **kwargs):
        super(TransformerAutoencoder, self).__init__()

        self.id = kwargs.get("id", "")
        self.d_model = kwargs.get("d_model", 4)
        self.feat_in_size = kwargs.get("feat_in_size", 8)
        self.num_heads = kwargs.get("num_heads", 1)
        self.ff_size = kwargs.get("ff_size", 16)
        self.dropout = kwargs.get("dropout", 0.1)
        self.max_len = kwargs.get("max_len", 10)
        self.num_layers = kwargs.get("num_layers", 3)
        self.pe_scale_factor = kwargs.get("pe_scale_factor", 1.0)
        self.mask = kwargs.get("mask", False)

        self.InputLayerEncoder = InputLayer(
            self.feat_in_size, self.d_model, self.dropout, self.max_len, self.pe_scale_factor)
        self.Encoder = Encoder(
            self.d_model, self.num_heads, self.ff_size, self.dropout, self.num_layers, self.max_len, self.mask)
        self.OutputLayer = OutputLayer(self.feat_in_size, self.d_model)

        self.InputLayerEncoder.init_weights()
        self.OutputLayer.init_weights()

    def forward(self, src):
        # src N x seq_len x feat_in_size_src
        x = self.InputLayerEncoder(src)  # N x seq_len x d_model
        memory = self.Encoder(x)  # N x seq_len x d_model
        return self.OutputLayer(memory)

    def forward_encoder(self, src):
        x = self.InputLayerEncoder(src)  # N x seq_len x d_model
        memory = self.Encoder(x)  # N x seq_len x d_model
        return memory


if __name__ == "__main__":
    feat_in_size = 8
    d_model = 4
    num_heads = 1
    ff_size = 16
    dropout = 0.1
    num_layers = 3
    seq_len = 10
    batch_size = 32
    pe_scale_factor = 1.0

    print("test input layer")
    x = torch.randn(batch_size, seq_len, feat_in_size)
    input_layer = InputLayer(
        feat_in_size, d_model, dropout, seq_len, pe_scale_factor)
    output = input_layer(x)
    print("input shape", x.shape)
    print("output shape", output.shape)

    print("test output layer")
    x = torch.randn(batch_size, seq_len, d_model)
    output_layer = OutputLayer(feat_in_size, d_model)
    output = output_layer(x)
    print("input shape", x.shape)
    print("output shape", output.shape)

    print("test model")

    x = torch.randn(batch_size, seq_len, feat_in_size)
    model = TransformerAutoencoder(
        d_model=d_model, feat_in_size=feat_in_size, num_heads=num_heads, ff_size=ff_size, dropout=dropout, num_layers=num_layers, seq_len=seq_len, pe_scale_factor=pe_scale_factor)
    output = model(x)
    memory = model.Encoder(model.InputLayerEncoder(x))
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    print("Memory shape:", memory.shape)
