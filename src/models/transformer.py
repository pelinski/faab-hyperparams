import torch
import math


class PositionalEncoding(torch.nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/comp_feat_len))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/comp_feat_len))
        \text{where pos is the word position and i is the embed idx)
    Args:
        comp_feat_len: the embed dim (required).
        dropout: the dropout value (default=0.1).
        seq_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(comp_feat_len)
    """

    def __init__(self, comp_feat_len, seq_len, dropout=0.1, scale_factor=1.0):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        self.scale_factor = scale_factor

        # shape (seq_len, comp_feat_len)
        pe = torch.zeros(seq_len, comp_feat_len)
        position = torch.arange(
            0, seq_len, dtype=torch.float)  # Shape (seq_len)
        position = position.unsqueeze(1)  # Shape (seq_len, 1)
        div_term = torch.exp(torch.arange(0, comp_feat_len, 2).float(
        ) * (-math.log(10000.0) / comp_feat_len))  # Shape (comp_feat_len/2)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Insert a new dimension for batch size
        pe = pe.unsqueeze(0)  # Shape (1, seq_len, comp_feat_len)
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
    def __init__(self, comp_feat_len, num_heads, ff_size, dropout, num_layers, d_model, mask=False):
        super(Encoder, self).__init__()
        norm_encoder = torch.nn.LayerNorm(comp_feat_len)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            comp_feat_len, num_heads, ff_size, dropout, activation="gelu", batch_first=True)
        self.Encoder = torch.nn.TransformerEncoder(
            encoder_layer, num_layers, norm_encoder)
        self.square_subsequent_mask = self.generate_square_subsequent_mask(
            d_model) if mask else None

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float(
            '-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src):
        # N x d_model x comp_feat_len
        out = self.Encoder(src, mask=self.square_subsequent_mask)
        return out


class InputLayer(torch.nn.Module):
    def __init__(self, feat_len, seq_len, comp_feat_len, d_model,  dropout, pe_scale_factor):
        super(InputLayer, self).__init__()
        self.PositionalEncoding = PositionalEncoding(
            feat_len, seq_len, dropout, pe_scale_factor)
        self.LinearFeatures = torch.nn.Linear(
            feat_len, comp_feat_len, bias=True)

        if seq_len != d_model:
            self.time_compression = True
            self.LinearTime = torch.nn.Linear(seq_len, d_model, bias=True)
        else:
            self.time_compression = False
            self.LinearTime = None

    def init_weights(self, initrange=0.1):
        self.LinearFeatures.bias.data.zero_()
        self.LinearFeatures.weight.data.uniform_(-initrange, initrange)
        if self.time_compression:
            self.LinearTime.bias.data.zero_()
            self.LinearTime.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        x = self.PositionalEncoding(src)  # N x seq_len x feat_len
        x = self.LinearFeatures(x)  # N x seq_len x comp_feat_len
        if self.time_compression:
            x = x.permute(0, 2, 1)  # N x comp_feat_len x seq_len
            x = self.LinearTime(x)  # N x comp_feat_ln x d_model
            x = x.permute(0, 2, 1)  # N x d_model x comp_feat_len
        return x


class OutputLayer(torch.nn.Module):
    def __init__(self, out_feat_len, out_seq_len, comp_feat_len, d_model):
        super(OutputLayer, self).__init__()
        self.LinearFeatures = torch.nn.Linear(
            comp_feat_len, out_feat_len, bias=True)
        if d_model != out_seq_len:
            self.time_compression = True
            self.LinearTime = torch.nn.Linear(d_model, out_seq_len, bias=True)
        else:
            self.time_compression = False
            self.LinearTime = None

    def init_weights(self, initrange=0.1):
        self.LinearFeatures.bias.data.zero_()
        self.LinearFeatures.weight.data.uniform_(-initrange, initrange)
        if self.time_compression:
            self.LinearTime.bias.data.zero_()
            self.LinearTime.weight.data.uniform_(-initrange, initrange)

    def forward(self, decoder_out):
        y = self.LinearFeatures(decoder_out)  # N x seq_len x out_feat_len
        if self.time_compression:
            y = y.permute(0, 2, 1)  # N x out_feat_len x seq_len
            y = self.LinearTime(y)  # N x out_feat_len x out_seq_len
            y = y.permute(0, 2, 1)  # N x out_seq_len x out_feat_len
        return y


class TransformerAutoencoder(torch.nn.Module):
    def __init__(self, **kwargs):
        super(TransformerAutoencoder, self).__init__()

        self.id = kwargs.get("id", "")
        self.seq_len = kwargs.get("seq_len", 512)
        self.d_model = kwargs.get("d_model", 32)
        self.feat_len = kwargs.get("feat_len", 8)
        self.comp_feat_len = kwargs.get("comp_feat_len", 4)
        self.num_heads = kwargs.get("num_heads", 1)
        self.ff_size = kwargs.get("ff_size", 16)
        self.dropout = kwargs.get("dropout", 0.1)
        self.num_layers = kwargs.get("num_layers", 3)
        self.pe_scale_factor = kwargs.get("pe_scale_factor", 1.0)
        self.mask = kwargs.get("mask", False)

        self.InputLayerEncoder = InputLayer(
            self.feat_len, self.seq_len, self.comp_feat_len, self.d_model, self.dropout, self.pe_scale_factor)
        self.Encoder = Encoder(
            self.comp_feat_len, self.num_heads, self.ff_size, self.dropout, self.num_layers, self.d_model, self.mask)
        self.OutputLayer = OutputLayer(
            self.feat_len, self.seq_len, self.comp_feat_len, self.d_model)

        self.InputLayerEncoder.init_weights()
        self.OutputLayer.init_weights()

    def forward(self, src):
        # src N x seq_len x feat_len_src
        x = self.InputLayerEncoder(src)  # N x d_model x comp_feat_len
        memory = self.Encoder(x)  # N x d_model x comp_feat_len
        return self.OutputLayer(memory)  # N x out_seq_len x out_feat_len

    def forward_encoder(self, src):
        x = self.InputLayerEncoder(src)  # N x d_model x comp_feat_len
        memory = self.Encoder(x)  # N x d_model x comp_feat_len
        return memory


if __name__ == "__main__":
    batch_size = 5
    feat_len = 8
    seq_len = 512
    comp_feat_len = 4
    d_model = 32
    num_heads = 1
    ff_size = 16
    dropout = 0.1
    num_layers = 3
    pe_scale_factor = 1.0

    print("test input layer")
    x = torch.randn(batch_size, seq_len, feat_len)
    input_layer = InputLayer(
        feat_len, seq_len, comp_feat_len, d_model, dropout, pe_scale_factor)
    output = input_layer(x)
    print("input shape", x.shape)
    print("output shape", output.shape)

    assert output.shape == (batch_size, d_model, comp_feat_len)

    print("test output layer")
    x = torch.randn(batch_size, d_model, comp_feat_len)
    output_layer = OutputLayer(feat_len, seq_len, comp_feat_len, d_model)
    output = output_layer(x)
    print("input shape", x.shape)
    print("output shape", output.shape)
    assert output.shape == (batch_size, seq_len, feat_len)

    print("test model")
    x = torch.randn(batch_size, seq_len, feat_len)
    model = TransformerAutoencoder(
        d_model=d_model,
        comp_feat_len=comp_feat_len, feat_len=feat_len, num_heads=num_heads, ff_size=ff_size, dropout=dropout, num_layers=num_layers, seq_len=seq_len, pe_scale_factor=pe_scale_factor)
    output = model(x)
    memory = model.Encoder(model.InputLayerEncoder(x))
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    print("Memory shape:", memory.shape)

    assert output.shape == (batch_size, seq_len, feat_len)
    assert memory.shape == (batch_size, d_model, comp_feat_len)
