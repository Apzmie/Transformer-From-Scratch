import math
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.W_Q = nn.Parameter(torch.empty(d_model, d_model))
        self.W_K = nn.Parameter(torch.empty(d_model, d_model))
        self.W_V = nn.Parameter(torch.empty(d_model, d_model))
        self.W_O = nn.Parameter(torch.empty(d_model, d_model))
        self._reset_parameters()

    def _reset_parameters(self):
        for param in [self.W_Q, self.W_K, self.W_V, self.W_O]:
            nn.init.xavier_uniform_(param)

    def forward(self, enc=None, dec=None, causal_mask=None, padding_mask=None):
        if enc is not None and dec is None:
            Q = torch.matmul(enc, self.W_Q)
            K = torch.matmul(enc, self.W_K)
            V = torch.matmul(enc, self.W_V)
        elif dec is not None and enc is None:
            Q = torch.matmul(dec, self.W_Q)
            K = torch.matmul(dec, self.W_K)
            V = torch.matmul(dec, self.W_V)
        elif enc is not None and dec is not None:
            Q = torch.matmul(dec, self.W_Q)
            K = torch.matmul(enc, self.W_K)
            V = torch.matmul(enc, self.W_V)

        b, s, _ = Q.shape
        Q = Q.view(b, s, self.num_heads, self.d_head).transpose(1, 2)
        K = K.view(b, -1, self.num_heads, self.d_head).transpose(1, 2)
        V = V.view(b, -1, self.num_heads, self.d_head).transpose(1, 2)

        QK_matmul = torch.matmul(Q, K.transpose(-2, -1))
        scale = QK_matmul / math.sqrt(self.d_head)

        if causal_mask is not None:
            scale = scale.masked_fill(causal_mask == 0, float('-inf'))
        if padding_mask is not None:
            scale = scale.masked_fill(padding_mask == 0, float('-inf'))

        softmax = torch.softmax(scale, dim=-1)
        SV_matmul = torch.matmul(softmax, V)
        concat = SV_matmul.transpose(1, 2).contiguous().view(b, s, -1)
        linear = torch.matmul(concat, self.W_O)
        return linear


class AddandNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, sublayer_out):
        out = self.norm(x + sublayer_out)
        return out


class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        d_ff = d_model * 4
        self.W1 = nn.Parameter(torch.empty(d_model, d_ff))
        self.b1 = nn.Parameter(torch.zeros(d_ff))
        self.W2 = nn.Parameter(torch.empty(d_ff, d_model))
        self.b2 = nn.Parameter(torch.zeros(d_model))
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W1)
        nn.init.xavier_uniform_(self.W2)

    def forward(self, x):
        linear1 = torch.matmul(x, self.W1) + self.b1
        relu = torch.relu(linear1)
        linear2 = torch.matmul(relu, self.W2) + self.b2
        return linear2


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.multiatt = MultiHeadAttention(d_model, num_heads)
        self.addnorm1 = AddandNorm(d_model)
        self.ff = FeedForward(d_model)
        self.addnorm2 = AddandNorm(d_model)

    def forward(self, encoder_input, padding_mask=None):
        multiatt_out = self.multiatt(enc=encoder_input, padding_mask=padding_mask)
        addnorm1_out = self.addnorm1(encoder_input, multiatt_out)
        ff_out = self.ff(addnorm1_out)
        addnorm2_out = self.addnorm2(addnorm1_out, ff_out)
        return addnorm2_out


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.multiatt1 = MultiHeadAttention(d_model, num_heads)
        self.addnorm1 = AddandNorm(d_model)
        self.multiatt2 = MultiHeadAttention(d_model, num_heads)
        self.addnorm2 = AddandNorm(d_model)
        self.ff = FeedForward(d_model)
        self.addnorm3 = AddandNorm(d_model)

    def forward(self, decoder_input, encoder_output, padding_mask=None, enc_padding_mask=None):
        seq_len = decoder_input.shape[1]
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=decoder_input.device)).unsqueeze(0).unsqueeze(1)
        multiatt1_out = self.multiatt1(dec=decoder_input, causal_mask=causal_mask, padding_mask=padding_mask)
        addnorm1_out = self.addnorm1(decoder_input, multiatt1_out)
        multiatt2_out = self.multiatt2(enc=encoder_output, dec=addnorm1_out, padding_mask=enc_padding_mask)
        addnorm2_out = self.addnorm2(addnorm1_out, multiatt2_out)
        ff_out = self.ff(addnorm2_out)
        addnorm3_out = self.addnorm3(addnorm2_out, ff_out)
        return addnorm3_out


class Transformer(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, vocab_size, enc_max_len, dec_max_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        pe_enc = self.PositionalEncoding(enc_max_len, d_model)
        pe_dec = self.PositionalEncoding(dec_max_len, d_model)
        self.register_buffer("pe_enc", pe_enc)
        self.register_buffer("pe_dec", pe_dec)
        self.encoder = nn.ModuleList([EncoderLayer(d_model, num_heads) for _ in range(num_layers)])
        self.decoder = nn.ModuleList([DecoderLayer(d_model, num_heads) for _ in range(num_layers)])
        self.linear = nn.Linear(d_model, vocab_size)

    def PositionalEncoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        exp_scale = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * exp_scale)
        pe[:, 1::2] = torch.cos(pos * exp_scale)
        pe = pe.unsqueeze(0)
        return pe

    def forward(self, enc_input_tensor, dec_input_tensor, vocab):
        device = enc_input_tensor.device
        pe_enc = self.pe_enc[:, :enc_input_tensor.shape[1], :]
        pe_dec = self.pe_dec[:, :dec_input_tensor.shape[1], :]
        encoder_input = self.embedding(enc_input_tensor) + pe_enc
        decoder_input = self.embedding(dec_input_tensor) + pe_dec
        enc_padding_mask = (enc_input_tensor != vocab["<PAD>"]).unsqueeze(1).unsqueeze(2).to(device)
        dec_padding_mask = (dec_input_tensor != vocab["<PAD>"]).unsqueeze(1).unsqueeze(2).to(device)

        x = encoder_input
        for layer in self.encoder:
            x = layer(x, padding_mask=enc_padding_mask)
        encoder_output = x

        y = decoder_input
        for layer in self.decoder:
            y = layer(y, encoder_output, padding_mask=dec_padding_mask, enc_padding_mask=enc_padding_mask)
        logits = self.linear(y)
        return logits
