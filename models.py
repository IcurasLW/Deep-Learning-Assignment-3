import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer, TransformerEncoder, TransformerDecoder, TransformerDecoderLayer
import numpy as np
import pandas as pd

torch.set_default_dtype(torch.float32)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'



class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers, args):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers
        self.window_size = args.window_size
        
        self.rnn = nn.RNN(input_size, hidden_size, num_stacked_layers, 
                            batch_first=True, nonlinearity='relu')
        self.fc = nn.Linear(hidden_size, 3)


    def forward(self, x):
        
        x = x.to(DEVICE).to(torch.float32).squeeze(0)
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(DEVICE)
        
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out



class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers, args):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers
        self.window_size = args.window_size
        self.gru = nn.GRU(input_size, hidden_size, num_stacked_layers, 
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, 3)
        
    def forward(self, x):
        x = x.to(DEVICE).to(torch.float32).squeeze(0)
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out



class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers, args):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers
        self.window_size = args.window_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers, 
                            batch_first=True)
        
        self.fc = nn.Linear(hidden_size, 3)

    def forward(self, x):
        x = x.to(DEVICE).to(torch.float32)
        x = x.squeeze(0)
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(DEVICE)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(DEVICE)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


class Transformer(nn.Module):
    def __init__(self, args, input_size, hidden_size, num_layers, num_heads, output_size):
        # feautre_size equals to embedding dimension (d_model)
        super().__init__()

        self.hidden_size = hidden_size
        self.embedding = nn.Linear(input_size, hidden_size)
        self.encoder_layer = TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, batch_first=True)
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers=num_layers)       
        self.decoder = nn.Linear(hidden_size, output_size)
        self.window_size = args.window_size
        
    def _generate_positional_encoding(self, max_len):
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, self.hidden_size, 2).float() * -(torch.log(torch.tensor(10000.0)) / self.hidden_size))
        pos_enc = torch.zeros((max_len, 1, self.hidden_size))
        pos_enc[:, 0, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 0, 1::2] = torch.cos(position * div_term)
        return pos_enc.to(DEVICE).to(torch.float32)
    
    def forward(self, src):
        # src size: (batch, sequence, feature)
        src = src.squeeze(0)
        len_of_src = src.size()[1]
        positional_encoding = self._generate_positional_encoding(len_of_src).to(DEVICE)
        # src_mask = nn.Transformer.generate_square_subsequent_mask(len_of_src).to(DEVICE)
        src = self.embedding(src)
        src = src.permute(1, 0, 2)
        src = src + positional_encoding
        src = src.permute(1, 0, 2)
        output = self.transformer_encoder(src)
        output = output[:, -1, :]
        output = self.decoder(output)
        return output