import torch
import torch.nn as nn
import torch.nn.functional as F
from decoder_layer import \
    DecoderLayer  # Ensure DecoderLayer is translated to PyTorch
from utility import \
    positional_encoding  # Ensure this is compatible with PyTorch


class Decoder(nn.Module):
    
    def __init__(self, num_layers, d_model, num_heads, dff, filter_size,
                 image_shape, max_position_encoding):
        super(Decoder, self).__init__()
        
        self.num_layers = num_layers
        self.d_model = d_model

        # Conv2d in PyTorch doesn't support 'same' padding directly, so we handle it separately
        self.padding = filter_size // 2  # Assuming filter_size is odd
        self.embedding = nn.Conv2d(in_channels=image_shape[-1],  # Assuming image_shape includes channels
                                   out_channels=d_model, 
                                   kernel_size=filter_size, 
                                   padding=self.padding)
        self.pos_encoding = positional_encoding(max_position_encoding, image_shape, d_model)
        
        self.dec_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, dff, filter_size)
                                         for _ in range(num_layers)])
    

    def forward(self, x, enc_output, look_ahead_mask):
        seq_len = x.size(1)
        attention_weights = {}

        x = F.relu(self.embedding(F.pad(x, (self.padding, self.padding, self.padding, self.padding))))
        x = x * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        x = x + self.pos_encoding[:, :seq_len, :, :, :]

        for i, layer in enumerate(self.dec_layers):
            x, block1, block2 = layer(x, enc_output, look_ahead_mask)
            attention_weights[f'decoder_layer{i+1}_block1'] = block1
            attention_weights[f'decoder_layer{i+1}_block2'] = block2

        return x, attention_weights
