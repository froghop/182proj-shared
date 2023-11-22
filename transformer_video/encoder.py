import torch
import torch.nn as nn
from encoder_layer import EncoderLayer
from utility import positional_encoding


class Encoder(nn.Module):
    # (batch_size, inp_seq_len, rows, cols, d_model)
    def __init__(self, num_layers, d_model, num_heads, dff, filter_size,
                 image_shape, max_position_encoding):
        super(Encoder, self).__init__()
        
        self.num_layers = num_layers
        self.d_model = d_model

        # self.embedding = nn.Conv2D(1, d_model, filter_size, padding='same', activation='relu')
        self.embedding = nn.Sequential(
            nn.Conv2d(1, d_model, filter_size, padding='same'),
            nn.ReLU(),
        )
        self.pos_encoding = positional_encoding(max_position_encoding, image_shape, d_model)
        
        self.enc_layers  = [EncoderLayer(d_model, num_heads, dff, filter_size) for _ in range(num_layers)]
    

    def forward(self, x, mask):

        # x.shape = (batch_size, seq_len, rows, cols, depth)
        seq_len = x.shape[1]
        
        # image embedding and position encoding
        # [320, 16, 16, 1]
        #print(x.shape)
        x = self.embedding(x.contiguous().view((x.shape[0]*x.shape[1], *x.shape[2:]))).unsqueeze(dim=0)
        #print(x.shape)
        # [5, 128, 16, 16]
        x *= torch.sqrt(torch.tensor(float(self.d_model), dtype=torch.float32))

        # want [1, 5, 16, 16, 128] [batch size, seq len, im size, im size, d_model]
        print("self.pos_encoding.shape:", self.pos_encoding.shape)
        print("x.shape:", x.shape)
        # Repeat the positional encoding for each sequence in the batch
        pos_encoding = self.pos_encoding.repeat(x.size(0), 1, 1, 1, 1)

        # Slice the positional encoding to match the spatial dimensions of x
        pos_encoding = pos_encoding[:, :seq_len, :, :, :]

        print("pos_encoding.shape", pos_encoding.shape)
        x = x + pos_encoding.permute(0, 1, 4, 2, 3)



        for layer in range(self.num_layers):
            print("Layer: ", layer)
            x = self.enc_layers[layer](x, mask)

        return x # (batch_size, seq_len, rows, cols, d_model)
