# import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
from decoder import Decoder
from encoder import Encoder


class Transformer(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, filter_size,
                 image_shape, pe_input, pe_target, out_channel):
        super(Transformer, self).__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, filter_size, image_shape, pe_input)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, filter_size, image_shape, pe_target)
        self.final_layer = nn.Conv2d(d_model, out_channel, filter_size, padding='same')

    def forward(self, inp, tar, look_ahead_mask):
        # enc_output.shape = (batch_size, inp_seq_len, rows, cols, d_model)
        print("Transformer.forward input", inp.shape)
        # maybe get rid of input row col
        enc_output = self.encoder(inp, None)

        # dec_output.shape = (batch_size, tar_seq_len, rows, cols, d_model)
        dec_output, attention_weights = self.decoder(tar, enc_output, look_ahead_mask)
        
        # final_output.shape = (batch_size, tar_seq_len, rows, cols, out_channel)
        final_output = self.final_layer(dec_output)

        final_output = torch.sigmoid(final_output)

        return final_output, attention_weights


# =================================

# class Transformer(tf.keras.Model):
#     def __init__(self, num_layers, d_model, num_heads, dff, filter_size,
#                  image_shape, pe_input, pe_target, out_channel):
#         super(Transformer, self).__init__()

#         self.encoder = Encoder(num_layers, d_model, num_heads, dff, filter_size,
#                                image_shape, pe_input)
#         self.decoder = Decoder(num_layers, d_model, num_heads, dff, filter_size,
#                                image_shape, pe_target)

#         self.final_layer = tf.keras.layers.Conv2D(out_channel, filter_size,
#                                                   padding='same', activation='sigmoid')

#     def call(self, inp, tar, training, look_ahead_mask):

#         # enc_output.shape = (batch_size, inp_seq_len, rows, cols, d_model)
#         enc_output = self.encoder(inp, training, None)

#         # dec_output.shape = (batch_size, tar_seq_len, rows, cols, d_model)
#         dec_output, attention_weights = self.decoder(tar, enc_output,
#                                                      training, look_ahead_mask)
        
#         # final_output.shape = (batch_size, tar_seq_len, rows, cols, out_channel)
#         final_output = self.final_layer(dec_output)

#         return final_output, attention_weights

