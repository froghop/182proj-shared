import torch
import torch.nn as nn

from .multi_head_attention import MultiHeadAttention
from .utility import feed_forward_network


class DecoderLayer(nn.Module):
    
    def __init__(self, d_model, num_heads, dff, filter_size):
        super(DecoderLayer, self).__init__()
        
        self.mha1 = MultiHeadAttention(d_model, num_heads, filter_size)
        self.mha2 = MultiHeadAttention(d_model, num_heads, filter_size)

        self.ffn = feed_forward_network(dff, d_model, filter_size)
        
        self.layernorm1 = nn.BatchNorm2d(d_model, eps=1e-6)
        self.layernorm2 = nn.BatchNorm2d(d_model, eps=1e-6)
        self.layernorm3 = nn.BatchNorm2d(d_model, eps=1e-6)
        
        # No dropouts for now
                
        
    def forward(self, x, enc_output, training, look_ahead_mask):

        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        out1 = self.layernorm1(x + attn1)
        
        attn2, attn_weights_block2 = self.mha2(out1, enc_output, enc_output, None)
        out2 = self.layernorm2(out1 + attn2)

        ffn_output = self.ffn(out2)
        out3 = self.layernorm3(out2 + ffn_output)
        
        return out3, attn_weights_block1, attn_weights_block2
