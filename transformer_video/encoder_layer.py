import torch.nn as nn

from .multi_head_attention import MultiHeadAttention
from .utility import feed_forward_network


class EncoderLayer(nn.Module):
    
    def __init__(self, d_model, num_heads, dff, filter_size):
        super(EncoderLayer, self).__init__()
        
        self.mha = MultiHeadAttention(d_model, num_heads, filter_size)
        self.ffn = feed_forward_network(dff, d_model, filter_size)
        
        self.layernorm1 = nn.BatchNorm2d(d_model, eps=1e-6)
        self.layernorm2 = nn.BatchNorm2d(d_model, eps=1e-6)
        
        # No dropouts for now
                
        
    def call(self, x, training, mask):
        
        attn_output, _ = self.mha(x, x, x, mask)
        # (batch_size, input_seq_len, rows, cols, d_model)
        out1 = self.layernorm1(x + attn_output, training=training)
        # (batch_size, input_seq_len, rows, cols, d_model)
        
        ffn_output = self.ffn(out1) # (batch_size, input_seq_len, rows, cols, d_model)
        out2 = self.layernorm2(out1 + ffn_output, training=training)
        
        return out2

