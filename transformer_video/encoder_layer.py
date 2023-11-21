import torch.nn as nn
from multi_head_attention import MultiHeadAttention
from utility import feed_forward_network


class EncoderLayer(nn.Module):
    
    def __init__(self, d_model, num_heads, dff, filter_size):
        super(EncoderLayer, self).__init__()
        
        self.mha = MultiHeadAttention(d_model, num_heads, filter_size)
        self.ffn = feed_forward_network(dff, d_model, filter_size)
        
        # self.layernorm1 = nn.BatchNorm2d(d_model, eps=1e-6)
        # self.layernorm2 = nn.BatchNorm2d(d_model, eps=1e-6)

        # Use LayerNorm instead of BatchNorm2d
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        
        # No dropouts for now
                
        
    def forward(self, x, mask):
        print("x.shape before self.mha:", x.shape)
        attn_output, _ = self.mha(x, x, x, mask)
        # (batch_size, input_seq_len, rows, cols, d_model)
        out1 = self.layernorm1(x + attn_output)
        # (batch_size, input_seq_len, rows, cols, d_model)
        print("out1.shape before self.ffn:", out1.shape)
        ffn_output = self.ffn(out1) # (batch_size, input_seq_len, rows, cols, d_model)
        out2 = self.layernorm2(out1 + ffn_output)
        print("out2.shape:", out2.shape)
        return out2

