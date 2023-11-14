# masha

# import tensorflow as tf
import torch
import torch.nn as nn

INFINITY = 1e9

def scaled_dot_product_attention(q, k, v, mask=None):
    """
    Calculate the attention weights

    Args:
    q : query shape = (batch_sz, heads, seq_len_q, rows, cols, depth)
    k : keys shape = (batch_sz, heads, seq_len_k, rows, cols, depth)
    v : values shape = (batch_sz, heads, seq_len_v, rows, cols, depth)
    mask : shape = (batch_sz, heads, seq_len, rows, cols, depth)
    
    Returns:
    outputs, attention_weights
    """

    seq_len = q.shape[2]
    dim_k = float(k.size(-1))

    attention_weights = []
    outputs = []
    for seq in range(seq_len):

        query = q[:, :, seq, :, :, :].unsqueeze(2)

        scaled_dot_product = torch.sum(
            torch.sum(
                torch.sum(query * k, dim=-1),
                dim=-1
            ),
            dim=-1
        ) / torch.sqrt(dim_k)


        if mask is not None:

            mask = torch.cat(
                (torch.zeros_like(mask[:, :, :seq + 1]), mask[:, :, :seq_len - seq - 1]),
                dim=-1
            )
            scaled_dot_product += (mask * -INFINITY)
                                   

        attention_weight = torch.nn.functional.softmax(scaled_dot_product, dim=-1)
        output = attention_weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * v


        output = torch.sum(output, dim=2)

        attention_weights.append(attention_weight)
        outputs.append(output)


    attention_weights = torch.stack(attention_weights)
    outputs = torch.stack(outputs)

    attention_weights = attention_weights.permute(1, 2, 0, 3)
    outputs = outputs.permute(1, 2, 0, 3, 4, 5)

    return outputs, attention_weights

class MultiHeadAttention(tf.keras.layers.Layer):
    
    def __init__(self, d_model, num_heads, filter_size):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.filter_size = filter_size

        assert (d_model % num_heads == 0)

        self.depth = d_model // num_heads


        self.wq = nn.Conv2d(d_model, d_model, filter_size, padding='same')
        self.wk = nn.Conv2d(d_model, d_model, filter_size, padding='same')
        self.wv = nn.Conv2d(d_model, d_model, filter_size, padding='same')

        self.final_weight = nn.Conv2d(d_model, d_model, filter_size, padding='same')


    def split_heads(self, x):
        """
        Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is 
        (batch_size, num_heads, seq_len, rows, cols, depth)
        
        Args:
        x : shape = (batch_size, seq_len, rows, cols, depth)
        """

        x = x.view(x.shape[:4] + (self.num_heads, self.depth))
        return x.permute(0, 4, 1, 2, 3, 5)
        
    def call(self, q, k, v, mask=None):
        
        shape_q = q.shape
        q = self.wq(q) # (batch_size, seq_len_q, rows, cols, depth)
        k = self.wk(k) # (batch_size, seq_len_k, rows, cols, depth)
        v = self.wv(v) # (batch_size, seq_len_v, rows, cols, depth)

        q = self.split_heads(q) # (batch_size, num_heads, seq_len_q, rows, cols, depth)
        k = self.split_heads(k) # (batch_size, num_heads, seq_len_k, rows, cols, depth)
        v = self.split_heads(v) # (batch_size, num_heads, seq_len_v, rows, cols, depth)


        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = scaled_attention.permute(0, 2, 3, 4, 1, 5)

        concat_attention = scaled_attention.view(shape_q)

        output = self.final_weight(concat_attention)

        return output, attention_weights
