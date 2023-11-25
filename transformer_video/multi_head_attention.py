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
        query = q[:, :, seq, :, :, :]
        key = k[:, :, seq, :, :, :]
        scaled_dot_product = torch.matmul(query, key.permute(0, 1, 2, 4, 5, 3)) / torch.sqrt(torch.tensor(dim_k, dtype=torch.float32))

        if mask is not None:
            mask = torch.cat(
                (torch.zeros_like(mask[:, :, :, :seq + 1, :, :]), mask[:, :, :, :seq_len - seq - 1, :, :]),
                dim=-1
            )
            scaled_dot_product += (mask * -INFINITY)

        attention_weight = torch.nn.functional.softmax(scaled_dot_product, dim=-1)
        output = torch.matmul(attention_weight, v[:, :, seq, :, :, :])

        attention_weights.append(attention_weight)
        outputs.append(output)

    attention_weights = torch.stack(attention_weights)
    outputs = torch.stack(outputs)

    attention_weights = attention_weights.permute(1, 2, 0, 3)
    outputs = outputs.permute(1, 2, 0, 3, 4, 5)

    return outputs, attention_weights


class MultiHeadAttention(nn.Module):
    
    def __init__(self, d_model, num_heads, filter_size):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.filter_size = filter_size

        assert (d_model % num_heads == 0)

        self.depth = d_model // num_heads

        # expects [batch_size, channels, height, width]
        self.wq = nn.Conv2d(d_model, d_model, filter_size, padding='same')
        self.wk = nn.Conv2d(d_model, d_model, filter_size, padding='same')
        self.wv = nn.Conv2d(d_model, d_model, filter_size, padding='same')

        #self.final_weight = nn.Conv2d(d_model, d_model, filter_size, padding='same')
        self.final_weight = nn.Conv2d(d_model * num_heads, d_model, filter_size, padding='same')


    def split_heads(self, x):
        """
        Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is 
        (batch_size, num_heads, seq_len, rows, cols, depth)
        
        Args:
        x : shape = (batch_size, seq_len, rows, cols, depth)
        """

        # x = x.view(x.shape[:4] + (self.num_heads, self.depth))
        # return x.permute(0, 4, 1, 2, 3, 5)
        x = x.view(x.shape[0], self.num_heads, x.shape[1], x.shape[2], x.shape[3], -1)
        return x.permute(0, 2, 3, 4, 1, 5).contiguous()
        
    def forward(self, q, k, v, mask=None):
        
        shape_q = q.shape

        # Reshape input tensors to [batch_size * num_heads, channels, height, width]
        q = q.view(-1, shape_q[-3], shape_q[-2], shape_q[-1])
        k = k.view(-1, shape_q[-3], shape_q[-2], shape_q[-1])
        v = v.view(-1, shape_q[-3], shape_q[-2], shape_q[-1])

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        # Reshape back to [batch_size, num_heads, channels, height, width]
        q = q.view(shape_q[0], self.num_heads, -1, shape_q[-2], shape_q[-1])
        k = k.view(shape_q[0], self.num_heads, -1, shape_q[-2], shape_q[-1])
        v = v.view(shape_q[0], self.num_heads, -1, shape_q[-2], shape_q[-1])

        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        # Reshape to [batch_size, seq_len, rows, cols, num_heads * depth]
        concat_attention = scaled_attention.reshape(shape_q[0], -1, scaled_attention.shape[-3], scaled_attention.shape[-2], self.num_heads * self.depth)

        output = self.final_weight(concat_attention)
        print("completed MHA")
        return output, attention_weights