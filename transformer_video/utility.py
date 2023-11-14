import numpy as np
import torch
import torch.nn as nn

BASE = 10000

def feed_forward_network(dff, d_model, filter_size):
    return nn.Sequential(
        nn.Conv2d(in_channels=d_model, out_channels=dff, kernel_size=filter_size, padding=filter_size//2),
        nn.ReLU(),
        nn.Conv2d(in_channels=dff, out_channels=d_model, kernel_size=filter_size, padding=filter_size//2)
    )

def get_angles(pos, i, base_dim, d_model):
    angle_rates = 1 / np.power(BASE, (2 * (i // 2)) / np.float32(d_model))
    angle_rates = np.broadcast_to(
        np.reshape(angle_rates, (1, 1, 1, -1)),
        (1, base_dim[0], base_dim[1], d_model)
    )
    return pos * angle_rates

def positional_encoding(position, base_dim, d_model):
    angle_rads = get_angles(
        np.arange(position)[:, np.newaxis, np.newaxis, np.newaxis],
        np.arange(d_model)[np.newaxis, np.newaxis, np.newaxis, :],
        base_dim,
        d_model
    )

    # apply sin to even indices; 2i
    angle_rads[:, :, :, 0::2] = np.sin(angle_rads[:, :, :, 0::2])

    # apply cos to odd indices; 2i + 1
    angle_rads[:, :, :, 1::2] = np.cos(angle_rads[:, :, :, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return torch.tensor(pos_encoding, dtype=torch.float32)

def create_look_ahead_mask(seq_len):
    return torch.ones((1, 1, seq_len), dtype=torch.float32)
