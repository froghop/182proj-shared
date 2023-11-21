# masha
import time

import numpy as np
import torch
import torch.nn as nn
from transformer import Transformer
from utility import create_look_ahead_mask


class VideoPrediction:
    def __init__(self, num_layers, d_model, num_heads, dff, filter_size, image_shape, 
                 pe_input, pe_target, out_channel, loss_function='mse', optimizer='rmsprop'):
        self.transformer = Transformer(num_layers, d_model, num_heads, dff, filter_size,
                                       image_shape, pe_input, pe_target, out_channel)
        self.loss_object = nn.MSELoss() if loss_function == 'mse' else nn.BCELoss()
        self.optimizer   = torch.optim.RMSprop(self.transformer.parameters(), lr=0.001, alpha=0.9) \
            if optimizer == 'rmsprop' else torch.optim.Adam(self.transformer.parameters())

    def loss_function(self, real, pred):
        return self.loss_object(real, pred)
        
    def train_step(self, inp, tar):
        
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]
        
        look_ahead_mask = create_look_ahead_mask(tar.shape[1])
        loss = 0

        self.optimizer.zero_grad()

        predictions, _ = self.transformer(inp, tar_inp, True, look_ahead_mask)
        loss = self.loss_function(predictions, tar_real)

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self, inp, tar, inp_val, tar_val, epochs, batch_size, epoch_print=5):

        start = time.time()
        for epoch in range(epochs):
            total_loss = 0
            total_batch = inp.shape[0] // batch_size
            
            for batch in range(total_batch):
                index = batch * batch_size
                enc_inp = inp[index:index + batch_size, :, :, :]
                dec_inp = tar[index:index + batch_size, :, :, :]
                
                batch_loss = self.train_step(enc_inp, dec_inp)
                total_loss += batch_loss

            total_batch += 1
            if epoch % epoch_print == 0:
                print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / total_batch))
                print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
                start = time.time()
                
    def eval_step(self, inp, tar):
        
        batch_loss = 0
        image_size = inp.shape[2:]
        
        look_ahead_mask = create_look_ahead_mask(tar.shape[1])
            
        half = inp.shape[1] // 2
        output = inp[:, half:, :, :, :]
        encoder_input = inp[:, :half, :, :, :]
        
        for t in range(tar.shape[1]):
            prediction, _ = self.transformer(
                encoder_input, output, False, look_ahead_mask
            )
                
            predict = prediction[:, -1:, :, :, :]        
            batch_loss += self.loss_function(predict, tar[:, t:t+1])    

            output = torch.cat([output, predict], dim=1)
            encoder_input = torch.cat([encoder_input, output[:, 0:1]], dim=1)[:, 1:]
            output = output[:, 1:]
        
        return (batch_loss / int(tar.shape[1]))

    def evaluate(self, inp, tar, batch_size):
        
        start = time.time()
        total_loss = 0
        total_batch = inp.shape[0] // batch_size
            
        for batch in range(total_batch):
            index = batch * batch_size
            enc_inp = inp[index:index + batch_size, :, :, :]
            dec_inp = tar[index:index + batch_size, :, :, :]
                
            batch_loss = self.eval_step(enc_inp, dec_inp)
            total_loss += batch_loss

        total_batch += 1
        
        return total_loss / total_batch

    def predict(self, inp, tar_seq_len):

        inp = torch.unsqueeze(inp, 0)
        image_size = inp.shape[2:]
        
        look_ahead_mask = create_look_ahead_mask(tar_seq_len)
        
        predictions = []
        half = inp.shape[1] // 2
        output = inp[:, half:, :, :, :]
        encoder_input = inp[:, :half :, :, :]
        
        for t in range(tar_seq_len):
            print(f"encoder_input.shape: {encoder_input.shape}")
            print(f"output.shape: {output.shape}")
            print(f"look_ahead_mask.shape: {look_ahead_mask.shape}")
            prediction, _ = self.transformer(
                encoder_input, output, look_ahead_mask
            )
            print("Prediction shape:", prediction.shape)    
            predict = prediction[:, -1:, :, :, :]        
            output = torch.cat([output, predict], dim=1)
            
            encoder_input = torch.cat([encoder_input, output[:, 0:1]], dim=1)[:, 1:]

            output = output[:, 1:]
    
            predictions.append(
                predict.detach().numpy().reshape(
                    image_size
                )
            )
            
        return np.array(predictions)
