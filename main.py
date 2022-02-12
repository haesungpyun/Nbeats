import torch as t
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
"""
NbeatsBlock > GenericBasis > Nbeats 
"""

class GenericBasis(nn.Module):
    def __init__(self, backcast_size : int, forecast_size : int, theta_size):
        super(GenericBasis, self).__init__()
        self.fc_backcast = nn.Linear(theta_size, backcast_size)
        self.fc_forecast = nn.Linear(theta_size, forecast_size)

    def forward(self, theta_backcast : t.Tensor, theta_forecast : t.Tensor):
        block_backcast = self.fc_backcast(theta_backcast)
        block_forecast = self.fc_forecast(theta_forecast)
        return block_backcast, block_forecast



class TrendBasis(nn.Module):
    def __init__(self, degree_of_polynomial: int, backcast_size: int, forecast_size: int):
        super().__init__()
        self.polynomial_size = degree_of_polynomial + 1  # degree of polynomial with constant term
        self.backcast_time = t.nn.Parameter(
            t.tensor(np.concatenate([np.power(np.arange(backcast_size, dtype=np.float) / backcast_size, i)[None, :]
                                     for i in range(self.polynomial_size)]), dtype=t.float32),
            requires_grad=False)
        self.forecast_time = t.nn.Parameter(
            t.tensor(np.concatenate([np.power(np.arange(forecast_size, dtype=np.float) / forecast_size, i)[None, :]
                                     for i in range(self.polynomial_size)]), dtype=t.float32), requires_grad=False)

    def forward(self, theta: t.Tensor):
        backcast = t.einsum('bp,pt->bt', theta_backcast, self.backcast_time)
        forecast = t.einsum('bp,pt->bt', theta_forcast, self.forecast_time)
        return backcast, forecast

class NbeatsBlock(nn.Module):
    def __init__(self,
                 input_size,
                 theta_size,
                 basis_function : t.nn.Module,
                 layer_num : int,        # 주로 layer_num = 4
                 layer_size : int,
                 share_thetas : bool = False,
                 dropout : int  = 1.0):

        super(NbeatsBlock, self).__init__()
        self.fc = t.nn.ModuleList([t.nn.Linear(in_features= input_size, out_features= layer_size)] +
                                      [t.nn.Linear(in_features= layer_size, out_features= layer_size)
                                      for _ in range(layer_num-1)])
        if share_thetas:
            self.theta_f_fc = self.theta_b_fc = t.nn.Linear(in_features=layer_size, out_features= theta_size)
        else:
            self.theta_f_fc = t.nn.Linear(in_features=layer_size, out_features= theta_size)
            self.theta_b_fc = t.nn.Linear(in_features=layer_size, out_features= theta_size)
        self.basis_function = basis_function
        self.dropout = nn.Dropout(p = dropout)

    def forward(self, x : t.Tensor):
        block_input = x
        for linear in self.fc:
            block_input = F.relu(linear(block_input))
            block_input = self.dropout(block_input)
        theta_backcast = self.theta_b_fc(block_input)
        theta_forecast = self.theta_f_fc(block_input)
        return self.basis_function(theta_backcast, theta_forecast)



class Nbeats(nn.Module):

    def __init__(self, blocks : t.nn.ModuleList):
        super(Nbeats).__init__()
        self.blocks = blocks

    def forward(self, x: t.Tensor, input_mask : t.Tensor) -> t.Tensor:
        residuals = x.flip(dims=(1,))
        input_mask = input_mask.flip(dims=(1,))
        forecast = x[:, -1:]

        for i, block in enumerate(self.blocks):
            backcast, block_forecast = block(residuals)
            residuals = (residuals - backcast) * input_mask
            forecast = forecast + block_forecast
        return forecast



