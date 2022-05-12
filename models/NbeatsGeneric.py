import gin
import torch as t
import torch.nn as nn
import numpy as np
from models.basis_functions import GenericBasis
from models.NbeatsBlock import NbeatsBlock


@gin.configurable()
class NbeatsGeneric(nn.Module):
    """
    Create N-BEATS generic model.
    """

    def __init__(self, input_size: int, output_size: int, block_num: int,
                 stack_num: int, layer_num: int, layer_size: int, share_thetas: bool = False, dropout: float = 1.0):
        super(NbeatsGeneric, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.stack_num = stack_num
        self.block_num = block_num
        self.layer_num = layer_num
        self.layer_size = layer_size
        self.share_thetas = share_thetas
        self.dropout = dropout

        self.generic_stack = t.nn.ModuleList([NbeatsBlock(input_size=self.input_size,
                                                          theta_size=self.input_size + self.output_size,
                                                          basis_function=GenericBasis(backcast_size=self.input_size,
                                                                                      forecast_size=self.output_size,
                                                                                      theta_size=self.input_size + self.output_size),
                                                          layer_num=self.layer_num,
                                                          layer_size=self.layer_size,
                                                          share_thetas=self.share_thetas,
                                                          dropout=self.dropout)
                                              for _ in range(self.block_num * self.stack_num)])

    def forward(self, x):
        residuals = x.flip(dims=(1,))
        forecast = x[:, -1:]

        for i, block in enumerate(self.generic_stack):
            backcast, block_forecast = block(residuals)
            residuals = (residuals - backcast)
            forecast = forecast + block_forecast
        return forecast
