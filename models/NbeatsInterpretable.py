import os
import torch as t
import torch.nn as nn
import numpy as np
from models.basis_functions import TrendBasis, SeasonalityBasis
from models.NbeatsBlock import NbeatsBlock

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class NbeatsInterpretable(t.nn.Module):
    """
    Create N-BEATS interpretable model.
    """

    def __init__(self, input_size: int, output_size: int, trend_layer_size: int, trend_layer_num: int,
                 trend_block_num: int,
                 trend_stack_num: int, degree_of_polynomial: int,
                 seasonality_layer_size: int, seasonality_layer_num: int, seasonality_block_num: int,
                 seasonality_stack_num: int,
                 harmoincs_num: int,
                 share_thetas: bool = False, dropout: float = 1.0):
        super(NbeatsInterpretable, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.trend_layer_size = trend_layer_size
        self.trend_layer_num = trend_layer_num
        self.trend_block_num = trend_block_num
        self.trend_stack_num = trend_stack_num
        self.degree_of_polynomial = degree_of_polynomial
        self.seasonality_layer_size = seasonality_layer_size
        self.seasonality_layer_num = seasonality_layer_num
        self.seasonality_block_num = seasonality_block_num
        self.seasonality_stack_num = seasonality_stack_num
        self.harmonics_num = harmoincs_num
        self.share_thetas = share_thetas
        self.dropout = dropout

        self.trend_stack = [NbeatsBlock(input_size=self.input_size,
                                        theta_size=degree_of_polynomial + 1,
                                        basis_function=TrendBasis(
                                            degree_of_polynomial=self.degree_of_polynomial,
                                            backcast_size=self.input_size,
                                            forecast_size=self.output_size),
                                        layer_num=self.trend_layer_num,
                                        layer_size=self.trend_layer_size,
                                        share_thetas=self.share_thetas,
                                        dropout=self.dropout)
                            for _ in range(self.trend_block_num)]

        self.seasonality_stack = [NbeatsBlock(input_size=self.input_size,
                                              theta_size=2 * int(np.ceil(self.harmonics_num / 2 * self.output_size) - (
                                                      self.harmonics_num - 1)),
                                              basis_function=SeasonalityBasis(
                                                  harmonics=self.harmonics_num,
                                                  backcast_size=self.input_size,
                                                  forecast_size=self.output_size),
                                              layer_num=self.seasonality_layer_num,
                                              layer_size=self.seasonality_layer_size,
                                              share_thetas=self.share_thetas,
                                              dropout=self.dropout)
                                  for _ in range(self.seasonality_block_num)]

        self.interpretable_stack = nn.ModuleList(self.trend_stack)
        for _ in range(self.trend_stack_num - 1):
            self.interpretable_stack.extend(self.trend_stack)
        for _ in range(self.seasonality_stack_num):
            self.interpretable_stack.extend(self.seasonality_stack)

    def forward(self, x):
        residuals = x.flip(dims=(1,))
        forecast = x[:, -1:]

        for i, block in enumerate(self.interpretable_stack):
            backcast, block_forecast = block(residuals)
            residuals = (residuals - backcast)
            forecast = forecast + block_forecast
        return forecast
