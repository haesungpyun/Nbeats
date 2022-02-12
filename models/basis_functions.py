import torch as t
import numpy as np
import torch.nn as nn


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
    def __init__(self, degree_of_polynomial: int, backcast_size: int, forecast_size: int): #theta_size = degree_of_poly
        super().__init__()
        self.polynomial_size = degree_of_polynomial + 1  # degree of polynomial with constant term
        self.backcast_time = t.nn.Parameter(
            t.tensor(np.concatenate([np.power(np.arange(backcast_size, dtype=np.float) / backcast_size, i)[None, :]
                                     for i in range(self.polynomial_size)]), dtype=t.float32),
            requires_grad=False)
        self.forecast_time = t.nn.Parameter(
            t.tensor(np.concatenate([np.power(np.arange(forecast_size, dtype=np.float) / forecast_size, i)[None, :]
                                     for i in range(self.polynomial_size)]), dtype=t.float32), requires_grad=False)

    def forward(self, theta_backcast : t.Tensor, theta_forecast : t.Tensor):
        backcast = t.einsum('bp,pt->bt', theta_backcast, self.backcast_time)
        forecast = t.einsum('bp,pt->bt', theta_forecast, self.forecast_time)
        return backcast, forecast



class SeasonalityBasis(nn.Module):
    def __init__(self, harmonics: int, backcast_size: int, forecast_size: int):
        super().__init__()
        self.frequency = np.append(np.zeros(1, dtype=np.float32),
                                   np.arange(harmonics, harmonics / 2 * forecast_size,
                                             dtype=np.float32) / harmonics)[None, :]
        backcast_grid = -2 * np.pi * (
                np.arange(backcast_size, dtype=np.float32)[:, None] / forecast_size) * self.frequency
        forecast_grid = 2 * np.pi * (
                np.arange(forecast_size, dtype=np.float32)[:, None] / forecast_size) * self.frequency
        self.backcast_cos_template = t.nn.Parameter(t.tensor(np.transpose(np.cos(backcast_grid)), dtype=t.float32),
                                                    requires_grad=False)
        self.backcast_sin_template = t.nn.Parameter(t.tensor(np.transpose(np.sin(backcast_grid)), dtype=t.float32),
                                                    requires_grad=False)
        self.forecast_cos_template = t.nn.Parameter(t.tensor(np.transpose(np.cos(forecast_grid)), dtype=t.float32),
                                                    requires_grad=False)
        self.forecast_sin_template = t.nn.Parameter(t.tensor(np.transpose(np.sin(forecast_grid)), dtype=t.float32),
                                                    requires_grad=False)

    def forward(self, theta_backcast: t.Tensor, theta_forecast: t.Tensor):
        params_per_harmonic_b = theta_backcast.shape[1] // 2
        params_per_harmonic_f = theta_backcast.shape[1] // 2
        backcast_harmonics_cos = t.einsum('bp,pt->bt', theta_backcast[:, :params_per_harmonic_b],
                                          self.backcast_cos_template)
        backcast_harmonics_sin = t.einsum('bp,pt->bt', theta_backcast[:, params_per_harmonic_b:], self.backcast_sin_template)
        backcast = backcast_harmonics_sin + backcast_harmonics_cos
        forecast_harmonics_cos = t.einsum('bp,pt->bt',
                                          theta_forecast[:, :params_per_harmonic_f], self.forecast_cos_template)
        forecast_harmonics_sin = t.einsum('bp,pt->bt', theta_forecast[:, params_per_harmonic_f:],
                                          self.forecast_sin_template)
        forecast = forecast_harmonics_sin + forecast_harmonics_cos

        return backcast, forecast