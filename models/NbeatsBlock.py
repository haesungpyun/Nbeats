import torch as t
import torch.nn as nn
import torch.nn.functional as F


class NbeatsBlock(nn.Module):
    def __init__(self,
                 input_size,
                 theta_size, #
                 basis_function : t.nn.Module,
                 layer_num : int,        # 주로 layer_num = 4
                 layer_size : int,
                 share_thetas : bool = False,
                 dropout : float  = 1.0):

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
