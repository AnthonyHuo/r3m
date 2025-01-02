# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

class FCNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim,
                 hidden_sizes=(64,64),
                 nonlinearity='tanh',   # either 'tanh' or 'relu'
                 in_shift = None,
                 in_scale = None,
                 out_shift = None,
                 out_scale = None):
        super(FCNetwork, self).__init__()

        self.obs_dim = obs_dim
        print(obs_dim)
        self.act_dim = act_dim
        assert type(hidden_sizes) == tuple
        self.layer_sizes = (256+30, ) + hidden_sizes + (act_dim, )
        # self.layer_sizes = (obs_dim, ) + hidden_sizes + (act_dim, )
        self.set_transformations(in_shift, in_scale, out_shift, out_scale)
        self.proprio_only = False

        # Batch Norm Layers
        # self.bn = torch.nn.BatchNorm1d(obs_dim)
        self.bn = torch.nn.BatchNorm1d(256+30)

        # hidden layers
        # print(self.layer_sizes)
        self.fc_layers = nn.ModuleList([nn.Linear(self.layer_sizes[i], self.layer_sizes[i+1]) \
                         for i in range(len(self.layer_sizes) -1)])
        self.nonlinearity = torch.relu if nonlinearity == 'relu' else torch.tanh
        self.feature_neck_hidden_dim = 256
        num_tokens_edge = 14
        # self.neck = nn.Sequential(
        #         # nn.LayerNorm(768),
        #         Rearrange("b (h w) c -> b c h w", h=num_tokens_edge, w=num_tokens_edge),
        #         # nn.BatchNorm2d(768),
        #         nn.Conv2d(self.obs_dim-30, self.feature_neck_hidden_dim, kernel_size=4, stride=2, padding=1), #14x14 -> 7x7
        #         nn.ReLU() if nonlinearity == 'relu' else nn.Tanh(), # just to keep the same as super class
        #         nn.Conv2d(self.feature_neck_hidden_dim, self.feature_neck_hidden_dim, kernel_size=3, stride=2), #7x7 -> 3x3
        #         nn.ReLU() if nonlinearity == 'relu' else nn.Tanh(),
        #         nn.Conv2d(self.feature_neck_hidden_dim, self.feature_neck_hidden_dim, kernel_size=3, stride=1), #3x3 -> 1x1
        #         nn.ReLU() if nonlinearity == 'relu' else nn.Tanh(),
        #         nn.Flatten(),
        #         # nn.Linear(self.feature_neck_hidden_dim,self.feature_neck_hidden_dim),
        #         # # n.ReLU() if nonlinearity == 'relu' else nn.Tanh(),
        #         # nn.Linear(self.feature_neck_hidden_dim,self.act_dim),
        # )
        self.neck = nn.Sequential(
            Rearrange("b (h w) c -> b c h w", h=num_tokens_edge, w=num_tokens_edge),
            nn.Conv2d(self.obs_dim-30, 256, kernel_size=4, stride=2, padding=1),
            nn.LayerNorm([256, 7, 7]),
            nn.ReLU() if nonlinearity == "relu" else nn.Tanh(),  # 14x14 -> 7x7  # just to keep the same as super class
            nn.Conv2d(256, 256, kernel_size=3, stride=2),
            nn.LayerNorm([256, 3, 3]),
            nn.ReLU() if nonlinearity == "relu" else nn.Tanh(),  # 7x7 -> 3x3
            nn.Conv2d(256, 256, kernel_size=3, stride=1),
            nn.LayerNorm([256, 1, 1]),
            nn.ReLU() if nonlinearity == "relu" else nn.Tanh(),  # 3x3 -> 1x1
            nn.Flatten(),
        )

    def set_transformations(self, in_shift=None, in_scale=None, out_shift=None, out_scale=None):
        # store native scales that can be used for resets
        self.transformations = dict(in_shift=in_shift,
                           in_scale=in_scale,
                           out_shift=out_shift,
                           out_scale=out_scale
                          )
        self.in_shift  = torch.from_numpy(np.float32(in_shift)) if in_shift is not None else torch.zeros(self.obs_dim)
        self.in_scale  = torch.from_numpy(np.float32(in_scale)) if in_scale is not None else torch.ones(self.obs_dim)
        self.out_shift = torch.from_numpy(np.float32(out_shift)) if out_shift is not None else torch.zeros(self.act_dim)
        self.out_scale = torch.from_numpy(np.float32(out_scale)) if out_scale is not None else torch.ones(self.act_dim)

    def forward(self, x):
        # Small MLP runs on CPU
        # Required for the way the Gaussian MLP class does weight saving and loading.
        if x.is_cuda:
            out = x.to('cpu')
        else:
            out = x
        
        ## BATCHNORM
        # print(out.shape)
        # out = self.bn(out)
        # for i in range(len(self.fc_layers)-1):
        #     out = self.fc_layers[i](out)
        #     out = self.nonlinearity(out)
        # out = self.fc_layers[-1](out)
        # out = out * self.out_scale + self.out_shift
       
        prio = out[:,0,384:]
        out = out[:,:,:384]
      
        out = self.neck(out)
        out = torch.cat([out,prio],dim=-1)
       
        out = self.bn(out)
        for i in range(len(self.fc_layers)-1):
            out = self.fc_layers[i](out)
            out = self.nonlinearity(out)
        out = self.fc_layers[-1](out)
        out = out * self.out_scale + self.out_shift
        return out
