import sys
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

def MyModel(args):
    if args.loss == "nllloss":
        last_act = nn.LogSoftmax(dim=-1)
    else:
        last_act = nn.Sigmoid()

    if args.model == "CNN":
        model = ConvNet(input_dim=args.input_dim, n_feature_in=args.n_feature_in, n_feature_out=args.n_feature_out, hidden_dim=args.hidden_dim, output_dim=args.output_dim, n_layer=args.n_layer, r_drop=args.r_drop, last_act=last_act)
    elif args.model == "MLP":
        model = MLP(n_feature_in=args.n_feature_in, n_feature_out=args.n_feature_out, hidden_dim=args.hidden_dim, output_dim=args.output_dim, n_layer=args.n_layer, r_drop=args.r_drop, last_act=last_act)
    else:
        print("Error: unkonwn model", file=sys.stderr)
        sys.exit(1)
    return model

class ConvBlock(nn.Module):

    def __init__(self, nin=32, nout=32, kernel_size=5, stride=2, padding="same", r_drop=0):
        super().__init__()
        self.conv = nn.Conv2d(nin, nout, kernel_size=kernel_size, stride=stride, padding=padding)
        self.drop = nn.Dropout(r_drop)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.drop(x)
        x = self.act(x)

        return x
  
class ConvNet(nn.Module):

    def __init__(self, n_feature_in=3, n_feature_out=3, hidden_dim=32, input_dim=106, output_dim=1, n_layer=4, kernel_size=5, r_drop=0, last_act=nn.LogSoftmax(dim=1)):
        super().__init__()

        padding = int( kernel_size / 2 )

        input_dims = [ n_feature_in ] + [ hidden_dim * 2**i for i in range(n_layer-1) ]
        output_dims = [ hidden_dim * 2**i for i in range(n_layer) ]
        if n_layer == 1:
            dropout_rates = [ r_drop ]
        else:
            dropout_rates = [0] + [ r_drop for i in range(n_layer-1) ] 
        self.blocks = nn.ModuleList([
            ConvBlock(nin=i, nout=j, stride=2, kernel_size=kernel_size, padding=padding, r_drop=r)
            for i, j, r in zip(input_dims, output_dims, dropout_rates) 
            ])
        
        tmp = input_dim
        for i in range(n_layer): 
            tmp = int( ( tmp + 1 ) / 2 )
        final_dim = tmp * tmp * output_dims[-1]
        
        self.linear = nn.Linear(final_dim, n_feature_out*output_dim)

        self.output_act = last_act
        self.output_dim = output_dim

    def forward(self, x):
        ## x: (batch, seq, input_dim)

        batch_size = x.size(0)
        for blk in self.blocks:
            
            x = blk(x)

        x = x.contiguous().view(batch_size, -1)

        x = self.linear(x)
        if self.output_dim > 1:
            x = x.view(batch_size, -1, self.output_dim)
            
        x = self.output_act(x)
        
        # x: (batch, n_feature_out, output_dim) or (batch, n_feature_out)
        return x


class MLP(nn.Module):

    def __init__(self, n_feature_in=3, n_feature_out=1, hidden_dim=32, output_dim=1, n_layer=4, r_drop=0, last_act=nn.LogSoftmax(dim=1)):
        super().__init__()

        self.output_dim = output_dim

        dims = [ n_feature_in ] + [ hidden_dim for i in range(n_layer-1) ] + [ n_feature_out * output_dim ]
        layers = []
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims)-2:
                layers.append(nn.LeakyReLU())
                if i > 0:
                    layers.append(nn.Dropout(r_drop))

        self.model = nn.Sequential(*layers)

        self.output_act = last_act
    
    def forward(self, x):
        ## x: (batch, seq, input_dim)
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x = self.model(x)

        if self.output_dim > 1:
            x = x.view(batch_size, -1, self.output_dim)
            
        x = self.output_act(x)

        # x: (batch, n_feature_out, output_dim) or (batch, n_feature_out)
        return x
         