import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np
from torch_geometric.nn.conv import GraphConv, GENConv
from torch_geometric.nn.models import InnerProductDecoder 
from torch.optim import Adam
from tqdm import tqdm
np.set_printoptions(threshold=sys.maxsize)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPS = 1e-15
MAX_LOGSTD = 10

''' Deep_SAD implementation '''
''' Modified GENConv() here 
    /home/ubuntu/anaconda3/envs/pytorch_latest_p37/lib/python3.7/site-packages/torch_geometric/nn/conv/gen_conv.py
    to include bias option '''

class GCNEncoder_VAE(nn.Module):
    def __init__(self, hidden_channels, n_layers, n_aggr, dropout, bias=False):
        super(GCNEncoder_VAE, self).__init__()
        self.gcn_aggrs = torch.nn.ModuleList()
        self.dropout = nn.Dropout(p=dropout)        
        for a in range(n_aggr):
            gcn = GENConv(hidden_channels, hidden_channels, num_layers=n_layers, aggr='softmax', bias=bias, learn_t =True)
            self.gcn_aggrs.append(gcn)
        self.gcn_mu = GENConv(hidden_channels, hidden_channels, num_layers=n_layers, aggr='softmax', bias=bias, learn_t =True)
        self.gcn_logstd = GENConv(hidden_channels, hidden_channels, num_layers=n_layers, aggr='softmax', bias=bias, learn_t =True) 

    def forward(self, x, edge_index, edge_attr=None):
        for gcn_aggr in self.gcn_aggrs:
            x = gcn_aggr(x, edge_index, edge_attr)
            x = torch.relu(x)
            x = self.dropout(x)
        mu = self.gcn_mu(x, edge_index, edge_attr)
        logstd = self.gcn_logstd(x, edge_index, edge_attr)
        return mu, logstd

class GCNDecoder(nn.Module):
    def __init__(self, hidden_channels, out_channels, n_layers, n_aggr, dropout, bias=False):
        super(GCNDecoder, self).__init__()
        self.gcn_aggrs = torch.nn.ModuleList()
        self.dropout = nn.Dropout(p=dropout)        
        for a in range(n_aggr-1):
            gcn = GENConv(hidden_channels, hidden_channels, num_layers=n_layers, aggr='softmax', bias=bias, learn_t =True)
            self.gcn_aggrs.append(gcn)    
        gcn = GENConv(hidden_channels, out_channels, num_layers=n_layers, aggr='softmax', bias=bias, learn_t =True)
        self.gcn_aggrs.append(gcn)        

    def forward(self, x_hat, edge_index, edge_attr=None):
        for gcn_aggr in self.gcn_aggrs:
            x_hat = gcn_aggr(x_hat, edge_index, edge_attr)
            x_hat = torch.relu(x_hat)
            x_hat = self.dropout(x_hat)
        return x_hat 


class DeepSAD_GVAE(nn.Module):
    def __init__(self, n_channels, hidden_channels, n_layers, n_aggr, dropout, e_channels=None, bias=False):
        super(DeepSAD_GVAE, self).__init__()
        self.n_linear = nn.Linear(n_channels, hidden_channels)
        if e_channels != None:
            self.e_linear = nn.Linear(e_channels, hidden_channels)
        
        self.encode = GCNEncoder_VAE(hidden_channels, n_layers, n_aggr, dropout, bias=bias)
        self.decode = GCNDecoder(hidden_channels, n_channels, n_layers, n_aggr, dropout, bias=bias)
        self.eps = 1e-6

    def reparametrize(self, mu, logstd):
        return mu + torch.randn_like(logstd) * torch.exp(logstd)        

    def forward(self, x, edge_index, edge_attr=None, centers=None, stdev=None):
        self.N = x.shape[0]
        self.x_in = x.clone()
        self.c = centers
        self.std = stdev

        x = self.n_linear(x)
        if edge_attr != None:
            edge_attr = self.e_linear(edge_attr)
        
        self.__mu__, self.__logstd__ = self.encode(x, edge_index, edge_attr)
        self.__logstd__ = self.__logstd__.clamp(max=MAX_LOGSTD)
        self.Z = self.reparametrize(self.__mu__, self.__logstd__)        
        if self.c is None:
            self.Z_reconst = self.decode(self.Z, edge_index, edge_attr)
            return self.Z_reconst, self.Z
        else:
            return self.Z
    
    def loss(self,):
        node_loss = torch.mean(torch.norm(self.x_in - self.Z_reconst, dim=1)) 
        # node_loss = F.binary_cross_entropy(F.tanh(self.Z_reconst), self.x_in)
        kl_l = self.kl_loss()
        return node_loss + kl_l

    def kl_loss(self, mu=None, logstd=None):        
        mu = self.__mu__ if mu is None else mu
        logstd = self.__logstd__ if logstd is None else logstd.clamp(
            max=MAX_LOGSTD)
        return -0.5 * torch.mean(torch.mean(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1))

    def HSClassifierLoss(self, x_labels, eta): 
        # dist = torch.sum((self.Z - self.c) ** 2, dim=1)
        dist = torch.norm(((self.Z - self.c) / (2 * self.std)), dim=1)
        losses = torch.where(x_labels == 0, dist, eta * ((dist + self.eps) ** x_labels.float()))
        loss = torch.mean(losses)
        return loss, dist
