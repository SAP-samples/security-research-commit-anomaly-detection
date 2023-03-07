import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np
from torch_geometric.nn.conv import GraphConv, GENConv, GATv2Conv
from torch_geometric.nn.models import InnerProductDecoder 
from torch.optim import Adam
from tqdm import tqdm
np.set_printoptions(threshold=sys.maxsize)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPS = 1e-15
MAX_LOGSTD = 10

''' GENConv based Classifier '''

class GCNEncoder(nn.Module):
    def __init__(self, hidden_channels, n_layers, n_aggr, dropout, bias=True):
        super(GCNEncoder, self).__init__()
        self.gcn_aggrs = torch.nn.ModuleList()
        self.dropout = nn.Dropout(p=dropout)        
        for a in range(n_aggr):
            gcn = GENConv(hidden_channels, hidden_channels, num_layers=n_layers, aggr='softmax', bias=bias, learn_t =True)
            self.gcn_aggrs.append(gcn)        

    def forward(self, x, edge_index, edge_attr=None):
        for g, gcn_aggr in enumerate(self.gcn_aggrs):            
            x = gcn_aggr(x, edge_index, edge_attr)
            x = torch.relu(x)
            x = self.dropout(x)
        
        return x

class LinDecoder(nn.Module):
    def __init__(self, hidden_channels, n_layers, dropout, bias=True):
        super(LinDecoder, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.dropout = nn.Dropout(p=dropout)        
        for a in range(n_layers):
            if a < n_layers-1:
                lin = nn.Linear(hidden_channels, hidden_channels, bias=bias)
            elif a == n_layers-1:
                lin = nn.Linear(hidden_channels, 2, bias=bias)
            self.layers.append(lin)        

    def forward(self, Z):
        for l, lin in enumerate(self.layers):            
            Z = lin(Z)
            Z = torch.relu(Z)
            Z = self.dropout(Z)
        
        return Z

class GEN_Classifier(nn.Module):
    def __init__(self, n_channels, hidden_channels, n_layers, n_aggr, dropout, e_channels=None, bias=True):
        super(GEN_Classifier, self).__init__()
        self.n_linear = nn.Linear(n_channels, hidden_channels)
        if e_channels != None:
            self.e_linear = nn.Linear(e_channels, hidden_channels)
        
        self.encode = GCNEncoder(hidden_channels, n_layers, n_aggr, dropout, bias=bias)
        self.decode = LinDecoder(hidden_channels, n_aggr, dropout, bias=bias)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, edge_index, edge_attr=None):
       
        x = self.n_linear(x)
        if edge_attr != None:
            edge_attr = self.e_linear(edge_attr)
        
        self.Z = self.encode(x, edge_index, edge_attr)
        self.Z = self.decode(self.Z)
        self.Z_probs = self.Z.argmax(dim=-1)
        return self.Z_probs.float()
    
    def CE_loss(self, targets):        
        NLL_loss = self.loss(self.Z, targets.long())      
        return NLL_loss       

    
