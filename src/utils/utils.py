import random as rd
import numpy as np
import torch, pickle
import scipy.sparse as sp
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from torch_geometric.utils import subgraph
from torch_geometric.data import Dataset, InMemoryDataset
from .gen_graph_dataset import Repo_Graph


class Repo_Dataset(Dataset):
    def __init__(self, N):
        super().__init__()        
        self.N = N
        self._data_list = None
        
    def len(self):
        return self.N

    def get(self, idx):
        if self._data_list is None:
            return self.data[idx]
        else:
            repo_graph = pickle.load(open(self._data_list[idx], "rb"))
            return repo_graph

    def load(self, ):
        self.data = []
        if self._data_list is None:
            for n in range(self.N):
                repo_graph = Repo_Graph()
                self.data.append(repo_graph)
        else:
            for url in self._data_list:
                repo_graph = pickle.load(open(url, "rb"))
                self.data.append(repo_graph)
        return self.data
   

class Repo_Dataset_IM(InMemoryDataset):
    def __init__(self, N):
        super().__init__()
        self.N = N

    def len(self):
        return self.N
    
    def get(self, idx):
        repo_graph = self.data[idx]
        return repo_graph

    def load(self, ):
        self.data = []
        if self._data_list is None:
            for n in range(self.N):
                repo_graph = Repo_Graph()
                self.data.append(repo_graph)
        else:
            for url in self._data_list:
                repo_graph = pickle.load(open(url, "rb"))
                self.data.append(repo_graph)
        return self.data
        

"""
	Utility functions 
"""
def gen_semi_labels(batch_label, nrml_th=0.1, anm_th=0.5):
    n_nrml_nodes = len(batch_label[batch_label == 0])
    nrml_indices = torch.where(batch_label == 0)[0].tolist()
    nrml_label_indices = rd.sample(nrml_indices, int(round(n_nrml_nodes * nrml_th)))
    
    n_anm_nodes = len(batch_label[batch_label == 1])
    anm_indices = torch.where(batch_label == 1)[0].tolist()
    anm_label_indices = rd.sample(anm_indices, int(round(n_anm_nodes * anm_th)))
    
    semi_labels = torch.zeros(len(batch_label), dtype=torch.int64)
    semi_labels[nrml_label_indices] = 1
    semi_labels[anm_label_indices] = -1

    return semi_labels

def load_dataset(name='Yelp', path="working/datasets/YelpChi.mat"):
    dataset = loadmat(path)
    adj = torch.tensor(dataset['homo'].toarray())
    # if name == 'yelp':
    #     adj_r1 = dataset['net_rur'].toarray()
    #     adj_r2 = dataset['net_rtr'].toarray()
    #     adj_r3 = dataset['net_rsr'].toarray()
    # elif name == 'amazon':
    #     adj_r1 = dataset['net_upu'].toarray()
    #     adj_r2 = dataset['net_usu'].toarray()
    #     adj_r3 = dataset['net_uvu'].toarray()
    
    features = torch.tensor(dataset['features'].toarray())
    labels = torch.tensor(dataset['label'])
    labels = labels.reshape(labels.shape[1])
    print(torch.sum(adj)/2)
    print(adj.shape, features.shape, labels.shape)

    return adj, features, labels


def get_anom_ratio(repo):
	n_nodes_after = repo.node_features.shape[0]
	n_targets_after = torch.sum(repo.targets).item()
	anom_ratio = np.round((n_targets_after/n_nodes_after) * 100, 2)
	return anom_ratio


def convert_to_edgeindices(adj):
    edge_indices_1, edge_indices_2 = np.where(adj == 1)
    edge_indices = torch.zeros((2, len(edge_indices_1)))
    edge_indices[0,:] = torch.tensor(edge_indices_1)
    edge_indices[1,:] = torch.tensor(edge_indices_2)
    return edge_indices.long()

def f_score_(nodes_pred, nodes, th=0.5):
    
    nodes_pred_copy_neg = np.where(nodes == 0, nodes_pred, 100)
    fp = len(np.where((nodes_pred_copy_neg > th) & (nodes_pred_copy_neg != 100))[0])
    tn = len(np.where((nodes_pred_copy_neg <= th) & (nodes_pred_copy_neg != 100))[0])
    
    nodes_pred_copy_pos = np.where(nodes == 1, nodes_pred, 100)
    fn = len(np.where((nodes_pred_copy_pos <= th) & (nodes_pred_copy_pos != 100))[0])
    tp = len(np.where((nodes_pred_copy_pos > th) & (nodes_pred_copy_pos != 100))[0])

    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)    

    f_score = 2 * ((precision * recall) / (precision + recall + 1e-12))
    return np.round(recall, 4), np.round(precision, 4), np.round(f_score, 4)


def create_split(repo, c_idx_1, c_idx_2, c_idx_3):
    repo.edge_indices, idxs = torch.unique(repo.edge_indices, dim=1, return_inverse=True)
    repo.edge_features = repo.edge_features[torch.unique(idxs)]

    # split_adj = repo.adj.clone()        
    # split_adj[c_idx_2, :], split_adj[:, c_idx_2] = 0, 0
    # split_adj[c_idx_3, :], split_adj[:, c_idx_3] = 0, 0

    split_adj = torch.zeros(repo.adj.shape, dtype=torch.bool)
    split_adj[c_idx_1, :], split_adj[:, c_idx_1] = repo.adj[c_idx_1, :], repo.adj[:, c_idx_1]
     

    idx_split = torch.unique(np.argwhere(split_adj == 1).flatten())
    edge_indices, edge_features \
        = subgraph(idx_split, repo.edge_indices, repo.edge_features, relabel_nodes=True)

    node_features = repo.node_features[idx_split].clone()  
    node_labels = repo.node_labels[idx_split].clone()  
    targets = repo.targets[idx_split].clone()         
    del split_adj, idx_split 

    return node_features, edge_indices, edge_features, node_labels, targets

def train_val_test_split(data, tt_state, tv_state):
    train_data = Repo_Dataset_IM(data.len())
    train_data.load()
    val_data = Repo_Dataset_IM(data.len())
    val_data.load()
    test_data = Repo_Dataset_IM(data.len())
    test_data.load()
    print("\nSplitting data .... \n")
    for r, repo in enumerate(data):
        index = list(np.where(repo.node_type == 2)[0])
        c_targets = repo.targets[repo.node_type == 2]
        c_idx_tv, c_idx_test, c_targets_tv, _ \
            = train_test_split(index, c_targets, stratify=c_targets,
                                test_size=0.2, random_state=tt_state, shuffle=True)         
        c_idx_train, c_idx_val, _, _ \
            = train_test_split(c_idx_tv, c_targets_tv, stratify=c_targets_tv,
                                test_size=0.2, random_state=tv_state, shuffle=True)         
        
        train_data[r].node_features, train_data[r].edge_indices, \
            train_data[r].edge_features, train_data[r].node_labels, train_data[r].targets \
                = create_split(repo, c_idx_train, c_idx_val, c_idx_test)
        
        val_data[r].node_features, val_data[r].edge_indices, \
            val_data[r].edge_features, val_data[r].node_labels, val_data[r].targets \
                = create_split(repo, c_idx_val, c_idx_train, c_idx_test)
        
        test_data[r].node_features, test_data[r].edge_indices, \
            test_data[r].edge_features, test_data[r].node_labels, test_data[r].targets \
                = create_split(repo, c_idx_test, c_idx_train, c_idx_val)

    return train_data, val_data, test_data

def normalize(mx):
	"""
		Row-normalize sparse matrix
		Code from https://github.com/williamleif/graphsage-simple/
	"""
	rowsum = np.array(mx.sum(1)) + 0.01
	r_inv = np.power(rowsum, -1).flatten()
	r_inv[np.isinf(r_inv)] = 0.
	r_mat_inv = sp.diags(r_inv)
	mx = r_mat_inv.dot(mx)
	return torch.tensor(mx)


