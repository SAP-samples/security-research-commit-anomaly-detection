import networkx as nx
import pygraphviz as pgv
from IPython.core.display import display
import torch_geometric
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore')

def print_graph_stats(GR_GAD_df, node_type, features, edge_features, labels, g, url, _print=False):
    n_features = features.shape[1]
    n_edge_features = edge_features.shape[1]

    n_nodes = len(labels)
    branches = len(node_type[node_type == 0])
    developers = len(node_type[node_type == 1])
    commits = len(node_type[node_type == 2])
    files = len(node_type[node_type == 3])
    methods = len(node_type[node_type == 4])

    n_norm_nodes = len(labels[labels == 0])
    n_anom_nodes = len(labels[labels == 1])
    anom_ratio = np.round((n_anom_nodes / n_nodes) * 100, 2)

    t_edges = (n_nodes**2)
    n_edges = (edge_features.shape[0])
    sparsity = np.round((n_edges / t_edges) * 100, 2)

    GR_GAD_df.loc[g] \
        = [url, branches, developers, commits, files, methods, \
            n_anom_nodes, anom_ratio, n_edges, t_edges, sparsity]

    if _print:
        print("\nTotal number of node features: ", n_features)
        print("Total number of edge features: ", n_edge_features)
        print("Total number of nodes: ", n_nodes)
        print("Total number of normal nodes: ", n_norm_nodes)

        print("Total number of branch nodes: ", branches)
        print("Total number of developer nodes: ", developers)
        print("Total number of commit nodes: ", commits)
        print("Total number of file nodes: ", files)
        print("Total number of method nodes: ", methods)
        print("Total number of anomalous nodes: ", n_anom_nodes)
        print("Percentage of anomalous nodes: ", anom_ratio)
        print("Total number of possible edges: ", t_edges)
        print("Total number of actual edges: ", n_edges)
        print("Sparsity: ", sparsity)
    
    return GR_GAD_df, n_features, n_nodes, anom_ratio, t_edges, n_edges, sparsity, n_edge_features

def create_edge_dict(outliers, p_class, node_names):
    outlier_edge_names = np.empty(outliers.shape, dtype=object)
    edge_dict = {'class': p_class, \
                    'bb_edge': 0, 'bd_edge': 0, 'bc_edge': 0, 'bf_edge': 0, 'bm_edge': 0,\
                    'db_edge': 0, 'dd_edge': 0, 'dc_edge': 0, 'df_edge': 0, 'dm_edge': 0, \
                    'cb_edge': 0, 'cd_edge': 0, 'cc_edge': 0, 'cf_edge': 0, 'cm_edge': 0, \
                    'fb_edge': 0, 'fd_edge': 0, 'fc_edge': 0, 'ff_edge': 0, 'fm_edge': 0, \
                    'mb_edge': 0, 'md_edge': 0, 'mc_edge': 0, 'mf_edge': 0, 'mm_edge': 0}
    for e in range(outliers.size()[1]):
        outlier_edge_names[0,e] = node_names[outliers[0,e]]
        outlier_edge_names[1,e] = node_names[outliers[1,e]]
        # edge_dict['n_edges'] += 1  
        if 'branch_' in str(outlier_edge_names[0,e]):
            if 'dev_' in str(outlier_edge_names[1,e]):
                edge_dict['bd_edge'] += 1  
            elif 'commit_' in str(outlier_edge_names[1,e]):
                edge_dict['bc_edge'] += 1  
            elif 'file_' in str(outlier_edge_names[1,e]):
                edge_dict['bf_edge'] += 1  
            elif 'method_' in str(outlier_edge_names[1,e]):
                edge_dict['bm_edge'] += 1
            elif 'branch_' in str(outlier_edge_names[1,e]):
                edge_dict['bb_edge'] += 1  
        elif 'dev_' in str(outlier_edge_names[0,e]):
            if 'branch_' in str(outlier_edge_names[1,e]):
                edge_dict['db_edge'] += 1  
            elif 'commit_' in str(outlier_edge_names[1,e]):
                edge_dict['dc_edge'] += 1  
            elif 'file_' in str(outlier_edge_names[1,e]):
                edge_dict['df_edge'] += 1  
            elif 'method_' in str(outlier_edge_names[1,e]):
                edge_dict['dm_edge'] += 1
            elif 'dev_' in str(outlier_edge_names[1,e]):
                edge_dict['dd_edge'] += 1  
        elif 'commit_' in str(outlier_edge_names[0,e]):
            if 'branch_' in str(outlier_edge_names[1,e]):
                edge_dict['cb_edge'] += 1 
            elif 'dev_' in str(outlier_edge_names[1,e]):
                edge_dict['cd_edge'] += 1  
            elif 'file_' in str(outlier_edge_names[1,e]):
                edge_dict['cf_edge'] += 1  
            elif 'method_' in str(outlier_edge_names[1,e]):
                edge_dict['cm_edge'] += 1  
            elif 'commit_' in str(outlier_edge_names[1,e]):
                edge_dict['cc_edge'] += 1 
        elif 'file_' in str(outlier_edge_names[0,e]):
            if 'branch_' in str(outlier_edge_names[1,e]):
                edge_dict['fb_edge'] += 1 
            elif 'dev_' in str(outlier_edge_names[1,e]):
                edge_dict['fd_edge'] += 1  
            elif 'commit_' in str(outlier_edge_names[1,e]):
                edge_dict['fc_edge'] += 1  
            elif 'method_' in str(outlier_edge_names[1,e]):
                edge_dict['fm_edge'] += 1  
            elif 'file_' in str(outlier_edge_names[1,e]):
                edge_dict['ff_edge'] += 1 
        elif 'method_' in str(outlier_edge_names[0,e]):
            if 'branch_' in str(outlier_edge_names[1,e]):
                edge_dict['mb_edge'] += 1 
            elif 'dev_' in str(outlier_edge_names[1,e]):
                edge_dict['md_edge'] += 1  
            elif 'commit_' in str(outlier_edge_names[1,e]):
                edge_dict['mc_edge'] += 1  
            elif 'file_' in str(outlier_edge_names[1,e]):
                edge_dict['mf_edge'] += 1  
            elif 'method_' in str(outlier_edge_names[1,e]):
                edge_dict['mm_edge'] += 1 
        
    return outlier_edge_names, edge_dict

def get_edge_names(adj_pred, adj, node_names, p_class, out_f):
    adj_pred_copy = np.where(adj == p_class, adj_pred.detach().cpu().numpy(), 100)
    if p_class == 1:
        outliers = torch.tensor(np.argwhere((adj_pred_copy <= 0.5) & (adj_pred_copy != 100)).T)
        tp_edges = np.argwhere(adj_pred_copy != 100).T.shape[1]
        fp_edges = outliers.shape[1]
    elif p_class == 0:
        outliers = torch.tensor(np.argwhere((adj_pred_copy > 0.5) & (adj_pred_copy != 100)).T)
        tn_edges = np.argwhere(adj_pred_copy != 100).T.shape[1]
        fn_edges = outliers.shape[1]

    t_edges = torch.tensor(np.argwhere(adj_pred_copy != 100).T)
    outlier_edge_names, edge_dict = create_edge_dict(outliers, p_class, node_names)
    t_edge_names, t_edge_dict = create_edge_dict(t_edges, p_class, node_names)

    for (k1,v1), (k2,v2) in zip(edge_dict.items(), t_edge_dict.items()):
        edge_dict[k1] = [v1, np.round((v1/(v2 + 1e-20)), 2)] 
    
    print("Total No. of Class "+str(p_class)+" Edges: ", np.argwhere(adj_pred_copy != 100).T.shape[1], \
            "\nFalsely predicted No. of Class "+str(p_class)+" Edges: ", outliers.shape[1], file=out_f)#, \
    
    return outlier_edge_names, edge_dict

def get_tpfp(adj_pred, adj, p_class, threshold):
    adj_pred_copy = np.where(adj == p_class, adj_pred.detach().cpu().numpy(), 100)
    if p_class == 1:
        outliers = torch.tensor(np.argwhere((adj_pred_copy <= threshold) & (adj_pred_copy != 100)).T)
        f_edges = outliers.shape[1]
        t_edges = np.argwhere(adj_pred_copy != 100).T.shape[1] - f_edges
    elif p_class == 0:
        outliers = torch.tensor(np.argwhere((adj_pred_copy > threshold) & (adj_pred_copy != 100)).T)
        f_edges = outliers.shape[1]
        t_edges = np.argwhere(adj_pred_copy != 100).T.shape[1] - f_edges

    return t_edges, f_edges


def create_edge_dict_hetero(outliers, p_class, node_names_row, node_names_col):
    outlier_edge_names = np.empty(outliers.shape, dtype=object)
    edge_dict = {'class': p_class, \
                    'bb_edge': 0, 'bd_edge': 0, 'bc_edge': 0, 'bf_edge': 0, 'bm_edge': 0,\
                    'db_edge': 0, 'dd_edge': 0, 'dc_edge': 0, 'df_edge': 0, 'dm_edge': 0, \
                    'cb_edge': 0, 'cd_edge': 0, 'cc_edge': 0, 'cf_edge': 0, 'cm_edge': 0, \
                    'fb_edge': 0, 'fd_edge': 0, 'fc_edge': 0, 'ff_edge': 0, 'fm_edge': 0, \
                    'mb_edge': 0, 'md_edge': 0, 'mc_edge': 0, 'mf_edge': 0, 'mm_edge': 0}
    for e in range(outliers.size()[1]):
        outlier_edge_names[0,e] = node_names_row[outliers[0,e]]
        outlier_edge_names[1,e] = node_names_col[outliers[1,e]]
        # edge_dict['n_edges'] += 1  
        if 'branch_' in str(outlier_edge_names[0,e]):
            if 'dev_' in str(outlier_edge_names[1,e]):
                edge_dict['bd_edge'] += 1  
            elif 'commit_' in str(outlier_edge_names[1,e]):
                edge_dict['bc_edge'] += 1  
            elif 'file_' in str(outlier_edge_names[1,e]):
                edge_dict['bf_edge'] += 1  
            elif 'method_' in str(outlier_edge_names[1,e]):
                edge_dict['bm_edge'] += 1
            elif 'branch_' in str(outlier_edge_names[1,e]):
                edge_dict['bb_edge'] += 1  
        elif 'dev_' in str(outlier_edge_names[0,e]):
            if 'branch_' in str(outlier_edge_names[1,e]):
                edge_dict['db_edge'] += 1  
            elif 'commit_' in str(outlier_edge_names[1,e]):
                edge_dict['dc_edge'] += 1  
            elif 'file_' in str(outlier_edge_names[1,e]):
                edge_dict['df_edge'] += 1  
            elif 'method_' in str(outlier_edge_names[1,e]):
                edge_dict['dm_edge'] += 1
            elif 'dev_' in str(outlier_edge_names[1,e]):
                edge_dict['dd_edge'] += 1  
        elif 'commit_' in str(outlier_edge_names[0,e]):
            if 'branch_' in str(outlier_edge_names[1,e]):
                edge_dict['cb_edge'] += 1 
            elif 'dev_' in str(outlier_edge_names[1,e]):
                edge_dict['cd_edge'] += 1  
            elif 'file_' in str(outlier_edge_names[1,e]):
                edge_dict['cf_edge'] += 1  
            elif 'method_' in str(outlier_edge_names[1,e]):
                edge_dict['cm_edge'] += 1  
            elif 'commit_' in str(outlier_edge_names[1,e]):
                edge_dict['cc_edge'] += 1 
        elif 'file_' in str(outlier_edge_names[0,e]):
            if 'branch_' in str(outlier_edge_names[1,e]):
                edge_dict['fb_edge'] += 1 
            elif 'dev_' in str(outlier_edge_names[1,e]):
                edge_dict['fd_edge'] += 1  
            elif 'commit_' in str(outlier_edge_names[1,e]):
                edge_dict['fc_edge'] += 1  
            elif 'method_' in str(outlier_edge_names[1,e]):
                edge_dict['fm_edge'] += 1  
            elif 'file_' in str(outlier_edge_names[1,e]):
                edge_dict['ff_edge'] += 1 
        elif 'method_' in str(outlier_edge_names[0,e]):
            if 'branch_' in str(outlier_edge_names[1,e]):
                edge_dict['mb_edge'] += 1 
            elif 'dev_' in str(outlier_edge_names[1,e]):
                edge_dict['md_edge'] += 1  
            elif 'commit_' in str(outlier_edge_names[1,e]):
                edge_dict['mc_edge'] += 1  
            elif 'file_' in str(outlier_edge_names[1,e]):
                edge_dict['mf_edge'] += 1  
            elif 'method_' in str(outlier_edge_names[1,e]):
                edge_dict['mm_edge'] += 1 
        
    return outlier_edge_names, edge_dict

def get_edge_names_one(adj_pred, adj, p_class, node_names_row, node_names_col):
    adj_pred_copy = np.where(adj == p_class, adj_pred.detach().cpu().numpy(), 100)
    if p_class == 1:
        outliers = torch.tensor(np.argwhere((adj_pred_copy <= 0.5) & (adj_pred_copy != 100)).T)
    elif p_class == 0:
        outliers = torch.tensor(np.argwhere((adj_pred_copy > 0.5) & (adj_pred_copy != 100)).T)

    t_edges = torch.tensor(np.argwhere(adj_pred_copy != 100).T)
    outlier_edge_names, edge_dict = create_edge_dict_hetero(outliers, p_class, node_names_row, node_names_col)
    t_edge_names, t_edge_dict = create_edge_dict_hetero(t_edges, p_class, node_names_row, node_names_col)

    # for (k1,v1), (k2,v2) in zip(edge_dict.items(), t_edge_dict.items()):
    #     edge_dict[k1] = [v1, np.round((v1/(v2 + 1e-20)), 2)] 
    
    t_class_edges = np.argwhere(adj_pred_copy != 100).T.shape[1]
    fp_class_edges = outliers.shape[1]    
    
    return outlier_edge_names, edge_dict, t_edge_dict, t_class_edges, fp_class_edges

def get_edge_names_hetero(bc_adj_pred, bc_adj, cb_adj_pred, cb_adj, \
                    dc_adj_pred, dc_adj, cd_adj_pred, cd_adj,  \
                    cc_adj_pred, cc_adj, cf_adj_pred, cf_adj,  \
                    cm_adj_pred, cm_adj, fm_adj_pred, fm_adj, \
                    branch_names, dev_names, commit_names, \
                    file_names, method_names, p_class, out_f):
    N = 8
    outlier_edge_names, edge_dict, t_edge_dict, t_class_edges, fp_class_edges \
        = np.empty(N, dtype=object), np.empty(N, dtype=object), \
            np.empty(N, dtype=object), np.empty(N, dtype=object), np.empty(N, dtype=object)

    outlier_edge_names[0], edge_dict[0], t_edge_dict[0], t_class_edges[0], fp_class_edges[0] \
        = get_edge_names_one(bc_adj_pred, bc_adj, p_class, branch_names, commit_names)
    outlier_edge_names[1], edge_dict[1], t_edge_dict[1], t_class_edges[1], fp_class_edges[1] \
        = get_edge_names_one(dc_adj_pred, dc_adj, p_class, dev_names, commit_names)
    outlier_edge_names[2], edge_dict[2], t_edge_dict[2], t_class_edges[2], fp_class_edges[2] \
        = get_edge_names_one(cc_adj_pred, cc_adj, p_class, commit_names, commit_names)
    outlier_edge_names[3], edge_dict[3], t_edge_dict[3], t_class_edges[3], fp_class_edges[3] \
        = get_edge_names_one(cf_adj_pred, cf_adj, p_class, commit_names, file_names)
    outlier_edge_names[4], edge_dict[4], t_edge_dict[4], t_class_edges[4], fp_class_edges[4] \
        = get_edge_names_one(cm_adj_pred, cm_adj, p_class, commit_names, method_names)
    outlier_edge_names[5], edge_dict[5], t_edge_dict[5], t_class_edges[5], fp_class_edges[5] \
        = get_edge_names_one(fm_adj_pred, fm_adj, p_class, file_names, method_names)
    outlier_edge_names[6], edge_dict[6], t_edge_dict[6], t_class_edges[6], fp_class_edges[6] \
        = get_edge_names_one(cb_adj_pred, cb_adj, p_class, commit_names, branch_names)
    outlier_edge_names[7], edge_dict[7], t_edge_dict[7], t_class_edges[7], fp_class_edges[7] \
        = get_edge_names_one(cd_adj_pred, cd_adj, p_class, commit_names, dev_names)

    outlier_edge_names = np.concatenate(list(outlier_edge_names), axis=1)
    all_edge_dict, all_t_edge_dict = edge_dict[0].copy(), t_edge_dict[0].copy()
    for j in range(1,len(edge_dict)):
        for k in edge_dict[j]:
            all_edge_dict[k] += edge_dict[j][k]
            all_t_edge_dict[k] += t_edge_dict[j][k]
    
    for (k1,v1), (k2,v2) in zip(all_edge_dict.items(), all_t_edge_dict.items()):
        all_edge_dict[k1] = [v1, np.round((v1/(v2 + 1e-20)), 2)] 
    
    print("Total No. of Class "+str(p_class)+" Edges: ", np.sum(t_class_edges), \
            "\nFalsely predicted No. of Class "+str(p_class)+" Edges: ", np.sum(fp_class_edges), file=out_f)#, \
    
    return outlier_edge_names, all_edge_dict


# nodes_df = pd.read_pickle("working/graphdata/nodes_df.pkl")
# edge_indices = np.load("working/graphdata/edge_indices.npy")
# edge_names = np.load("working/graphdata/edge_names.npy", allow_pickle=True)
# edge_features_df = pd.read_pickle("working/graphdata/edge_features_df.pkl")

# display(nodes_df.dtypes.unique())

def plot_simple_graph(nodes_df, edge_indices):
    nodes_float_df = nodes_df.select_dtypes(include='float64')
    nodes_object_df = nodes_df.select_dtypes(include='object')
    nodes_onehot_df = pd.get_dummies(nodes_object_df,drop_first=True)

    # print(nodes_float_df.shape, nodes_object_df.shape, nodes_onehot_df.shape)

    nodes = torch.tensor(pd.concat([nodes_float_df, nodes_onehot_df], axis=1).to_numpy())
    edge_indices = torch.tensor(edge_indices)
    # print(nodes.size(), edge_indices.size())

    data = torch_geometric.data.Data(x=nodes, edge_index=edge_indices)
    g = torch_geometric.utils.to_networkx(data, to_undirected=False)

    nx.draw(g, with_labels=True, font_weight='bold')
    plt.savefig("working/graphdata/simple_graph.png")
    print("\nSimple graph plot saved at working/graphdata/simple_graph.png")


'''Heterogenous'''

# bc_edge_indices_hetero = np.load("working/graphdata/bc_edge_indices_hetero.npy")
# dc_edge_indices_hetero = np.load("working/graphdata/dc_edge_indices_hetero.npy")
# cc_edge_indices_hetero = np.load("working/graphdata/cc_edge_indices_hetero.npy")
# cf_edge_indices_hetero = np.load("working/graphdata/cf_edge_indices_hetero.npy")
# cm_edge_indices_hetero = np.load("working/graphdata/cm_edge_indices_hetero.npy")
# fm_edge_indices_hetero = np.load("working/graphdata/fm_edge_indices_hetero.npy")

def plot_hetero_graph(nodes_df, bc_edge_indices_hetero, dc_edge_indices_hetero, cc_edge_indices_hetero, 
                cf_edge_indices_hetero, cm_edge_indices_hetero, fm_edge_indices_hetero):
    ag = pgv.AGraph(strict=False, directed=True)
    
    for i,j in zip(bc_edge_indices_hetero[0,:], bc_edge_indices_hetero[1,:]):
        ag.add_node('branch: '+nodes_df[nodes_df.branch_hash.notnull()].branch_name.iloc[i], color='green')
        ag.add_node('commit: '+nodes_df[nodes_df.commit_hash.notnull()].commit_commit_hash.iloc[j], color='blue')
        ag.add_edge('branch: '+nodes_df[nodes_df.branch_hash.notnull()].branch_name.iloc[i], 
                    'commit: '+nodes_df[nodes_df.commit_hash.notnull()].commit_commit_hash.iloc[j], color='green')

    for i,j in zip(dc_edge_indices_hetero[0,:], dc_edge_indices_hetero[1,:]):
        ag.add_node('dev: '+nodes_df[nodes_df.dev_hash.notnull()].dev_name.iloc[i], color='red')
        ag.add_edge('dev: '+nodes_df[nodes_df.dev_hash.notnull()].dev_name.iloc[i], 
                    'commit: '+nodes_df[nodes_df.commit_hash.notnull()].commit_commit_hash.iloc[j], color='red')
    
    for i,j in zip(cc_edge_indices_hetero[0,:], cc_edge_indices_hetero[1,:]):
        ag.add_edge('commit: '+nodes_df[nodes_df.commit_hash.notnull()].commit_commit_hash.iloc[i], 
                    'commit: '+nodes_df[nodes_df.commit_hash.notnull()].commit_commit_hash.iloc[j], color='blue')
    
    for i,j in zip(cf_edge_indices_hetero[0,:], cf_edge_indices_hetero[1,:]):
        ag.add_node('file: '+nodes_df[nodes_df.file_hash.notnull()].file_name.iloc[j], color='brown')
        ag.add_edge('commit: '+nodes_df[nodes_df.commit_hash.notnull()].commit_commit_hash.iloc[i], 
                    'file: '+nodes_df[nodes_df.file_hash.notnull()].file_name.iloc[j], color='cyan')
    
    for i,j in zip(cm_edge_indices_hetero[0,:], cm_edge_indices_hetero[1,:]):
        ag.add_node('method: '+nodes_df[nodes_df.method_hash.notnull()].method_name.iloc[j], color='orange')
        ag.add_edge('commit: '+nodes_df[nodes_df.commit_hash.notnull()].commit_commit_hash.iloc[i], 
                    'method: '+nodes_df[nodes_df.method_hash.notnull()].method_name.iloc[j], color='purple')
    
    for i,j in zip(fm_edge_indices_hetero[0,:], fm_edge_indices_hetero[1,:]):
        ag.add_edge('file: '+nodes_df[nodes_df.file_hash.notnull()].file_name.iloc[i], 
                    'method: '+nodes_df[nodes_df.method_hash.notnull()].method_name.iloc[j], color='brown')
    
    ag.layout('dot')
    ag.draw('working/graphdata/hetero_graph.png')
    print("\nHeterogeneous graph plot saved at working/graphdata/hetero_graph.png")

# plot_graph(nodes_df, bc_edge_indices_hetero, dc_edge_indices_hetero, cc_edge_indices_hetero, 
            # cf_edge_indices_hetero, cm_edge_indices_hetero, fm_edge_indices_hetero)


# from pydriller import RepositoryMining
# from git import Repo
# path = "repos/python-dictionaries-readme-ds-apply-000"
# remote_path = "https://github.com/learn-co-students/python-dictionaries-readme-ds-apply-000.git"
# r = Repo(path)
# repo_branches = r.git.branch('-r')
# repo_branches = repo_branches.split('\n')[1:]
# # repo_branches[0] = r.git.branch().split()[-1]
# print(repo_branches)
# for branch in repo_branches:
#     branch = branch.split(' ')[2:]
#     print(branch)
#     # branch = '/'.join(branch)
#     for commit in RepositoryMining(path_to_repo=remote_path, only_in_branch=branch).traverse_commits():
# # for commit in RepositoryMining(path_to_repo=path).traverse_commits():
#             print(commit.hash, commit.branches)