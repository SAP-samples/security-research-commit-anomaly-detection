import torch
import numpy as np
from src.utils.utils import get_anom_ratio
import sys, pickle, copy
from torch_geometric.utils import to_dense_adj
np.set_printoptions(threshold=sys.maxsize)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import warnings
warnings.simplefilter(action='ignore')

''' Type 1. including commits from one repo (2) to another (1): 
		a. including file and method nodes and edges from repo 2
		b. randomly assigning the branch and dev edges from repo 1 
        c. last commit of repo 1 is the parent of first injected commit
           and remaining injected commits have parent-child edges sequentially 
'''
def another_repo(repo_1_orig, repo_1, repo_2, p_anomals=0.02):    
    # Getting indices of commit nodes (node_type 2) and randomly selecting a few using p_anomals
    r1_c_indices = np.where(repo_1_orig.node_type == 2)[0]
    r2_c_indices = np.where(repo_2.node_type == 2)[0]
    anomals_ratio = int(len(r1_c_indices) * p_anomals) if int(len(r1_c_indices) * p_anomals) != 0 else 1

    anom_indices = copy.deepcopy(repo_1.node_type)
    while len(anom_indices) > (0.05 * repo_1.node_features.shape[0]):
        c_indices = np.random.choice(r2_c_indices, anomals_ratio, replace=False) 
        # Getting indices of connected file and method nodes and edge indices
        fm_indices, c_edge_indices, c_edge_features = [], [], []    
        for idx in c_indices:
            idx_edge_indices = repo_2.edge_indices[:, repo_2.edge_indices[0, :] == idx] 
            exclude_c_indices = [i for i in range(len(idx_edge_indices[1,:])) if idx_edge_indices[1,i] not in list(r2_c_indices)]
            idx_edge_indices_filtered = idx_edge_indices[:, exclude_c_indices]
            c_edge_indices.append(idx_edge_indices_filtered)
            
            idx_edge_features = repo_2.edge_features[repo_2.edge_indices[0, :] == idx] 
            idx_edge_features_filtered = idx_edge_features[exclude_c_indices, :]
            c_edge_features.append(idx_edge_features_filtered)        

            idx_fm_indices = idx_edge_indices_filtered[1,:]
            fm_indices.append(idx_fm_indices)
        c_edge_features = np.vstack(c_edge_features)
        c_edge_indices = np.hstack(c_edge_indices).astype(np.int)
        fm_indices = np.hstack(fm_indices)
        # Concatenating all new node indices to be injected as anomalies
        anom_indices = np.concatenate((c_indices, fm_indices))
        anom_indices = np.unique(anom_indices)
    # Getting nodes, node names and node types of anomalies to be injected
    anom_nodes = repo_2.node_features[anom_indices]
    anom_node_names = repo_2.node_names[anom_indices]
    anom_node_type = repo_2.node_type[anom_indices]
    # Injecting nodes, node names and node types
    r1_orig_n = repo_1_orig.node_features.shape[0]
    repo_1.node_features = torch.cat((repo_1.node_features, anom_nodes), dim=0)
    repo_1.node_names = np.concatenate((repo_1.node_names, anom_node_names))
    repo_1.node_type = torch.cat((repo_1.node_type, anom_node_type))
    # Transforming edge indices and edge features according to repo 1
    anom_edge_indices = c_edge_indices.copy()
    for n, o in zip(range(r1_orig_n, r1_orig_n + len(anom_indices)), anom_indices):
        anom_edge_indices[0, c_edge_indices[0,:] == o] = n
        anom_edge_indices[1, c_edge_indices[1,:] == o] = n

    # Randomly selecting branch and developer nodes and last commit as parent node from repo 1 for anomalies to be injected 
    b_indices = np.random.choice(np.where(repo_1_orig.node_type == 0)[0], anomals_ratio, replace=True) 
    b_anom_edge_indices = np.zeros((2, len(b_indices)), dtype=np.int)
    b_anom_edge_indices[0, :] = b_indices
    
    d_indices = np.random.choice(np.where(repo_1_orig.node_type == 1)[0], anomals_ratio, replace=True) 
    d_anom_edge_indices = np.zeros((2, len(d_indices)), dtype=np.int)
    d_anom_edge_indices[0, :] = d_indices
    
    p_indices = np.zeros(len(c_indices), dtype=np.int)
    p_indices[0] = np.where(repo_1_orig.node_type == 2)[0][-1] 
    p_anom_edge_indices = np.zeros((2, len(p_indices)), dtype=np.int)
    p_anom_edge_indices[0, 0] = p_indices[0]

    for n, o in zip(range(r1_orig_n, r1_orig_n + len(c_indices)), c_indices):
        b_anom_edge_indices[1, c_indices == o] = n
        d_anom_edge_indices[1, c_indices == o] = n
        p_anom_edge_indices[1, c_indices == o] = n
        p_anom_edge_indices[0, 1:][(c_indices == o)[:-1]] = n
    # b_anom_edge_features = np.random.random((len(b_indices), repo_1.edge_features.shape[1]))
    # b_anom_edge_features[:, repo_1.edge_features[(repo_1.edge_indices[0,:] == b_indices[0])][0,:] == 0] = 0.
    # d_anom_edge_features = np.random.random((len(d_indices), repo_1.edge_features.shape[1]))
    # d_anom_edge_features[:, repo_1.edge_features[(repo_1.edge_indices[0,:] == d_indices[0])][0,:] == 0] = 0.
    # p_anom_edge_features = np.random.random((len(p_indices), repo_1.edge_features.shape[1]))
    # p_anom_edge_features[:, repo_1.edge_features[(repo_1.edge_indices[0,:] == r1_c_indices[0])][0,:] == 0] = 0.

    b_anom_edge_features = np.zeros((len(b_indices), repo_1.edge_features.shape[1]))
    d_anom_edge_features = np.zeros((len(d_indices), repo_1.edge_features.shape[1]))
    p_anom_edge_features = np.zeros((len(p_indices), repo_1.edge_features.shape[1]))

    # Injecting edge indices and edge features
    repo_1.edge_indices = torch.cat((repo_1.edge_indices, torch.tensor(b_anom_edge_indices), \
                                        torch.tensor(d_anom_edge_indices), \
                                        torch.tensor(p_anom_edge_indices),\
                                        torch.tensor(anom_edge_indices)), dim=1)
    repo_1.edge_features = torch.cat((repo_1.edge_features, torch.tensor(b_anom_edge_features, dtype=torch.float32), \
                                        torch.tensor(d_anom_edge_features, dtype=torch.float32), \
                                        torch.tensor(p_anom_edge_features, dtype=torch.float32), \
                                        torch.tensor(c_edge_features, dtype=torch.float32)), dim=0)
    repo_1.adj = torch.zeros(repo_1.node_features.shape[0], repo_1.node_features.shape[0], dtype=torch.bool)
    repo_1.adj[repo_1.edge_indices[0,:], repo_1.edge_indices[1,:]] = 1
    # Adding Label nodes and Targets
    print("Number of Type 1 anomalous nodes to be injected: ", len(anom_indices) + len(d_indices))
    labels = torch.ones(len(anom_indices), dtype=torch.int8)
    targets = torch.ones(len(anom_indices), dtype=torch.int8)
    repo_1.node_labels = torch.cat((repo_1.node_labels, labels))
    repo_1.targets = torch.cat((repo_1.targets, targets))

    repo_1.node_labels[d_indices] = 1
    repo_1.targets[d_indices] = 1
    

def get_node_edge_indices(repo, c_indices, swap_nodes=None, anom_type=None):
    c_edge_indices_in, c_edge_indices_out, \
        c_edge_features_in, c_edge_features_out = [], [], [], []
    for i, idx in enumerate(c_indices):
        idx_edge_indices_in = repo.edge_indices[:, repo.edge_indices[1,:] == idx] 
        idx_edge_indices_out = repo.edge_indices[:, repo.edge_indices[0,:] == idx] 
        c_edge_indices_in.append(idx_edge_indices_in)
        c_edge_indices_out.append(idx_edge_indices_out)

        idx_edge_features_in = repo.edge_features[repo.edge_indices[1,:] == idx] 
        idx_edge_features_out = repo.edge_features[repo.edge_indices[0,:] == idx]
        
        if anom_type == 2 or anom_type == 3 or anom_type == 4:
            pcc_edges = [i for i in idx_edge_indices_in[0, :] if i in np.where(repo.node_type == 2)[0]]
            idx_edge_features_in[:, :19] = swap_nodes[i, :19]
            idx_edge_features_in[:, 62:64] = swap_nodes[i, 62:64]
            idx_edge_features_in[pcc_edges, :19] = 0.
            idx_edge_features_in[pcc_edges, 62:64] = 0.

            fm_edges = [i for i in idx_edge_indices_in[0, :] if i in np.where(repo.node_type == 3)[0]]
            idx_edge_features_out[:, :19] = swap_nodes[i, :19]
            idx_edge_features_out[:, 62:64] = swap_nodes[i, 62:64]     
            idx_edge_features_out[fm_edges, :19] = 0.   
            idx_edge_features_out[fm_edges, 62:64] = 0.   

        c_edge_features_in.append(idx_edge_features_in)
        c_edge_features_out.append(idx_edge_features_out)
        
    c_edge_indices_in = np.hstack(c_edge_indices_in)
    dup_edges = np.where(c_edge_indices_in[0,:] != c_edge_indices_in[1,:])[0]
    c_edge_indices_in = c_edge_indices_in[:, dup_edges]
    c_edge_indices_out = np.hstack(c_edge_indices_out)
    c_edge_indices = np.hstack((c_edge_indices_in, c_edge_indices_out))
    
    c_edge_features_in = np.vstack(c_edge_features_in)
    c_edge_features_in = c_edge_features_in[dup_edges, :]
    c_edge_features_out = np.vstack(c_edge_features_out)
    c_edge_features = np.vstack((c_edge_features_in, c_edge_features_out))

    return c_edge_indices, c_edge_features              

def inject_anomals(repo, c_indices, c_edge_indices, c_edge_features, anom_nodes, anom_type):
    orig_b_indices = np.where(repo.node_type == 0)[0]
    orig_d_indices = np.where(repo.node_type == 1)[0]
    orig_c_indices = np.where(repo.node_type == 2)[0]
    # Getting node names and node types of anomalies to be injected
    anom_node_names = repo.node_names[c_indices]
    anom_node_type = repo.node_type[c_indices]
        
    r_orig_n = repo.node_features.shape[0]
    repo.node_features = torch.cat((repo.node_features, anom_nodes), dim=0)
    repo.node_names = np.concatenate((repo.node_names, anom_node_names))
    repo.node_type = torch.cat((repo.node_type, anom_node_type))
    # Transforming edge indices and edge features according to new indices
    anom_edge_indices = c_edge_indices.copy()
    for n, o in zip(range(r_orig_n, r_orig_n + len(c_indices)), c_indices):
        anom_edge_indices[0, c_edge_indices[0,:] == o] = n
        anom_edge_indices[1, c_edge_indices[1,:] == o] = n
    # Injecting edge indices and edge features
    repo.edge_indices = torch.cat((repo.edge_indices, torch.tensor(anom_edge_indices)), dim=1)
    repo.edge_features = torch.cat((repo.edge_features, torch.tensor(c_edge_features)), dim=0)
    repo.adj = torch.zeros(repo.node_features.shape[0], repo.node_features.shape[0], dtype=torch.bool)
    repo.adj[repo.edge_indices[0,:], repo.edge_indices[1,:]] = 1
    # Adding Label nodes and Targets
    labels = torch.ones(len(anom_nodes), dtype=torch.int8) * anom_type
    targets = torch.ones(len(anom_nodes), dtype=torch.int8)
    repo.node_labels = torch.cat((repo.node_labels, labels))
    repo.targets = torch.cat((repo.targets, targets))

    all_anom_node_indices = np.unique(anom_edge_indices.flatten())    
    dfm_anom_node_indices = [i for i in list(all_anom_node_indices) \
                            if (i not in list(orig_b_indices)) \
                                # & (i not in list(orig_d_indices)) \
                                    & (i not in list(orig_c_indices))]
    print("Number of Type "+str(anom_type)+" anomalous nodes to be injected: ", len(dfm_anom_node_indices))
    repo.node_labels[dfm_anom_node_indices] = anom_type
    repo.targets[dfm_anom_node_indices] = 1

    return repo

''' Type 2 & 3. add commits with completely garbled random node features or garbled timestamp features: 
		a. including the same branch, dev, file and method edges
'''
def garbled_features(repo_orig, repo, timestamp=False, commit_message=False, p_anomals=0.02, sampling=False):    
    # Getting indices of commit nodes (node_type 2) and randomly selecting a few using p_anomals
    r_c_indices = np.where(repo_orig.node_type == 2)[0]
    anomals_ratio = int(len(r_c_indices) * p_anomals) if int(len(r_c_indices) * p_anomals) != 0 else 1
    c_edge_features = copy.deepcopy(repo.edge_features)
    while (c_edge_features.shape[0] > (0.33) * repo.node_features.shape[0]):# \
        # | (c_edge_features.shape[0] < 0.02 * repo.node_features.shape[0]):
        c_indices = np.random.choice(r_c_indices, anomals_ratio, replace=False)

        swap_r_c_indices = [i for i in list(r_c_indices) if i not in list(c_indices)]
        swap_c_indices = np.random.choice(swap_r_c_indices, anomals_ratio, replace=False)
        anom_nodes = repo.node_features[c_indices]
        swap_nodes = repo.node_features[swap_c_indices]
        
        # Getting indices of connected branch, dev, file and method nodes and edge indices
        c_edge_indices, c_edge_features = get_node_edge_indices(repo, c_indices)
    
    if sampling:
        c_edge_features_new = copy.deepcopy(c_edge_features)
        temp = c_edge_features_new[c_edge_features != 0]
        half = int(len(temp) / 2)
        mean_edge_features = torch.mean(repo_orig.edge_features)
        stdev_edge_features = torch.std(repo_orig.edge_features)
        while np.mean(temp[:half]) < (mean_edge_features + 2*stdev_edge_features) \
            or np.abs(np.mean(temp[half:])) < abs(mean_edge_features - 2*stdev_edge_features): 
            temp[:half] = np.random.normal(mean_edge_features + 2*stdev_edge_features, \
                                            stdev_edge_features, temp[:half].shape)
            temp[half:] = np.random.normal(mean_edge_features - 2*stdev_edge_features, \
                                            stdev_edge_features, temp[half:].shape)
            c_edge_features_new[c_edge_features != 0] = temp
        
        swap_nodes = copy.deepcopy(anom_nodes)
        temp = swap_nodes[swap_nodes != 0]
        half = int(len(temp) / 2)
        mean_node_features = torch.mean(repo_orig.node_features)
        stdev_node_features = torch.std(repo_orig.node_features)
        while torch.mean(temp[:half]) < (mean_node_features + 2*stdev_node_features) \
            or np.abs(torch.mean(temp[half:])) < abs(mean_node_features - 2*stdev_node_features): 
            temp[:half] = torch.normal(mean_node_features + 2*stdev_node_features, \
                                        stdev_node_features, temp[:half].shape)
            temp[half:] = torch.normal(mean_node_features - 2*stdev_node_features, \
                                        stdev_node_features, temp[half:].shape)
            swap_nodes[swap_nodes != 0] = temp    
    
    if not timestamp and not commit_message:
        anom_type = 2
        ''' Garbling node features of commit nodes to be injected ''' 
        anom_nodes = swap_nodes        
        if sampling:
            c_edge_features = c_edge_features_new
        # else:
        #     _, c_edge_features = get_node_edge_indices(repo, c_indices, swap_nodes, anom_type)
        # c_edge_features[:, :19] = 0.
        # c_edge_features[:, 62:64] = 0.
    elif timestamp:
        anom_type = 3
        ''' Garbling timestamp features of commit nodes to be injected '''
        anom_nodes[:, :19] = swap_nodes[:, :19]
        anom_nodes[:, 31:33] = swap_nodes[:, 31:33]
        if sampling:
            c_edge_features[:, :19] = c_edge_features_new[:, :19]
            c_edge_features[:, 62:64] = c_edge_features_new[:, 62:64]
        # else:
        #     _, c_edge_features = get_node_edge_indices(repo, c_indices, swap_nodes, anom_type)
    elif commit_message:
        anom_type = 4
        ''' Swapping commit_message features of commit nodes to be injected '''           
        anom_nodes[:, (repo.node_features.shape[1] - 100):] = swap_nodes[:, (repo.node_features.shape[1] - 100):]
        # anom_nodes[:, 19:31] = swap_nodes[:, 19:31]
        # anom_nodes[:, 33:] = swap_nodes[:, 33:]
        
        # else:
        #     _, c_edge_features = get_node_edge_indices(repo, c_indices, swap_nodes, anom_type)
    # Injecting nodes, node names and node types
    repo = inject_anomals(repo, c_indices, c_edge_indices, c_edge_features, anom_nodes, anom_type)


def dist_matrix(node_features, p_anomals):
    M = node_features.shape[0]
    dist_mat = torch.zeros(M, M)
    for m in range(M):
        dist_mat[m, :] = torch.norm((node_features[m, :] - node_features), dim=1)
    if M > 1:
        anomals_ratio = int(M * p_anomals) if int(M * p_anomals) != 0 else 1
    else:
        anomals_ratio = 0
    topk_indices = torch.topk(dist_mat.flatten(), anomals_ratio)[1]

    swap_indices = torch.zeros(2, anomals_ratio).type(torch.LongTensor)
    swap_indices[0,:] = (topk_indices / M).type(torch.LongTensor)
    swap_indices[1,:] = (topk_indices % M).type(torch.LongTensor)

    return dist_mat, swap_indices
    
''' Type 5. swap features of two nodes that are maximum Euclidean distance apart in the repository:
		a. can be done for all types of nodes
		b. should be swapped between same type of nodes i.e. commit with commit and so on
'''
def swapping_features(repo_orig, repo, p_anomals=0.05):
    
    for type in np.unique(repo_orig.node_type.numpy()):            
        dist_mat, swap_indices = dist_matrix(repo_orig.node_features[repo_orig.node_type == type], p_anomals)
        if swap_indices.nelement() != 0: 
            print("Number of Type 5 anomalous nodes to be injected (node type: ", type, "): ", swap_indices.shape[1])
            repo_copy = copy.deepcopy(repo)
            repo.node_features[swap_indices[0,:], :] = copy.deepcopy(repo_copy.node_features[swap_indices[1,:], :])
            anom_nodes = np.unique(swap_indices[0,:].numpy())

            # Adding Label nodes and Targets                
            repo.node_labels[anom_nodes] = 5
            repo.targets[anom_nodes] = 1
            

def inject_anomalies(urls_list, path_process_graph, path_inject_anom, \
                        p_anomals1=0.015, p_anomals2=0.05, p_anomals3=0.05, p_anomals4=0.05, p_anomals5=0.05, \
                        all_types=True, type1=False, type2=False, type3=False, type4=False, type5=False, sampling=False):
    torch.manual_seed(1234)
    np.random.seed(1234)
    anom_ratio_before, anom_ratio_after1, anom_ratio_after2, \
        anom_ratio_after3, anom_ratio_after4, anom_ratio_after5 = [], [], [], [], [], []
    ''' Loading processed graphs '''
    for u, url in enumerate(urls_list):
        print("Repo: ", url, " No:", u+1)
        repo_1 = pickle.load(open(path_process_graph + "/repo_graph_"+url.split('/')[-1]+".pkl", "rb"))        
        repo_1_orig = copy.deepcopy(repo_1) 
        print("Number of nodes in the repo: ", repo_1.node_features.shape[0])

        init_ar = get_anom_ratio(repo_1)
        anom_ratio_before.append(init_ar)
        print("Inital Anomaly Ratio: ", init_ar)
        
        if sampling:
            type_str = '_s'
        else:
            type_str = ''
        if all_types or type1:
            ''' Injecting Type 1 Anomalies''' 
            u_2 =  u+1 if u < len(urls_list)-1 else 0 
            url_2 = urls_list[int(u_2)]
            repo_2 = pickle.load(open(path_process_graph + "/repo_graph_"+url_2.split('/')[-1]+".pkl", "rb"))           
            another_repo(repo_1_orig, repo_1, repo_2, p_anomals=p_anomals1)
            ar_after1 = get_anom_ratio(repo_1)
            print("Anomaly Ratio after injecting Type 1 anomalies: ", ar_after1)
            anom_ratio_after1.append(ar_after1)
            if not all_types and type1:
                type_str += '_1'             
        
        if all_types or type2:
            ''' Injecting Type 2 Anomalies'''
            garbled_features(repo_1_orig, repo_1, timestamp=False, p_anomals=p_anomals2, sampling=sampling)
            ar_after2 = get_anom_ratio(repo_1)
            print("Anomaly Ratio after injecting Type 2 anomalies: ", ar_after2)
            anom_ratio_after2.append(ar_after2)
            if not all_types and type2:
                type_str += '_2'             

        if all_types or type3:
            ''' Injecting Type 3 Anomalies'''
            garbled_features(repo_1_orig, repo_1, timestamp=True, p_anomals=p_anomals3, sampling=sampling)
            ar_after3 = get_anom_ratio(repo_1)
            print("Anomaly Ratio after injecting Type 3 anomalies: ", ar_after3)
            anom_ratio_after3.append(ar_after3)
            if not all_types and type3:
                type_str += '_3'             

        if all_types or type4:
            ''' Injecting Type 4 Anomalies'''
            garbled_features(repo_1_orig, repo_1, commit_message=True, p_anomals=p_anomals4, sampling=sampling)
            ar_after4 = get_anom_ratio(repo_1)
            print("Anomaly Ratio after injecting Type 4 anomalies: ", ar_after4)
            anom_ratio_after4.append(ar_after4)
            if not all_types and type4:
                type_str += '_4'   
        
        if all_types or type5:
            ''' Injecting Type 4 Anomalies'''
            swapping_features(repo_1_orig, repo_1, p_anomals=p_anomals5)
            ar_after5 = get_anom_ratio(repo_1)
            print("Anomaly Ratio after injecting Type 5 anomalies: ", ar_after5)
            anom_ratio_after5.append(ar_after5)
            if not all_types and type5:
                type_str += '_5'   

        pickle.dump(repo_1,  open(path_inject_anom + "/repo_graph_anom"+type_str+"_"+url.split('/')[-1]+".pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
        print("Anomalies injected and repo saved\n")

    print("Mean Inital Anomaly Ratio: ", np.round(np.mean(anom_ratio_before), 2))
    if all_types or type1:
        print("Mean Anomaly Ratio after injecting Type 1 anomalies: ", np.round(np.mean(anom_ratio_after1), 2))
    if all_types or type2:
        print("Mean Anomaly Ratio after injecting Type 2 anomalies: ", np.round(np.mean(anom_ratio_after2), 2))
    if all_types or type3:
        print("Mean Anomaly Ratio after injecting Type 3 anomalies: ", np.round(np.mean(anom_ratio_after3), 2))
    if all_types or type4:
        print("Mean Anomaly Ratio after injecting Type 4 anomalies: ", np.round(np.mean(anom_ratio_after4), 2))
    if all_types or type5:
        print("Mean Anomaly Ratio after injecting Type 5 anomalies: ", np.round(np.mean(anom_ratio_after5), 2))
    print("Done")


    

