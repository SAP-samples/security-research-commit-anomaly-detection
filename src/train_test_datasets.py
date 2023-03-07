import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch_geometric.utils import subgraph
from src.models.deepSAD_model import DeepSAD_GVAE
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import time, datetime, pickle
from skopt import gp_minimize
from functools import partial
from tqdm import tqdm
from torch.optim import Adam, lr_scheduler
from src.utils.utils import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_GVAE(adj_lists:np.array, features:np.array, labels:np.array, n_epoch:int, b_min:bool, out_f:str, \
                name:str, save_dir:str, hp:list, tv_state:int=2):

    batch_size, learn_r, hidden_features, n_layers, n_aggr, alpha, dropout = hp


    index = list(range(len(labels)))
    idx_train, idx_val, y_train, y_val = train_test_split(index, labels, stratify=labels, test_size=0.20,
                                                            random_state=tv_state, shuffle=True)
    print(f"tv_random_state: {tv_state}, train_features_mean: {features[idx_train].mean().item()}, \
            val_features_mean: {features[idx_val].mean().item()}")
    n_channels = features.shape[1]
    # initialize model input
    model = DeepSAD_GVAE(n_channels, hidden_features, n_layers, n_aggr, dropout).to(device)
    
    print(f'Hyperparameters: {hp}.')
    print(f'Hyperparameters: {hp}.', file=out_f)

    optimizer = Adam(model.parameters(), lr=learn_r, weight_decay=alpha)
    lr_decay_step, lr_decay_rate = 500, 0.96
    opt_scheduler = lr_scheduler.MultiStepLR(optimizer, range(lr_decay_step, lr_decay_step*1000, lr_decay_step), gamma=lr_decay_rate)
    
    model.train()
    # train the model
    for epoch in tqdm(range(n_epoch)):          
        num_batches = int(len(idx_train) / batch_size) + 1     

        losses, reconst, latent = torch.zeros(num_batches).to(device), \
                                    torch.zeros(features.shape).to(device), \
                                        torch.zeros(features.shape[0], hidden_features).to(device)
        epoch_time = 0
        start_time = time.time()
        # mini-batch training
        for b, batch in enumerate(range(num_batches)):
            start_time = time.time()
            i_start = batch * batch_size
            i_end = min((batch + 1) * batch_size, len(idx_train))
            idx_batch = idx_train[i_start:i_end]            
            batch_features = features[idx_batch]
            batch_adj_lists, _ = subgraph(torch.tensor(idx_batch), adj_lists, relabel_nodes=True)       
            # print(len(idx_train), batch_features.shape, adj_lists.shape, batch_adj_lists.shape)
            
            optimizer.zero_grad()
            reconst[idx_batch], latent[idx_batch] = model(x = batch_features.float().to(device), \
                                                edge_index = batch_adj_lists.long().to(device))

            train_loss = model.loss()            
            train_loss.backward()
            losses[b] = train_loss
            max_grad_norm = 1.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, norm_type=2)
            optimizer.step()
            opt_scheduler.step()
            
        end_time = time.time()
        epoch_time += end_time - start_time
        
        # validating the model every 50 epochs
        if epoch % 20 == 0:
            
            idx_batch = idx_val
            batch_features = features[idx_batch]
            batch_adj_lists, _ = subgraph(torch.tensor(idx_batch), adj_lists, relabel_nodes=True)       

            val_reconst, val_latent = model(x = batch_features.float().to(device), \
                                            edge_index = batch_adj_lists.long().to(device))

            val_loss = model.loss()            
            print(f'Epoch: {epoch}, train_loss: {losses.mean().item()}, val_loss: {val_loss.item()}')
            print(f'Epoch: {epoch}, train_loss: {losses.mean().item()}, val_loss: {val_loss.item()}', file=out_f)
    
    center = latent[idx_train].mean(dim=0)
    stdev = latent[idx_train].std(dim=0)
    state = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
            } 
    if b_min:
        save_root_center = save_dir+"/center_"+name+"_"+str(hp)+"_"+str(datetime.date.today())+".pt"
        torch.save(center, save_root_center)

        save_root_stdev = save_dir+"/stdev_"+name+"_"+str(hp)+"_"+str(datetime.date.today())+".pt"
        torch.save(stdev, save_root_stdev)

        save_root = save_dir+"/model_"+name+"_"+str(hp)+"_"+str(datetime.date.today())+".pt"
        torch.save(state, save_root)
        
        return val_loss.item() 

    else:             
        save_root_center = save_dir+"/center_"+str(n_epoch)+"_"+str(hp)+"_"+name+"_"+str(datetime.date.today())+".pt"
        torch.save(center, save_root_center)
    
        save_root_stdev = save_dir+"/stdev_"+str(n_epoch)+"_"+str(hp)+"_"+name+"_"+str(datetime.date.today())+".pt"
        torch.save(stdev, save_root_stdev)

        save_root = save_dir+"/model_"+str(n_epoch)+"_"+str(hp)+"_"+name+"_"+str(datetime.date.today())+".pt"
        print('save root:', save_root)
        print('save root:', save_root, file=out_f)
        torch.save(state, save_root)
        
        return center, stdev, state


def train_deepSAD(adj_lists:np.array, features:np.array, labels:np.array, n_epoch:int, b_min:bool, out_f:str, \
                    name:str, center:torch.tensor=None, stdev:torch.tensor=None, \
                    model_path:str=None, state:dict=None, save_dir=None, hp:list=None, tv_state:int=2):

    batch_size, learn_r, hidden_features, n_layers, n_aggr, alpha, dropout, eta, nrml_th, anm_th = hp

    index = list(range(len(labels)))
    idx_train, idx_val, y_train, y_val = train_test_split(index, labels, stratify=labels, test_size=0.20,
                                                            random_state=tv_state, shuffle=True)
    train_features = features[idx_train]
    train_adj_lists, _ = subgraph(torch.tensor(idx_train), adj_lists, relabel_nodes=True)
    val_features = features[idx_val]
    val_adj_lists, _ = subgraph(torch.tensor(idx_val), adj_lists, relabel_nodes=True)       


    n_channels = features.shape[1]
    # initialize model input
    model = DeepSAD_GVAE(n_channels, hidden_features, n_layers, n_aggr, dropout).to(device)
    if model_path:
        model_state = torch.load(model_path)
        model.load_state_dict(model_state["model"])
    elif state:
        model.load_state_dict(state["model"])

    print(f'Hyperparameters: {hp}.')
    print(f'Hyperparameters: {hp}.', file=out_f)

    optimizer = Adam(model.parameters(), lr=learn_r, weight_decay=alpha)
    lr_decay_step, lr_decay_rate = 500, 0.96
    opt_scheduler = lr_scheduler.MultiStepLR(optimizer, range(lr_decay_step, lr_decay_step*1000, lr_decay_step), gamma=lr_decay_rate)
    
    model.train()
    # train the model
    for epoch in tqdm(range(n_epoch)):    
        num_batches = int(len(idx_train) / batch_size) + 1     

        losses, dists = torch.zeros(num_batches).to(device), torch.zeros(len(labels)).to(device)         
        epoch_time = 0

        # mini-batch training
        for b, batch in enumerate(range(num_batches)):
            start_time = time.time()
            i_start = batch * batch_size
            i_end = min((batch + 1) * batch_size, len(idx_train))
            idx_batch = idx_train[i_start:i_end]
            batch_features = features[idx_batch]
            batch_adj_lists, _ = subgraph(torch.tensor(idx_batch), adj_lists, relabel_nodes=True)       
            batch_label = labels[np.array(idx_batch)]
        
            train_semi_labels = gen_semi_labels(batch_label, nrml_th, anm_th)

            optimizer.zero_grad()
            train_latent = model(x = batch_features.float().to(device), \
                                    edge_index = batch_adj_lists.long().to(device), \
                                        centers = center,
                                            stdev = stdev)

            b_loss, dists[idx_batch] = model.HSClassifierLoss(train_semi_labels.to(device), eta)            
            b_loss.backward()
            losses[b] = b_loss
            max_grad_norm = 1.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, norm_type=2)
            optimizer.step()
            opt_scheduler.step()
            
            end_time = time.time()
            epoch_time += end_time - start_time

        # validating the model every 50 epochs
        if epoch % 20 == 0:
            # train_dist_tanh = torch.tanh(dists[idx_train]) 
            train_dist_norm = dists[idx_train] 
            train_recall, train_precision, train_f_score = \
                f_score_(train_dist_norm.detach().cpu().numpy().flatten(), \
                            y_train.detach().cpu().numpy().flatten(), th=1.)                                
            train_roc_auc_score \
                = roc_auc_score(y_train.detach().cpu().numpy().flatten(), \
                                train_dist_norm.detach().cpu().numpy().flatten())

            
            val_semi_labels = gen_semi_labels(y_val, nrml_th, anm_th)

            val_latent = model(x = val_features.float().to(device), \
                                edge_index = val_adj_lists.long().to(device), \
                                    centers = center,
                                        stdev = stdev)

            val_loss, dists[idx_val] = model.HSClassifierLoss(val_semi_labels.to(device), eta)  

            # val_dist_tanh = torch.tanh(dists[idx_val]) 
            val_dist_norm = dists[idx_val]  
            val_recall, val_precision, val_f_score = \
                f_score_(val_dist_norm.detach().cpu().numpy().flatten(), \
                            y_val.detach().cpu().numpy().flatten(), th=1.)                                
            val_roc_auc_score \
                = roc_auc_score(y_val.detach().cpu().numpy().flatten(), \
                                val_dist_norm.detach().cpu().numpy().flatten())
       
            print("Epoch {} - Train_Loss: {} - Train_Recall: {} - Train_Precision: {} - Train_F_score: {} - Train_ROC_AUC_score: {} \
                            \n\t- Val_Loss: {} - Val_Recall: {} - Val_Precision: {} - Val_F_score: {} - Val_ROC_AUC_score: {}"\
                    .format(epoch, np.round(losses.mean().item(), 6), np.round(train_recall, 4), \
                            np.round(train_precision, 4), np.round(train_f_score, 4), \
                                np.round(train_roc_auc_score, 4), \
                            np.round(val_loss.mean().item(), 6), np.round(val_recall, 4), \
                            np.round(val_precision, 4), np.round(val_f_score, 4), \
                                np.round(val_roc_auc_score, 4)))
            print("Epoch {} - Train_Loss: {} - Train_Recall: {} - Train_Precision: {} - Train_F_score: {} - Train_ROC_AUC_score: {} \
                            \n\t- Val_Loss: {} - Val_Recall: {} - Val_Precision: {} - Val_F_score: {} - Val_ROC_AUC_score: {}"\
                    .format(epoch, np.round(losses.mean().item(), 6), np.round(train_recall, 4), \
                            np.round(train_precision, 4), np.round(train_f_score, 4), \
                                np.round(train_roc_auc_score, 4), \
                            np.round(val_loss.mean().item(), 6), np.round(val_recall, 4), \
                            np.round(val_precision, 4), np.round(val_f_score, 4), \
                                np.round(val_roc_auc_score, 4)), file=out_f)        
        state = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }        
        
    if b_min:
        return 1 - val_roc_auc_score 
    else:
        save_root_dist = save_dir+"/dists_"+str(n_epoch)+"_"+str(hp)+"_"+name+"_"+str(datetime.date.today())+".pt"
        torch.save(dists, save_root_dist)
        save_root_labels = save_dir+"/labels_"+str(n_epoch)+"_"+str(hp)+"_"+name+"_"+str(datetime.date.today())+".pt"
        torch.save(labels, save_root_labels)
        save_root_center = save_dir+"/center_"+str(n_epoch)+"_"+str(hp)+"_"+name+"_"+str(datetime.date.today())+".pt"
        torch.save(center, save_root_center)
        
        save_root = save_dir+"/model_"+str(n_epoch)+"_"+str(hp)+"_"+name+"_"+str(datetime.date.today())+".pt"
        print('save root:', save_root)
        print('save root:', save_root, file=out_f)
        torch.save(state, save_root)
        
        return dists

def bayes_opt(train_GVAE, train_deepSAD, train_adj_lists, train_features, y_train, b_epoch_GVAE, b_epoch_SAD, \
                b_min:bool, out_f:str, hp_space_GVAE:list, hp_space_SAD:list, name:str, n_calls_GVAE:int, n_calls_SAD:int, bm_dir:str):    
    print("Performing Bayesian Minimization")
    res_gp_GVAE = gp_minimize(partial(train_GVAE, train_adj_lists, train_features, y_train, \
                                        b_epoch_GVAE, b_min, out_f, name, bm_dir), hp_space_GVAE, n_calls=n_calls_GVAE)
    with open(bm_dir + '/best_hp_'+name+'_GVAE.txt', 'wb') as F:
        pickle.dump(res_gp_GVAE.x, F)
    center = torch.load(bm_dir+"/center_"+name+"_"+str(res_gp_GVAE.x)+"_"+str(datetime.date.today())+".pt")
    stdev = torch.load(bm_dir+"/stdev_"+name+"_"+str(res_gp_GVAE.x)+"_"+str(datetime.date.today())+".pt")
    state = torch.load(bm_dir+"/model_"+name+"_"+str(res_gp_GVAE.x)+"_"+str(datetime.date.today())+".pt")
    hp_space_SAD[2] = [res_gp_GVAE.x[2]]
    hp_space_SAD[3] = [res_gp_GVAE.x[3]]
    hp_space_SAD[4] = [res_gp_GVAE.x[4]]
    res_gp_SAD  = gp_minimize(partial(train_deepSAD, train_adj_lists, train_features, y_train, \
                                        b_epoch_SAD, b_min, out_f, name, center, stdev, None, state, bm_dir), hp_space_SAD, n_calls=n_calls_SAD)
    
    print("GVAE Results: \n", res_gp_GVAE.x_iters, res_gp_GVAE.func_vals)
    print("GVAE Results: \n", res_gp_GVAE.x_iters, res_gp_GVAE.func_vals, file=out_f)
    print("Best Hyperparameters GVAE: ", res_gp_GVAE.x, res_gp_GVAE.fun)
    print("Best Hyperparameters GVAE: ", res_gp_GVAE.x, res_gp_GVAE.fun, file=out_f)
    print("DeepSAD Results: \n", res_gp_SAD.x_iters, res_gp_SAD.func_vals)
    print("DeepSAD Results: \n", res_gp_SAD.x_iters, res_gp_SAD.func_vals, file=out_f)
    print("Best Hyperparameters DeepSAD: ", res_gp_SAD.x, res_gp_SAD.fun)
    print("Best Hyperparameters DeepSAD: ", res_gp_SAD.x, res_gp_SAD.fun, file=out_f)

    with open(bm_dir + '/best_hp_'+name+'_SAD.txt', 'wb') as F:
        pickle.dump(res_gp_SAD.x, F)
    return res_gp_GVAE.x, res_gp_SAD.x


def test_GVAESAD(adj_lists, features, labels, idx_test, y_test, center, stdev, name, \
                model_path_SAD:str=None, state:str=None, hp_SAD:list=None, plots:bool=False, out_f:str=None):
    test_features = features[idx_test]

    batch_size, learn_r, hidden_features, n_layers, n_aggr, alpha, dropout, eta, nrml_th, anm_th = hp_SAD
    n_channels = test_features.shape[1]
    model = DeepSAD_GVAE(n_channels, hidden_features, n_layers, n_aggr, dropout).to(device)
    if model_path_SAD:
        model_state = torch.load(model_path_SAD)
        model.load_state_dict(model_state["model"])
    elif state:
        model.load_state_dict(state["model"])
    model.eval()

    num_batches = int(len(idx_test) / batch_size) + 1     
    test_losses, test_dists = torch.zeros(num_batches).to(device), torch.zeros(len(labels)).to(device)         

    # mini-batch testing
    for b, batch in enumerate(range(num_batches)):
        start_time = time.time()
        i_start = batch * batch_size
        i_end = min((batch + 1) * batch_size, len(idx_test))
        idx_batch = idx_test[i_start:i_end]
        batch_features = features[idx_batch]
        batch_adj_lists, _ = subgraph(torch.tensor(idx_batch), adj_lists, relabel_nodes=True)       
        batch_label = labels[np.array(idx_batch)]
    
        test_semi_labels = gen_semi_labels(batch_label, nrml_th, anm_th)

        train_latent = model(x = batch_features.float().to(device), \
                                edge_index = batch_adj_lists.long().to(device), \
                                    centers = center,
                                        stdev = stdev)

        b_loss, test_dists[idx_batch] = model.HSClassifierLoss(test_semi_labels.to(device), eta)                
        test_losses[b] = b_loss

    # test_dist_tanh = torch.tanh(test_dists[idx_test]) 
    test_dist_norm = test_dists[idx_test]
    test_recall, test_precision, test_f_score = \
        f_score_(test_dist_norm.detach().cpu().numpy().flatten(), \
                    y_test.detach().cpu().numpy().flatten(), th=1.)                                
    test_roc_auc_score \
        = roc_auc_score(y_test.detach().cpu().numpy().flatten(), \
                        test_dist_norm.detach().cpu().numpy().flatten())

    print("Test_Loss: {} - Test_Recall: {} - Test_Precision: {} - Test_F_score: {} - Test_ROC_AUC_score: {}"\
            .format(np.round(test_losses.mean().item(), 6), np.round(test_recall, 4), \
                    np.round(test_precision, 4), np.round(test_f_score, 4), \
                        np.round(test_roc_auc_score, 4)))
    print("Test_Loss: {} - Test_Recall: {} - Test_Precision: {} - Test_F_score: {} - Test_ROC_AUC_score: {}"\
            .format(np.round(test_losses.mean().item(), 6), np.round(test_recall, 4), \
                    np.round(test_precision, 4), np.round(test_f_score, 4), \
                        np.round(test_roc_auc_score, 4)), file=out_f)
                
    if plots:
        ''' Simple Scatter Plot '''
        plt.figure(figsize=(10,10))
        dists = test_dists[idx_test].detach().cpu().numpy()
        # dists = torch.tanh(test_dist).detach().cpu().numpy()
        targets = y_test.cpu().numpy()
        
        x_dists_pos = np.where(targets == 0)[0]
        dists_pos = dists[targets == 0]
        # dists_pos = np.log(dists[targets == 0])
        plt.scatter(x_dists_pos, dists_pos, label=('Normal'))  
        x_dists_neg = np.where(targets == 1)[0]
        dists_neg = dists[targets == 1]
        # dists_neg = np.log(dists[targets == 1])
        plt.scatter(x_dists_neg, dists_neg, label=('Anomalous'))  

        plt.legend()
        plt.tight_layout()
        plt.savefig("test_distances_"+name)
        plt.show()
        ''' --------------------------------------- '''
    return test_recall, test_precision, test_f_score, test_roc_auc_score


def train_test_multiple(adj, features, labels, n_epoch_GVAE, n_epoch_SAD, out_f, name, hp_GVAE, hp_SAD, random_states, t_size, save_dir):
    
    N = len(random_states)
    test_recall, test_precision, test_f_score, test_roc_auc_score\
        = torch.zeros(N,N), torch.zeros(N,N), torch.zeros(N,N), torch.zeros(N,N)
    for tt, tt_state in enumerate(random_states):
        if name == 'yelp':
            index = list(range(len(labels)))
            idx_train, idx_test, y_train, y_test \
                = train_test_split(index, labels, stratify=labels, test_size=t_size,
                                        random_state=tt_state, shuffle=True)
        elif name == 'amazon':  # amazon
            # 0-3304 are unlabeled nodes
            index = list(range(3305, len(labels)))
            idx_train, idx_test, y_train, y_test \
                = train_test_split(index, labels[3305:], stratify=labels[3305:],
                                    test_size=t_size, random_state=tt_state, shuffle=True)    
        features = F.normalize(features)
        train_features = features[idx_train]
        adj_lists = convert_to_edgeindices(adj)
        train_adj_lists, _ = subgraph(torch.tensor(idx_train), adj_lists, relabel_nodes=True)
        test_adj_lists, _ = subgraph(torch.tensor(idx_test), adj_lists, relabel_nodes=True)
        print(f"test_random_state: {tt_state}, test_features_mean: {features[idx_test].mean().item()}")

        for tv, tv_state in enumerate(random_states):
            center, stdev, state = train_GVAE(train_adj_lists, train_features, y_train, n_epoch_GVAE, b_min=False, out_f=out_f, name=name, hp=hp_GVAE, tv_state=tv_state, save_dir=save_dir)

            center = torch.load(save_dir+"/center_"+str(n_epoch_GVAE)+"_"+str(hp_GVAE)+"_"+name+"_"+str(datetime.date.today())+".pt")

            stdev = torch.load(save_dir+"/stdev_"+str(n_epoch_GVAE)+"_"+str(hp_GVAE)+"_"+name+"_"+str(datetime.date.today())+".pt")

            model_path = save_dir+"/model_"+str(n_epoch_GVAE)+"_"+str(hp_GVAE)+"_"+name+"_"+str(datetime.date.today())+".pt"
            dists = train_deepSAD(train_adj_lists, train_features, y_train, n_epoch_SAD, b_min=False, out_f=out_f, name=name, center=center, stdev=stdev, model_path=model_path, hp=hp_SAD, tv_state=tv_state, save_dir=save_dir)

            model_path_SAD = save_dir+"/model_"+str(n_epoch_SAD)+"_"+str(hp_SAD)+"_"+name+"_"+str(datetime.date.today())+".pt"
            test_recall[tt, tv], test_precision[tt, tv], test_f_score[tt, tv], test_roc_auc_score[tt, tv] \
                = test_GVAESAD(adj_lists, features, labels, idx_test, y_test, center, stdev, name, model_path_SAD=model_path_SAD, hp_SAD=hp_SAD, out_f=out_f)    

    return test_recall, test_precision, test_f_score, test_roc_auc_score


