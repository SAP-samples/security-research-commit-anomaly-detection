import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch_geometric.utils import subgraph
from src.models.GEN_Classifier import GEN_Classifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import time, datetime, pickle
from skopt import gp_minimize
from functools import partial
from tqdm import tqdm
from torch.optim import Adam, lr_scheduler
from src.utils.utils import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_GEN_Classifier(adj_lists:np.array, features:np.array, labels:np.array, n_epoch:int, b_min:bool, out_f:str, \
                    name:str, model_path:str=None, state:dict=None, save_dir:str=None, hp:list=None, tv_state:int=2):

    batch_size, learn_r, hidden_features, n_layers, n_aggr, alpha, dropout = hp

    index = list(range(len(labels)))
    idx_train, idx_val, y_train, y_val = train_test_split(index, labels, stratify=labels, test_size=0.20,
                                                            random_state=tv_state, shuffle=True)
    train_features = features[idx_train]
    train_adj_lists, _ = subgraph(torch.tensor(idx_train), adj_lists, relabel_nodes=True)
    val_features = features[idx_val]
    val_adj_lists, _ = subgraph(torch.tensor(idx_val), adj_lists, relabel_nodes=True)       


    n_channels = features.shape[1]
    # initialize model input
    model = GEN_Classifier(n_channels, hidden_features, n_layers, n_aggr, dropout).to(device)
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

        losses, tv_probs = torch.zeros(num_batches).to(device), torch.zeros(len(labels)).to(device)             
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
        
            optimizer.zero_grad()
            tv_probs[idx_batch] = model(x = batch_features.float().to(device), \
                                    edge_index = batch_adj_lists.long().to(device))

            b_loss = model.CE_loss(batch_label.to(device))            
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
            train_probs = tv_probs[idx_train]            
            train_recall, train_precision, train_f_score = \
                f_score_(train_probs.detach().cpu().numpy().flatten(), \
                            y_train.detach().cpu().numpy().flatten(), th=0.5)                                
            train_roc_auc_score \
                = roc_auc_score(y_train.detach().cpu().numpy().flatten(), \
                                train_probs.detach().cpu().numpy().flatten())

            tv_probs[idx_val] = model(x = val_features.float().to(device), \
                                edge_index = val_adj_lists.long().to(device))

            val_loss = model.CE_loss(y_val.to(device))  

            val_probs = tv_probs[idx_val]  
            val_recall, val_precision, val_f_score = \
                f_score_(val_probs.detach().cpu().numpy().flatten(), \
                            y_val.detach().cpu().numpy().flatten(), th=0.5)                                
            val_roc_auc_score \
                = roc_auc_score(y_val.detach().cpu().numpy().flatten(), \
                                val_probs.detach().cpu().numpy().flatten())
       
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
        save_root_probs = save_dir+"/tv_probs_"+str(n_epoch)+"_"+str(hp)+"_"+name+"_"+str(datetime.date.today())+".pt"
        torch.save(tv_probs, save_root_probs)
        save_root_labels = save_dir+"/labels_"+str(n_epoch)+"_"+str(hp)+"_"+name+"_"+str(datetime.date.today())+".pt"
        torch.save(labels, save_root_labels)        
        
        save_root = save_dir+"/model_"+str(n_epoch)+"_"+str(hp)+"_"+name+"_"+str(datetime.date.today())+".pt"
        print('save root:', save_root)
        print('save root:', save_root, file=out_f)
        torch.save(state, save_root)
        
        return tv_probs

def bayes_opt(train_GEN_Classifier, train_adj_lists, train_features, y_train, b_epoch, \
                b_min:bool, out_f:str, hp_space:list, name:str, n_calls:int, bm_dir:str):    
    print("Performing Bayesian Minimization")
    
    res_gp  = gp_minimize(partial(train_GEN_Classifier, train_adj_lists, train_features, y_train, \
                                b_epoch, b_min, out_f, name, None, None, bm_dir), hp_space, n_calls=n_calls)
    
    print("All Results: \n", res_gp.x_iters, res_gp.func_vals)
    print("All Results: \n", res_gp.x_iters, res_gp.func_vals, file=out_f)
    print("Best Hyperparameters: ", res_gp.x, res_gp.fun)
    print("Best Hyperparameters: ", res_gp.x, res_gp.fun, file=out_f)
    
    with open(bm_dir + '/best_hp_'+name+'_GEN.txt', 'wb') as F:
        pickle.dump(res_gp.x, F)
    return res_gp.x


def test_GEN_Classifier(adj_lists, features, labels, idx_test, y_test, name, \
                model_path_GEN:str=None, state:str=None, hp_GEN:list=None, plots:bool=False, out_f:str=None):
    test_features = features[idx_test]

    batch_size, learn_r, hidden_features, n_layers, n_aggr, alpha, dropout= hp_GEN
    n_channels = test_features.shape[1]
    model = GEN_Classifier(n_channels, hidden_features, n_layers, n_aggr, dropout).to(device)
    if model_path_GEN:
        model_state = torch.load(model_path_GEN)
        model.load_state_dict(model_state["model"])
    elif state:
        model.load_state_dict(state["model"])
    model.eval()

    num_batches = int(len(idx_test) / batch_size) + 1     
    test_losses, probs = torch.zeros(num_batches).to(device), torch.zeros(len(labels)).to(device)         

    # mini-batch testing
    for b, batch in enumerate(range(num_batches)):
        start_time = time.time()
        i_start = batch * batch_size
        i_end = min((batch + 1) * batch_size, len(idx_test))
        idx_batch = idx_test[i_start:i_end]
        batch_features = features[idx_batch]
        batch_adj_lists, _ = subgraph(torch.tensor(idx_batch), adj_lists, relabel_nodes=True)       
        batch_label = labels[np.array(idx_batch)]
    
        probs[idx_batch] = model(x = batch_features.float().to(device), \
                                edge_index = batch_adj_lists.long().to(device))

        b_loss = model.CE_loss(batch_label.to(device))                
        test_losses[b] = b_loss

    test_probs = probs[idx_test]
    test_recall, test_precision, test_f_score = \
        f_score_(test_probs.detach().cpu().numpy().flatten(), \
                    y_test.detach().cpu().numpy().flatten(), th=0.5)                                
    test_roc_auc_score \
        = roc_auc_score(y_test.detach().cpu().numpy().flatten(), \
                        test_probs.detach().cpu().numpy().flatten())

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
        probs = test_probs.detach().cpu().numpy()
        targets = y_test.cpu().numpy()
        
        x_probs_pos = np.where(targets == 0)[0]
        probs_pos = probs[targets == 0]
        # probs_pos = np.log(probs[targets == 0])
        plt.scatter(x_probs_pos, probs_pos, label=('Normal'))  
        x_probs_neg = np.where(targets == 1)[0]
        probs_neg = probs[targets == 1]
        # probs_neg = np.log(probs[targets == 1])
        plt.scatter(x_probs_neg, probs_neg, label=('Anomalous'))  

        plt.legend()
        plt.tight_layout()
        plt.savefig("test_distances_"+name)
        plt.show()
        ''' --------------------------------------- '''
    return test_recall, test_precision, test_f_score, test_roc_auc_score


def train_test_multiple(adj, features, labels, n_epoch_GEN, out_f, name, hp_GEN, random_states, t_size, save_dir):
    
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
            
            tv_probs = train_GEN_Classifier(train_adj_lists, train_features, y_train, n_epoch_GEN, b_min=False, out_f=out_f, name=name, model_path=None, hp=hp_GEN, tv_state=tv_state, save_dir=save_dir)

            model_path_GEN = save_dir+"/model_"+str(n_epoch_GEN)+"_"+str(hp_GEN)+"_"+name+"_"+str(datetime.date.today())+".pt"
            test_recall[tt, tv], test_precision[tt, tv], test_f_score[tt, tv], test_roc_auc_score[tt, tv] \
                = test_GEN_Classifier(adj_lists, features, labels, idx_test, y_test, name, model_path_GEN=model_path_GEN, hp_GEN=hp_GEN, out_f=out_f)    

    return test_recall, test_precision, test_f_score, test_roc_auc_score


