import numpy as np
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Dataset, InMemoryDataset
from src.models.GEN_Classifier import GEN_Classifier
from torch.utils.data.dataset import random_split
from sklearn.metrics import roc_auc_score
import time, random, datetime, copy, pickle
from skopt import gp_minimize
from functools import partial
from tqdm import tqdm
from torch.optim import Adam, lr_scheduler
from src.utils.utils import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_GEN_Classifier(tv_data:list, n_epoch:int, b_min:bool, out_f:str, \
                    name:str, model_path:str=None, state:dict=None, save_dir=None, hp:list=None, tv_state:int=2):

    learn_r, hidden_features, n_layers, n_aggr, alpha, dropout = hp

    tv_data_c = copy.deepcopy(tv_data)    

    N = len(tv_data_c.indices)
    train_len = int(0.8 * N)
    train_data, val_data = random_split(tv_data_c, [train_len, N-train_len], generator=torch.Generator().manual_seed(tv_state))

    print(f"tv_random_state: {tv_state}")
    n_channels = train_data[0].node_features.shape[1]
    e_channels = train_data[0].edge_features.shape[1]
    # initialize model input
    model = GEN_Classifier(n_channels, hidden_features, n_layers, n_aggr, dropout, e_channels=e_channels).to(device)
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

        losses = torch.zeros(N).to(device)
        tv_probs = []        
        epoch_time = 0

        for r, repo in enumerate(train_data):
            start_time = time.time()            
        
            optimizer.zero_grad()
            train_probs = model(x = repo.node_features.float().to(device), \
                                    edge_index = repo.edge_indices.long().to(device), \
                                        edge_attr = repo.edge_features.float().to(device))

            loss = model.CE_loss(repo.targets.to(device))            
            loss.backward()
            losses[r] = loss
            tv_probs.append(train_probs)
            max_grad_norm = 1.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, norm_type=2)
            optimizer.step()
            opt_scheduler.step()
            
            end_time = time.time()
            epoch_time += end_time - start_time

        # validating the model every 50 epochs
        if epoch % 20 == 0:
            # train_dist_tanh = torch.tanh(probs[idx_train]) 
            train_recall, train_precision, train_f_score, train_roc_auc_score \
                = np.zeros(train_len), np.zeros(train_len), np.zeros(train_len), np.zeros(train_len)
            for r in range(train_len):
                train_probs = tv_probs[r]
                y_train = train_data[r].targets 
                train_recall[r], train_precision[r], train_f_score[r] = \
                    f_score_(train_probs.detach().cpu().numpy().flatten(), \
                                y_train.detach().cpu().numpy().flatten(), th=0.5)                                
                train_roc_auc_score[r] \
                    = roc_auc_score(y_train.detach().cpu().numpy().flatten(), \
                                    train_probs.detach().cpu().numpy().flatten())
            val_len = N - train_len
            val_recall, val_precision, val_f_score, val_roc_auc_score \
                = np.zeros(val_len), np.zeros(val_len), np.zeros(val_len), np.zeros(val_len)
            for r, repo in enumerate(val_data):
                start_time = time.time()            
            
                val_probs = model(x = repo.node_features.float().to(device), \
                                        edge_index = repo.edge_indices.long().to(device), \
                                            edge_attr = repo.edge_features.float().to(device),)

                loss = model.CE_loss(repo.targets.to(device))            
                losses[train_len + r] = loss            
                tv_probs.append(val_probs)

                y_val = repo.targets
                val_recall[r], val_precision[r], val_f_score[r] = \
                    f_score_(val_probs.detach().cpu().numpy().flatten(), \
                                y_val.detach().cpu().numpy().flatten(), th=0.5)                                
                val_roc_auc_score[r] \
                    = roc_auc_score(y_val.detach().cpu().numpy().flatten(), \
                                    val_probs.detach().cpu().numpy().flatten())
       
            print("Epoch {} - Train_Loss: {} - Train_Recall: {} - Train_Precision: {} - Train_F_score: {} - Train_ROC_AUC_score: {} \
                            \n\t- Val_Loss: {} - Val_Recall: {} - Val_Precision: {} - Val_F_score: {} - Val_ROC_AUC_score: {}"\
                    .format(epoch, np.round(losses[:train_len].mean().item(), 6), np.round(np.mean(train_recall), 4), \
                            np.round(np.mean(train_precision), 4), np.round(np.mean(train_f_score), 4), \
                                np.round(np.mean(train_roc_auc_score), 4), \
                            np.round(losses[train_len:].mean().item(), 6), np.round(np.mean(val_recall), 4), \
                            np.round(np.mean(val_precision), 4), np.round(np.mean(val_f_score), 4), \
                                np.round(np.mean(val_roc_auc_score), 4)))
            print("Epoch {} - Train_Loss: {} - Train_Recall: {} - Train_Precision: {} - Train_F_score: {} - Train_ROC_AUC_score: {} \
                            \n\t- Val_Loss: {} - Val_Recall: {} - Val_Precision: {} - Val_F_score: {} - Val_ROC_AUC_score: {}"\
                    .format(epoch, np.round(losses[:train_len].mean().item(), 6), np.round(np.mean(train_recall), 4), \
                            np.round(np.mean(train_precision), 4), np.round(np.mean(train_f_score), 4), \
                                np.round(np.mean(train_roc_auc_score), 4), \
                            np.round(losses[train_len:].mean().item(), 6), np.round(np.mean(val_recall), 4), \
                            np.round(np.mean(val_precision), 4), np.round(np.mean(val_f_score), 4), \
                                np.round(np.mean(val_roc_auc_score), 4)), file=out_f)        
        state = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }    
   
        
    if b_min:
        return 1 - np.mean(val_f_score) 
    else:
        save_root_probs = save_dir+"/tv_probs_"+str(n_epoch)+"_"+str(hp)+"_"+name+"_"+str(datetime.date.today())+".pt"
        torch.save(tv_probs, save_root_probs)

        save_root_targets = save_dir+"/targets_"+str(n_epoch)+"_"+str(hp)+"_"+name+"_"+str(datetime.date.today())+".pt"
        torch.save([repo.targets for repo in train_data] + [repo.targets for repo in val_data], save_root_targets)
      
        save_root = save_dir+"/model_"+str(n_epoch)+"_"+str(hp)+"_"+name+"_"+str(datetime.date.today())+".pt"
        print('save root:', save_root)
        print('save root:', save_root, file=out_f)
        torch.save(state, save_root)

        del model, optimizer, opt_scheduler, losses, tv_data, train_data, val_data, tv_data_c, train_probs, val_probs
        
        return tv_probs

def bayes_opt(train_GEN_Classifier, tv_data, b_epoch_GEN, \
                b_min:bool, out_f:str, hp_space_GEN:list, name:str, n_calls_GEN:int, bm_dir:str):    
    print("Performing Bayesian Minimization")
    
    res_gp_GEN  = \
        gp_minimize(partial(train_GEN_Classifier, tv_data, b_epoch_GEN, \
            b_min, out_f, name, None, None, bm_dir), hp_space_GEN, n_calls=n_calls_GEN)
    
    print("GEN_Classifier Results: \n", res_gp_GEN.x_iters, res_gp_GEN.func_vals)
    print("GEN_Classifier Results: \n", res_gp_GEN.x_iters, res_gp_GEN.func_vals, file=out_f)
    print("Best Hyperparameters GEN_Classifier: ", res_gp_GEN.x, res_gp_GEN.fun)
    print("Best Hyperparameters GEN_Classifier: ", res_gp_GEN.x, res_gp_GEN.fun, file=out_f)
    
    with open(bm_dir + 'best_hp_'+name+'_GEN.txt', 'wb') as F:
        pickle.dump(res_gp_GEN.x, F)
    return res_gp_GEN.x


def test_GEN_Classifier(test_data, name, model_path_GEN:str=None, state:str=None, hp_GEN:list=None, plots:bool=False, out_f:str=None, malicious:str=''):

    learn_r, hidden_features, n_layers, n_aggr, alpha, dropout = hp_GEN

    try:
        N = len(test_data.indices)
    except:
        N = test_data.len()
    n_channels = test_data[0].node_features.shape[1]
    e_channels = test_data[0].edge_features.shape[1]
    model = GEN_Classifier(n_channels, hidden_features, n_layers, n_aggr, dropout, e_channels=e_channels).to(device)
    if model_path_GEN:
        model_state = torch.load(model_path_GEN)
        model.load_state_dict(model_state["model"])
    elif state:
        model.load_state_dict(state["model"])
    model.eval()

    test_losses = torch.zeros(N).to(device)
    test_probs = []         
    test_recall, test_precision, test_f_score, test_roc_auc_score \
                = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
    # testing
    for r, repo in enumerate(test_data):
        start_time = time.time()

        test_prob = model(x = repo.node_features.float().to(device), \
                                edge_index = repo.edge_indices.long().to(device), \
                                    edge_attr = repo.edge_features.float().to(device))

        loss = model.CE_loss(repo.targets.to(device))            
        test_losses[r] = loss            
        test_probs.append(test_prob)

        y_val = repo.targets
        test_recall[r], test_precision[r], test_f_score[r] = \
            f_score_(test_prob.detach().cpu().numpy().flatten(), \
                        y_val.detach().cpu().numpy().flatten(), th=0.5)                                
        test_roc_auc_score[r] \
            = roc_auc_score(y_val.detach().cpu().numpy().flatten(), \
                            test_prob.detach().cpu().numpy().flatten())

    print("Test_Loss: {} - Test_Recall: {} - Test_Precision: {} - Test_F_score: {} - Test_ROC_AUC_score: {}"\
            .format(np.round(test_losses.mean().item(), 6), np.round(np.mean(test_recall), 4), \
                    np.round(np.mean(test_precision), 4), np.round(np.mean(test_f_score), 4), \
                        np.round(np.mean(test_roc_auc_score), 4)))
    print("Test_Loss: {} - Test_Recall: {} - Test_Precision: {} - Test_F_score: {} - Test_ROC_AUC_score: {}"\
            .format(np.round(test_losses.mean().item(), 6), np.round(np.mean(test_recall), 4), \
                    np.round(np.mean(test_precision), 4), np.round(np.mean(test_f_score), 4), \
                        np.round(np.mean(test_roc_auc_score), 4)), file=out_f)
                
    if plots:
        ''' Simple Scatter Plot '''
        plt.figure(figsize=(20,20))
        for r, repo in enumerate(test_data):
            plt.subplot(5, 4, r+1)

            probs = test_probs[r].detach().cpu().numpy()
            targets = repo.targets.detach().cpu().numpy() 
            labels = repo.node_labels.detach().cpu().numpy() 

            x_probs_pos = np.where(targets == 0)[0]
            probs_pos = probs[targets == 0]
            # probs_pos = np.log(probs[targets == 0])
            plt.scatter(x_probs_pos, probs_pos, label=('Normal'))  
            
            x_probs_neg = np.where(labels == 1)[0]
            probs_neg = probs[labels == 1]
            # probs_neg = np.log(probs[targets == 1])
            plt.scatter(x_probs_neg, probs_neg, label=('Anomalous 1'))  

            x_probs_neg = np.where(labels == 2)[0]
            probs_neg = probs[labels == 2]
            # probs_neg = np.log(probs[targets == 1])
            plt.scatter(x_probs_neg, probs_neg, label=('Anomalous 2'))  

            x_probs_neg = np.where(labels == 3)[0]
            probs_neg = probs[labels == 3]
            # probs_neg = np.log(probs[targets == 1])
            plt.scatter(x_probs_neg, probs_neg, label=('Anomalous 3'))  

            x_probs_neg = np.where(labels == 4)[0]
            probs_neg = probs[labels == 4]
            # probs_neg = np.log(probs[targets == 1])
            plt.scatter(x_probs_neg, probs_neg, label=('Anomalous 4'))  

            x_probs_neg = np.where(labels == 5)[0]
            probs_neg = probs[labels == 5]
            # probs_neg = np.log(probs[targets == 1])
            plt.scatter(x_probs_neg, probs_neg, label=('Anomalous 5'))  

            x_probs_label = np.where(labels == 11)[0]
            probs_label = probs[labels == 11]
            # probs_neg = np.log(probs[targets == 1])
            plt.scatter(x_probs_label, probs_label, label=('Malicious'), color='r')  

            x_probs_label = np.where(labels == 12)[0]
            probs_label = probs[labels == 12]
            # probs_neg = np.log(probs[targets == 1])
            plt.scatter(x_probs_label, probs_label, label=('Octopus'), color='r')  

        plt.legend()
        plt.tight_layout()
        plt.savefig("test_distances_"+name+malicious)
        plt.show()
        ''' --------------------------------------- '''
    del model, test_losses, test_data, test_prob, test_probs 
    
    return np.mean(test_recall), np.mean(test_precision), np.mean(test_f_score), np.mean(test_roc_auc_score) 


def train_test_multiple(data, mal_data, n_epoch_GEN, out_f, name, hp_GEN, random_states, t_size, save_dir):
    
    M = len(random_states)
    test_recall, test_precision, test_f_score, test_roc_auc_score\
        = np.zeros((M,M)), np.zeros((M,M)), np.zeros((M,M)), np.zeros((M,M))
    mal_test_recall, mal_test_precision, mal_test_f_score, mal_test_roc_auc_score\
        = np.zeros((M,M)), np.zeros((M,M)), np.zeros((M,M)), np.zeros((M,M))
    for tt, tt_state in enumerate(random_states):
        data_c = copy.deepcopy(data)
        N = data_c.len()
        tv_len, test_len = int(t_size * N), N
        tv_data, test_data = random_split(data_c, [tv_len, N-tv_len], generator=torch.Generator().manual_seed(tt_state))        

        print(f"test_random_state: {tt_state}")

        for tv, tv_state in enumerate(random_states):
            
            probs = train_GEN_Classifier(tv_data, n_epoch_GEN, b_min=False, out_f=out_f, name=name, model_path=None, hp=hp_GEN, tv_state=tv_state, save_dir=save_dir)

            model_path_GEN = save_dir+"/model_"+str(n_epoch_GEN)+"_"+str(hp_GEN)+"_"+name+"_"+str(datetime.date.today())+".pt"
            test_recall[tt, tv], test_precision[tt, tv], test_f_score[tt, tv], test_roc_auc_score[tt, tv] \
                = test_GEN_Classifier(test_data, name, model_path_GEN=model_path_GEN, hp_GEN=hp_GEN, out_f=out_f)    
            
            mal_test_recall[tt, tv], mal_test_precision[tt, tv], mal_test_f_score[tt, tv], mal_test_roc_auc_score[tt, tv] \
                = test_GEN_Classifier(mal_data, name, model_path_GEN=model_path_GEN, hp_GEN=hp_GEN, out_f=out_f, malicious='malicious')   

    return test_recall, test_precision, test_f_score, test_roc_auc_score, \
            mal_test_recall, mal_test_precision, mal_test_f_score, mal_test_roc_auc_score


class Repo_Dataset(Dataset):
    def __init__(self, ):
        super().__init__()        

    def len(self):
        return len(self._data_list)

    def get(self, idx):
        repo_graph = pickle.load(open(self._data_list[idx], "rb"))
        return repo_graph

    
class Repo_Dataset_IM(InMemoryDataset):
    def __init__(self, ):
        super().__init__()        

    def len(self):
        return len(self._data_list)
    
    def get(self, idx):
        repo_graph = self.data[idx]
        return repo_graph

    def load(self, ):
        self.data = []
        for url in self._data_list:
            repo_graph = pickle.load(open(url, "rb"))
            self.data.append(repo_graph)
        return self.data
        

