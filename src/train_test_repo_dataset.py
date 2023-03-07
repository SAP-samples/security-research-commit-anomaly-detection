import numpy as np
import matplotlib.pyplot as plt
import torch
from src.models.deepSAD_model import DeepSAD_GVAE
from torch.utils.data.dataset import random_split
from sklearn.model_selection import train_test_split
from torch_geometric.utils import subgraph
from sklearn.metrics import roc_auc_score
import time, random, datetime, copy, pickle
from skopt import gp_minimize
from functools import partial
from tqdm import tqdm
from torch.optim import Adam, lr_scheduler
from src.utils.utils import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_GVAE(train_data:list, val_data:list, n_epoch:int, b_min:bool, out_f:str, \
                name:str, save_dir:str, hp:list, tv_state:int=2):

    learn_r, hidden_features, n_layers, n_aggr, alpha, dropout = hp

    N = train_data.len()

    print(f"tv_random_state: {tv_state}")
    n_channels = train_data[0].node_features.shape[1]
    e_channels = train_data[0].edge_features.shape[1]
    # initialize model input
    model = DeepSAD_GVAE(n_channels, hidden_features, n_layers, n_aggr, dropout, e_channels=e_channels).to(device)
    
    print(f'Hyperparameters: {hp}.')
    print(f'Hyperparameters: {hp}.', file=out_f)

    optimizer = Adam(model.parameters(), lr=learn_r, weight_decay=alpha)
    lr_decay_step, lr_decay_rate = 500, 0.96
    opt_scheduler = lr_scheduler.MultiStepLR(optimizer, range(lr_decay_step, lr_decay_step*1000, lr_decay_step), gamma=lr_decay_rate)
    
    model.train()
    # train the model
    for epoch in tqdm(range(n_epoch)):          
        train_losses, train_centers, train_stdevs \
             = torch.zeros(N), \
                    torch.zeros(N, hidden_features).to(device), \
                        torch.zeros(N, hidden_features).to(device)
        epoch_time = 0
        start_time = time.time()

        for r, repo_o in enumerate(train_data):
            repo = copy.deepcopy(repo_o)
            start_time = time.time()
            
            optimizer.zero_grad()
            train_reconst, train_latent = model(x = repo.node_features.float().to(device), \
                                            edge_index = repo.edge_indices.long().to(device), \
                                                edge_attr = repo.edge_features.float().to(device))
            train_loss = model.loss()            
            train_loss.backward()
            train_losses[r] = train_loss.detach().cpu()
            train_centers[r] = train_latent.detach().mean(dim=0)
            train_stdevs[r] = train_latent.detach().std(dim=0)

            max_grad_norm = 1.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, norm_type=2)
            optimizer.step()
            opt_scheduler.step()

            del repo, train_reconst, train_latent            
            
        end_time = time.time()
        epoch_time += end_time - start_time
        
        # validating the model every 50 epochs
        if epoch % 20 == 0:
            val_losses = torch.zeros(N)
            for r, repo_o in enumerate(val_data):
                repo = copy.deepcopy(repo_o)
                start_time = time.time()
                
                model(x = repo.node_features.float().to(device), \
                            edge_index = repo.edge_indices.long().to(device), \
                                edge_attr = repo.edge_features.float().to(device))
                val_loss = model.loss()            
                val_losses[r] = val_loss.detach().cpu()

                del repo, val_loss
        
            print(f'Epoch: {epoch}, train_loss: {train_losses.mean().item()}, val_loss: {val_losses.mean().item()}')
            print(f'Epoch: {epoch}, train_loss: {train_losses.mean().item()}, val_loss: {val_losses.mean().item()}', file=out_f)
            
    center = train_centers.mean(dim=0)
    stdev = train_stdevs.mean(dim=0)
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
        
        return val_losses.mean().item() 

    else:             
        save_root_center = save_dir+"/center_"+str(n_epoch)+"_"+str(hp)+"_"+name+"_"+str(datetime.date.today())+".pt"
        torch.save(center, save_root_center)
    
        save_root_stdev = save_dir+"/stdev_"+str(n_epoch)+"_"+str(hp)+"_"+name+"_"+str(datetime.date.today())+".pt"
        torch.save(stdev, save_root_stdev)

        save_root = save_dir+"/model_"+str(n_epoch)+"_"+str(hp)+"_"+name+"_"+str(datetime.date.today())+".pt"
        print('save root:', save_root)
        print('save root:', save_root, file=out_f)
        torch.save(state, save_root)
        
        del model, optimizer, opt_scheduler, train_losses, val_losses, train_data, val_data, train_centers, train_stdevs
        
        return center, stdev, state


def train_deepSAD(train_data:list, val_data:list, n_epoch:int, b_min:bool, out_f:str, \
                    name:str, center:torch.tensor=None, stdev:torch.tensor=None, \
                    model_path:str=None, state:dict=None, save_dir=None, hp:list=None, tv_state:int=2):

    learn_r, hidden_features, n_layers, n_aggr, alpha, dropout, eta, nrml_th, anm_th = hp

    N = train_data.len()
           
    print(f"tv_random_state: {tv_state}")
    n_channels = train_data[0].node_features.shape[1]
    e_channels = train_data[0].edge_features.shape[1]
    # initialize model input
    model = DeepSAD_GVAE(n_channels, hidden_features, n_layers, n_aggr, dropout, e_channels=e_channels).to(device)
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

        losses = torch.zeros(N)#
        dists = []        
        epoch_time = 0

        for r, repo_o in enumerate(train_data):
            repo = copy.deepcopy(repo_o)
            start_time = time.time()            
        
            repo_semi_labels = gen_semi_labels(repo.targets, nrml_th, anm_th)

            optimizer.zero_grad()
            model(x = repo.node_features.float().to(device), \
                        edge_index = repo.edge_indices.long().to(device), \
                            edge_attr = repo.edge_features.float().to(device), \
                                centers = center, stdev = stdev)

            loss, dist = model.HSClassifierLoss(repo_semi_labels.to(device), eta)            
            loss.backward()
            losses[r] = loss.detach().cpu()
            dists.append(dist.detach().cpu())
            max_grad_norm = 1.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, norm_type=2)
            optimizer.step()
            opt_scheduler.step()
            
            end_time = time.time()
            epoch_time += end_time - start_time            
            
            del loss, dist, repo_semi_labels, repo

        # validating the model every 50 epochs
        if epoch % 20 == 0:
            # train_dist_tanh = torch.tanh(dists[idx_train]) 
            train_recall, train_precision, train_f_score, train_roc_auc_score \
                = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
            for r in range(N):
                train_dist_norm = dists[r]
                y_train = train_data[r].targets 
                train_recall[r], train_precision[r], train_f_score[r] = \
                    f_score_(train_dist_norm.detach().cpu().numpy().flatten(), \
                                y_train.detach().cpu().numpy().flatten(), th=1.)                                
                train_roc_auc_score[r] \
                    = roc_auc_score(y_train.detach().cpu().numpy().flatten(), \
                                    train_dist_norm.detach().cpu().numpy().flatten())
            del dists, train_dist_norm 

            val_recall, val_precision, val_f_score, val_roc_auc_score \
                = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
            val_losses = torch.zeros(N)
            val_dists = []   
            for r, repo_o in enumerate(val_data):
                repo = copy.deepcopy(repo_o)
                start_time = time.time()            
            
                repo_semi_labels = gen_semi_labels(repo.targets, nrml_th, anm_th)

                model(x = repo.node_features.float().to(device), \
                            edge_index = repo.edge_indices.long().to(device), \
                                edge_attr = repo.edge_features.float().to(device), \
                                    centers = center, stdev = stdev)

                loss, dist = model.HSClassifierLoss(repo_semi_labels.to(device), eta)            
                val_losses[r] = loss.detach().cpu()            
                # val_dists.append(dist)
                # val_dist_tanh = torch.tanh(dists[idx_val]) 
                val_dist_norm = dist  
                y_val = repo.targets
                val_recall[r], val_precision[r], val_f_score[r] = \
                    f_score_(val_dist_norm.detach().cpu().numpy().flatten(), \
                                y_val.detach().cpu().numpy().flatten(), th=1.)                                
                val_roc_auc_score[r] \
                    = roc_auc_score(y_val.detach().cpu().numpy().flatten(), \
                                    val_dist_norm.detach().cpu().numpy().flatten())
                
                del val_dist_norm, dist, loss, repo 

       
            print("Epoch {} - Train_Loss: {} - Train_Recall: {} - Train_Precision: {} - Train_F_score: {} - Train_ROC_AUC_score: {} \
                            \n\t- Val_Loss: {} - Val_Recall: {} - Val_Precision: {} - Val_F_score: {} - Val_ROC_AUC_score: {}"\
                    .format(epoch, np.round(losses.mean().item(), 6), np.round(np.mean(train_recall), 4), \
                            np.round(np.mean(train_precision), 4), np.round(np.mean(train_f_score), 4), \
                                np.round(np.mean(train_roc_auc_score), 4), \
                            np.round(val_losses.mean().item(), 6), np.round(np.mean(val_recall), 4), \
                            np.round(np.mean(val_precision), 4), np.round(np.mean(val_f_score), 4), \
                                np.round(np.mean(val_roc_auc_score), 4)))
            print("Epoch {} - Train_Loss: {} - Train_Recall: {} - Train_Precision: {} - Train_F_score: {} - Train_ROC_AUC_score: {} \
                            \n\t- Val_Loss: {} - Val_Recall: {} - Val_Precision: {} - Val_F_score: {} - Val_ROC_AUC_score: {}"\
                    .format(epoch, np.round(losses.mean().item(), 6), np.round(np.mean(train_recall), 4), \
                            np.round(np.mean(train_precision), 4), np.round(np.mean(train_f_score), 4), \
                                np.round(np.mean(train_roc_auc_score), 4), \
                            np.round(val_losses.mean().item(), 6), np.round(np.mean(val_recall), 4), \
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
        # save_root_dist = save_dir+"/dists_"+str(n_epoch)+"_"+str(hp)+"_"+name+"_"+str(datetime.date.today())+".pt"
        # torch.save(dists, save_root_dist)

        # save_root_targets = save_dir+"/targets_"+str(n_epoch)+"_"+str(hp)+"_"+name+"_"+str(datetime.date.today())+".pt"
        # torch.save([repo.targets for repo in train_data] + [repo.targets for repo in val_data], save_root_targets)
      
        # save_root_center = root+'tmp/'+str(datetime.date.today())+"/center_"+str(n_epoch)+"_"+str(hp)+"_"+name+"_"+str(datetime.date.today())+".pt"
        # torch.save(center, save_root_center)
        
        save_root = save_dir+"/model_"+str(n_epoch)+"_"+str(hp)+"_"+name+"_"+str(datetime.date.today())+".pt"
        print('save root:', save_root)
        print('save root:', save_root, file=out_f)
        torch.save(state, save_root)
        
        del model, optimizer, opt_scheduler, losses, train_data, val_data, center, stdev

        # return dists

def bayes_opt(train_GVAE, train_deepSAD, train_data, val_data, b_epoch_GVAE, b_epoch_SAD, \
                b_min:bool, out_f:str, hp_space_GVAE:list, hp_space_SAD:list, name:str, n_calls_GVAE:int, n_calls_SAD:int, bm_dir:str):    
    print("Performing Bayesian Minimization")
    res_gp_GVAE = \
        gp_minimize(partial(train_GVAE, train_data, val_data, b_epoch_GVAE, \
            b_min, out_f, name, bm_dir), hp_space_GVAE, n_calls=n_calls_GVAE)

    print("GVAE Results: \n", res_gp_GVAE.x_iters, res_gp_GVAE.func_vals)
    print("GVAE Results: \n", res_gp_GVAE.x_iters, res_gp_GVAE.func_vals, file=out_f)
    print("Best Hyperparameters GVAE: ", res_gp_GVAE.x, res_gp_GVAE.fun)
    print("Best Hyperparameters GVAE: ", res_gp_GVAE.x, res_gp_GVAE.fun, file=out_f)

    with open(bm_dir + 'best_hp_'+name+'_GVAE.txt', 'wb') as F:
        pickle.dump(res_gp_GVAE.x, F)

    center = torch.load(bm_dir+"/center_"+name+"_"+str(res_gp_GVAE.x)+"_"+str(datetime.date.today())+".pt")
    stdev = torch.load(bm_dir+"/stdev_"+name+"_"+str(res_gp_GVAE.x)+"_"+str(datetime.date.today())+".pt")
    state = torch.load(bm_dir+"/model_"+name+"_"+str(res_gp_GVAE.x)+"_"+str(datetime.date.today())+".pt")
    hp_space_SAD[1] = [res_gp_GVAE.x[1]]
    hp_space_SAD[2] = [res_gp_GVAE.x[2]]
    hp_space_SAD[3] = [res_gp_GVAE.x[3]]
    res_gp_SAD  = \
        gp_minimize(partial(train_deepSAD, train_data, val_data, b_epoch_SAD, \
            b_min, out_f, name, center, stdev, None, state, bm_dir), hp_space_SAD, n_calls=n_calls_SAD)
    
    print("DeepSAD Results: \n", res_gp_SAD.x_iters, res_gp_SAD.func_vals)
    print("DeepSAD Results: \n", res_gp_SAD.x_iters, res_gp_SAD.func_vals, file=out_f)
    print("Best Hyperparameters DeepSAD: ", res_gp_SAD.x, res_gp_SAD.fun)
    print("Best Hyperparameters DeepSAD: ", res_gp_SAD.x, res_gp_SAD.fun, file=out_f)
    
    with open(bm_dir + 'best_hp_'+name+'_SAD.txt', 'wb') as F:
        pickle.dump(res_gp_SAD.x, F)
    return res_gp_GVAE.x, res_gp_SAD.x


def test_GVAESAD(test_data, center, stdev, name, model_path_SAD:str=None, state:str=None, hp_SAD:list=None, plots:bool=False, out_f:str=None, malicious:str='', ret_dist:bool=False):

    learn_r, hidden_features, n_layers, n_aggr, alpha, dropout, eta, nrml_th, anm_th = hp_SAD

    try:
        N = len(test_data.indices)
    except:
        N = test_data.len()
    n_channels = test_data[0].node_features.shape[1]
    e_channels = test_data[0].edge_features.shape[1]
    model = DeepSAD_GVAE(n_channels, hidden_features, n_layers, n_aggr, dropout, e_channels=e_channels).to(device)
    if model_path_SAD:
        model_state = torch.load(model_path_SAD)
        model.load_state_dict(model_state["model"])
    elif state:
        model.load_state_dict(state["model"])
    model.eval()

    test_losses = torch.zeros(N)
    test_dists = []         
    test_recall, test_precision, test_f_score, test_roc_auc_score \
                = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
    # testing
    for r, repo_o in enumerate(test_data):
        repo = copy.deepcopy(repo_o)
        start_time = time.time()
        repo_semi_labels = gen_semi_labels(repo.targets, nrml_th, anm_th)

        model(x = repo.node_features.float().to(device), \
                    edge_index = repo.edge_indices.long().to(device), \
                        edge_attr = repo.edge_features.float().to(device), \
                            centers = center, stdev = stdev)

        loss, test_dist = model.HSClassifierLoss(repo_semi_labels.to(device), eta)            
        test_losses[r] = loss.detach().cpu()            
        test_dists.append(test_dist.detach().cpu())

        # test_dist_tanh = torch.tanh(dists[idx_val]) 
        test_dist_norm = test_dists[r]  
        y_test = repo.targets
        test_recall[r], test_precision[r], test_f_score[r] = \
            f_score_(test_dist_norm.detach().cpu().numpy().flatten(), \
                        y_test.detach().cpu().numpy().flatten(), th=1.)                                
        test_roc_auc_score[r] \
            = roc_auc_score(y_test.detach().cpu().numpy().flatten(), \
                            test_dist_norm.detach().cpu().numpy().flatten())        
           
        del repo_semi_labels, loss, test_dist, repo   

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
            plt.subplot(10, 10, r+1)

            dists = test_dists[r].detach().cpu().numpy()
            targets = repo.targets.detach().cpu().numpy() 
            labels = repo.node_labels.detach().cpu().numpy() 

            x_dists_pos = np.where(labels == 0)[0]
            dists_pos = dists[labels == 0]
            # dists_pos = np.log(dists[labels == 0])
            plt.scatter(x_dists_pos, dists_pos, label=('Normal'))  
            
            x_dists_neg = np.where(labels == 1)[0]
            dists_neg = dists[labels == 1]
            # dists_neg = np.log(dists[labels == 1])
            plt.scatter(x_dists_neg, dists_neg, label=('Anomalous 1'))  

            x_dists_neg = np.where(labels == 2)[0]
            dists_neg = dists[labels == 2]
            # dists_neg = np.log(dists[labels == 1])
            plt.scatter(x_dists_neg, dists_neg, label=('Anomalous 2'))  

            x_dists_neg = np.where(labels == 3)[0]
            dists_neg = dists[labels == 3]
            # dists_neg = np.log(dists[labels == 1])
            plt.scatter(x_dists_neg, dists_neg, label=('Anomalous 3'))  

            x_dists_neg = np.where(labels == 4)[0]
            dists_neg = dists[labels == 4]
            # dists_neg = np.log(dists[labels == 1])
            plt.scatter(x_dists_neg, dists_neg, label=('Anomalous 4'))  

            x_dists_neg = np.where(labels == 5)[0]
            dists_neg = dists[labels == 5]
            # dists_neg = np.log(dists[labels == 1])
            plt.scatter(x_dists_neg, dists_neg, label=('Anomalous 5'))  

            x_dists_label = np.where(labels == 11)[0]
            dists_label = dists[labels == 11]
            # dists_neg = np.log(dists[labels == 1])
            plt.scatter(x_dists_label, dists_label, label=('Malicious'), color='m')  

            x_dists_label = np.where(labels == 12)[0]
            dists_label = dists[labels == 12]
            # dists_neg = np.log(dists[labels == 1])
            plt.scatter(x_dists_label, dists_label, label=('Octopus'), color='y')  

        plt.legend()
        plt.tight_layout()
        plt.savefig("test_distances_"+name+malicious)
        plt.show()
        ''' --------------------------------------- '''
    
    del model, test_losses, test_data 
    
    if ret_dist:
        return np.mean(test_recall), np.mean(test_precision), np.mean(test_f_score), np.mean(test_roc_auc_score), test_dists
    else:
        del test_dists
        return np.mean(test_recall), np.mean(test_precision), np.mean(test_f_score), np.mean(test_roc_auc_score) 


def train_test_multiple(data, mal_data, n_epoch_GVAE, n_epoch_SAD, out_f, name, hp_GVAE, hp_SAD, random_states, t_size, save_dir):
    
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
            center, stdev, _ = train_GVAE(tv_data, n_epoch_GVAE, b_min=False, out_f=out_f, name=name, hp=hp_GVAE, tv_state=tv_state, save_dir=save_dir)

            print(torch.cuda.memory_reserved(0), torch.cuda.memory_allocated(0))

            center = torch.load(save_dir+"/center_"+str(n_epoch_GVAE)+"_"+str(hp_GVAE)+"_"+name+"_"+str(datetime.date.today())+".pt")

            stdev = torch.load(save_dir+"/stdev_"+str(n_epoch_GVAE)+"_"+str(hp_GVAE)+"_"+name+"_"+str(datetime.date.today())+".pt")

            model_path = save_dir+"/model_"+str(n_epoch_GVAE)+"_"+str(hp_GVAE)+"_"+name+"_"+str(datetime.date.today())+".pt"
            dists = train_deepSAD(tv_data, n_epoch_SAD, b_min=False, out_f=out_f, name=name, center=center, stdev=stdev, model_path=model_path, hp=hp_SAD, tv_state=tv_state, save_dir=save_dir)

            del dists 
            print(torch.cuda.memory_reserved(0), torch.cuda.memory_allocated(0))

            model_path_SAD = save_dir+"/model_"+str(n_epoch_SAD)+"_"+str(hp_SAD)+"_"+name+"_"+str(datetime.date.today())+".pt"
            test_recall[tt, tv], test_precision[tt, tv], test_f_score[tt, tv], test_roc_auc_score[tt, tv] \
                = test_GVAESAD(test_data, center, stdev, name, model_path_SAD=model_path_SAD, hp_SAD=hp_SAD, out_f=out_f) 
            
            print(torch.cuda.memory_reserved(0), torch.cuda.memory_allocated(0))

            mal_test_recall[tt, tv], mal_test_precision[tt, tv], mal_test_f_score[tt, tv], mal_test_roc_auc_score[tt, tv] \
                = test_GVAESAD(mal_data, center, stdev, name, model_path_SAD=model_path_SAD, hp_SAD=hp_SAD, out_f=out_f, malicious='malicious')

            print(torch.cuda.memory_reserved(0), torch.cuda.memory_allocated(0))

        del tv_data, test_data, data_c

    return test_recall, test_precision, test_f_score, test_roc_auc_score, \
            mal_test_recall, mal_test_precision, mal_test_f_score, mal_test_roc_auc_score



