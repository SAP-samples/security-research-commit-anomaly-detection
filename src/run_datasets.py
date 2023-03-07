import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch_geometric.utils import subgraph
from sklearn.model_selection import train_test_split
from src.train_test_datasets import *
import os, datetime
from skopt.space import Real, Categorical
from src.utils.utils import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

''''''''''' Main '''''''''

file_dir = os.path.dirname(os.path.realpath(__file__)) 
if not os.path.isdir(os.path.join(file_dir, "..", "tmp")):
    os.system('mkdir ' + os.path.join(file_dir, "..", "tmp"))
run_dir = os.path.join(file_dir, "..", "tmp", str(datetime.date.today()))
if not os.path.isdir(run_dir):
    os.system('mkdir ' + run_dir)

# name = "amazon"
name = "yelp"

if name == "amazon":
    path = os.path.join(file_dir, "..", "data", "datasets", "Amazon.mat")
elif name == "yelp":
    path = os.path.join(file_dir, "..", "data", "datasets", "YelpChi.mat")

b_min = False
train = True
test = True
plots = True

tvt_multiple = False

adj, features, labels = load_dataset(name, path)

if not tvt_multiple:
    r_state = 2
    if name == 'yelp':
        index = list(range(len(labels)))
        idx_train, idx_test, y_train, y_test \
            = train_test_split(index, labels, stratify=labels, test_size=0.60,
                                    random_state=r_state, shuffle=True)
    elif name == 'amazon':  # amazon
        # 0-3304 are unlabeled nodes
        index = list(range(3305, len(labels)))
        idx_train, idx_test, y_train, y_test \
            = train_test_split(index, labels[3305:], stratify=labels[3305:],
                                test_size=0.60, random_state=r_state, shuffle=True)    
    features = F.normalize(features)

    train_features = features[idx_train]
    adj_lists = convert_to_edgeindices(adj)
    train_adj_lists, _ = subgraph(torch.tensor(idx_train), adj_lists, relabel_nodes=True)
    test_adj_lists, _ = subgraph(torch.tensor(idx_test), adj_lists, relabel_nodes=True)
    # del adj, adj_lists, features, labels

    bm_dir = os.path.join(file_dir, "..", "tmp", "bm", str(datetime.date.today()))
    if b_min:
        ''' Performing Bayesian Minimization '''    
        hp_space_GVAE = [[400],
                        [1e-3], 
                        Categorical(np.arange(128, 160, 32), name='hidden_features'),
                        Categorical(np.arange(2, 3, 1), name='n_layers'),
                        Categorical(np.arange(1, 4, 1), name='n_aggr'),
                        Categorical([1e-5, 1e-4, 1e-3], name='alpha'),
                        Categorical(np.arange(0., 0.5, 0.25), name='dropout')]
        hp_space_SAD = [[400],
                        [1e-3], 
                        hp_space_GVAE[2],
                        hp_space_GVAE[3],
                        hp_space_GVAE[4],
                        Categorical([1e-3, 1e-2], name='alpha'),
                        Categorical(np.arange(0., 0.5, 0.25), name='dropout'),
                        Categorical(np.arange(2, 26, 8), name='eta'),
                        Categorical(np.arange(0.05, 0.20, 0.05), name='nrml_th'),
                        Categorical(np.arange(0.1, 0.7, 0.2), name='anm_th')]
        b_epoch_GVAE = 201
        b_epoch_SAD = 201
        n_calls_GVAE = 20
        n_calls_SAD = 110
        
        if not os.path.isdir(bm_dir):
            os.system('mkdir ' + bm_dir)
        out_f = open(bm_dir+"/bm_output_"+str(b_epoch_GVAE)+"_"+str(b_epoch_SAD)+"_"+str(datetime.date.today())+"_"+name+".txt", "a")

        hp_GVAE, hp_SAD = bayes_opt(train_GVAE, train_deepSAD, train_adj_lists, train_features, y_train, b_epoch_GVAE, b_epoch_SAD, \
                        b_min=b_min, out_f=out_f, hp_space_GVAE=hp_space_GVAE, hp_space_SAD= hp_space_SAD, \
                            name=name, n_calls_GVAE=n_calls_GVAE, n_calls_SAD=n_calls_SAD, bm_dir=bm_dir)
    else:            
        # ''' Loading optimum hyperparameters obtained from Bayesian Minimization ''' 
        # with open (bm_dir + '/best_hp_'+name+'_GVAE.txt', 'rb') as F:
        #     hp_GVAE = pickle.load(F)
        # with open (bm_dir + '/best_hp_'+name+'_SAD.txt', 'rb') as F:
        #     hp_SAD = pickle.load(F)

        hp_GVAE = [1000, 1e-3, 128, 2, 1, 1e-3, 0.0]
        hp_SAD = [1000, 1e-3, 128, 2, 1, 1e-2, 0.0, 10, 0.15, 0.5]

    n_epoch_GVAE = 501
    n_epoch_SAD = 201
    
    out_f = open(run_dir+"/output_"+str(n_epoch_GVAE)+"_"+str(hp_GVAE)+"_"+name+"_"+str(datetime.date.today())+".txt", "a")

    print("\nNo. of epochs: ", n_epoch_GVAE, "\n", datetime.datetime.now(), "\n", file=out_f)

    if train:
        center, stdev, state = train_GVAE(train_adj_lists, train_features, y_train, n_epoch_GVAE, b_min=False, out_f=out_f, name=name, hp=hp_GVAE, tv_state=r_state, save_dir=run_dir)

        center = torch.load(run_dir+"/center_"+str(n_epoch_GVAE)+"_"+str(hp_GVAE)+"_"+name+"_"+str(datetime.date.today())+".pt")

        stdev = torch.load(run_dir+"/stdev_"+str(n_epoch_GVAE)+"_"+str(hp_GVAE)+"_"+name+"_"+str(datetime.date.today())+".pt")

        model_path = run_dir+"/model_"+str(n_epoch_GVAE)+"_"+str(hp_GVAE)+"_"+name+"_"+str(datetime.date.today())+".pt"
        dists = train_deepSAD(train_adj_lists, train_features, y_train, n_epoch_SAD, b_min=False, out_f=out_f, name=name, center=center, stdev=stdev, model_path=model_path, hp=hp_SAD, tv_state=r_state, save_dir=run_dir)

    if plots:            
        ''' Simple Scatter Plot '''
        plt.figure(figsize=(10,10))
        tv_dists = torch.load(run_dir+"/dists_"+str(n_epoch_SAD)+"_"+str(hp_SAD)+"_"+name+"_"+str(datetime.date.today())+".pt")
        tv_labels = torch.load(run_dir+"/labels_"+str(n_epoch_SAD)+"_"+str(hp_SAD)+"_"+name+"_"+str(datetime.date.today())+".pt")        

        dists = tv_dists.detach().cpu().numpy()
        # dists = torch.tanh(dist).detach().cpu().numpy()
        targets = tv_labels.cpu().numpy()
        
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
        plt.savefig("train_val_distances_"+name)
        plt.show()
        ''' --------------------------------------- '''
        


    ''''''''''''' Test '''''''''''''
    if test:
        center = torch.load(run_dir+"/center_"+str(n_epoch_GVAE)+"_"+str(hp_GVAE)+"_"+name+"_"+str(datetime.date.today())+".pt")

        stdev = torch.load(run_dir+"/stdev_"+str(n_epoch_GVAE)+"_"+str(hp_GVAE)+"_"+name+"_"+str(datetime.date.today())+".pt")

        model_path_SAD = run_dir+"/model_"+str(n_epoch_SAD)+"_"+str(hp_SAD)+"_"+name+"_"+str(datetime.date.today())+".pt"
        test_recall, test_precision, test_f_score, test_roc_auc_score \
            = test_GVAESAD(adj_lists, features, labels, idx_test, y_test, center, stdev, name, model_path_SAD=model_path_SAD, hp_SAD=hp_SAD, plots=plots, out_f=out_f)

elif tvt_multiple:
    random_states = [2, 22, 32, 42]
    t_size = 0.2
    if name == 'yelp':
        hp_GVAE = [1000, 1e-3, 128, 2, 1, 1e-3, 0]
        hp_SAD = [1000, 1e-3, 128, 2, 1, 1e-2, 0, 10, 0.15, 0.5]
    elif name == 'amazon':
        hp_GVAE = [1000, 1e-3, 128, 2, 1, 1e-3, 0]
        hp_SAD = [1000, 1e-3, 128, 2, 1, 1e-2, 0, 10, 0.15, 0.5]

    n_epoch_GVAE = 500
    n_epoch_SAD = 200

    out_f = open(run_dir+"/tvt_output_"+str(random_states)+"_"+str(hp_GVAE)+"_"+str(hp_SAD)+"_"+name+"_"+str(datetime.date.today())+"_"+str(t_size)+".txt", "a")
    recall, precision, f_score, roc_auc_score_ \
        = train_test_multiple(adj, features, labels, n_epoch_GVAE, n_epoch_SAD, out_f, \
                                name, hp_GVAE, hp_SAD, random_states, t_size, save_dir=run_dir)
    
    print(f'Average_Recall: {recall.mean().item()}, Average_Precision: {precision.mean().item()}, \
            Average_F_Score: {f_score.mean().item()}, Average_ROC_AUC_Score: {roc_auc_score_.mean().item()}')
    print(f'Average_Recall: {recall.mean().item()}, Average_Precision: {precision.mean().item()}, \
            Average_F_Score: {f_score.mean().item()}, Average_ROC_AUC_Score: {roc_auc_score_.mean().item()}', file=out_f)
    print(f'Recall: {recall}, Precision: {precision}, \
            F_Score: {f_score}, ROC_AUC_Score: {roc_auc_score_}', file=out_f)