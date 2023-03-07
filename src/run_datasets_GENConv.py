import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch_geometric.utils import subgraph
from sklearn.model_selection import train_test_split
from src.train_test_datasets_GENConv import *
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
plots = False

tvt_multiple = True

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

    if b_min:
        ''' Performing Bayesian Minimization '''    
        hp_space_GEN = [[400],
                        Categorical([1e-4, 1e-3], name='learn_r'), 
                        Categorical(np.arange(128, 160, 32), name='hidden_features'),
                        Categorical(np.arange(2, 3, 1), name='n_layers'),
                        Categorical(np.arange(1, 4, 1), name='n_aggr'),
                        Categorical([1e-5, 1e-4, 1e-3, 1e-2], name='alpha'),
                        Categorical(np.arange(0., 0.25, 0.25), name='dropout')]
        
        b_epoch_GEN = 201
        n_calls_GEN = 25

        bm_dir = os.path.join(file_dir, "..", "tmp", "bm", str(datetime.date.today()))
        if not os.path.isdir(bm_dir):
            os.system('mkdir ' + bm_dir)
        out_f = open(bm_dir+"/bm_output_"+str(b_epoch_GEN)+"_"+str(datetime.date.today())+"_"+name+"_"+"GENConv.txt", "a")

        hp_GEN = bayes_opt(train_GEN_Classifier, train_adj_lists, train_features, y_train, b_epoch=b_epoch_GEN, \
                        b_min=b_min, out_f=out_f, hp_space=hp_space_GEN, \
                            name=name, n_calls=n_calls_GEN, bm_dir=bm_dir)
    else:            
        ''' Loading optimum hyperparameters obtained from Bayesian Minimization ''' 
        # with open (bm_dir + '/best_hp_'+name+'_GEN.txt', 'rb') as F:
        #     hp_GEN = pickle.load(F)
        
        hp_GEN = [400, 1e-4, 128, 2, 1, 1e-3, 0]

    n_epoch_GEN = 201
    
    out_f = open(run_dir+"/output_"+str(n_epoch_GEN)+"_"+str(hp_GEN)+"_"+name+"_"+str(datetime.date.today())+"_"+"GENConv.txt", "a")

    print("\nNo. of epochs: ", n_epoch_GEN, "\n", datetime.datetime.now(), "\n", file=out_f)

    if train:
        
        probs = train_GEN_Classifier(train_adj_lists, train_features, y_train, n_epoch_GEN, b_min=False, out_f=out_f, name=name, model_path=None, hp=hp_GEN, tv_state=r_state, save_dir=run_dir)
      

    if plots:            
        ''' Simple Scatter Plot '''
        plt.figure(figsize=(10,10))
        tv_probs = torch.load(run_dir+"/tv_probs_"+str(n_epoch_GEN)+"_"+str(hp_GEN)+"_"+name+"_"+str(datetime.date.today())+".pt")
        tv_labels = torch.load(run_dir+"/labels_"+str(n_epoch_GEN)+"_"+str(hp_GEN)+"_"+name+"_"+str(datetime.date.today())+".pt")        

        probs = tv_probs.detach().cpu().numpy()
        targets = tv_labels.cpu().numpy()
        
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
        plt.savefig("train_val_distances_"+name)
        plt.show()
        ''' --------------------------------------- '''
        


    ''''''''''''' Test '''''''''''''
    if test:
        model_path_GEN = run_dir+"/model_"+str(n_epoch_GEN)+"_"+str(hp_GEN)+"_"+name+"_"+str(datetime.date.today())+".pt"
        test_recall, test_precision, test_f_score, test_roc_auc_score \
            = test_GEN_Classifier(adj_lists, features, labels, idx_test, y_test, name, model_path_GEN=model_path_GEN, hp_GEN=hp_GEN, plots=plots, out_f=out_f)

elif tvt_multiple:
    random_states = [2, 22, 32, 42]
    t_size = 0.6
    if name == 'yelp':
        hp_GEN = [400, 1e-4, 128, 2, 1, 1e-4, 0]
    elif name == 'amazon':
        hp_GEN = [400, 1e-4, 128, 2, 1, 1e-4, 0]

    n_epoch_GEN = 200

    out_f = open(run_dir+"/tvt_output_"+str(random_states)+"_"+str(hp_GEN)+"_"+name+"_"+str(datetime.date.today())+"_"+str(t_size)+"_"+"GENConv.txt", "a")
    recall, precision, f_score, roc_auc_score_ \
        = train_test_multiple(adj, features, labels, n_epoch_GEN, out_f, \
                                name, hp_GEN, random_states, t_size, save_dir=run_dir)
    
    print(f'Average_Recall: {recall.mean().item()}, Average_Precision: {precision.mean().item()}, \
            Average_F_Score: {f_score.mean().item()}, Average_ROC_AUC_Score: {roc_auc_score_.mean().item()}')
    print(f'Average_Recall: {recall.mean().item()}, Average_Precision: {precision.mean().item()}, \
            Average_F_Score: {f_score.mean().item()}, Average_ROC_AUC_Score: {roc_auc_score_.mean().item()}', file=out_f)
    print(f'Recall: {recall}, Precision: {precision}, \
            F_Score: {f_score}, ROC_AUC_Score: {roc_auc_score_}', file=out_f)