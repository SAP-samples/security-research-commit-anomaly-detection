import numpy as np
import pandas as pd
import torch
from src.gen_repo_dataset import *
from torch.utils.data.dataset import random_split
from src.utils.graph_analysis import print_graph_stats 
import os, datetime, copy
from skopt.space import Real, Categorical
from src.utils.utils import *
from src.train_test_repo_dataset_GENConv import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



''''''''''' Main '''''''''

name = 'repo'
file_dir = os.path.dirname(os.path.realpath(__file__))
undirected = False
path_gen_graph = os.path.join(file_dir, "..", "data", "repodata")
if undirected:
    # path_process_graph = os.path.join(file_dir, "..", "data", "repodata")
    # path_inject_anom = os.path.join(file_dir, "..", "data", "repodata_anom")    
    path_process_graph = os.path.join(file_dir, "..", "data", "repodata_undirected")
    path_inject_anom = os.path.join(file_dir, "..", "data", "repodata_undirected_anom")
    path_labeled_anom = os.path.join(file_dir, "..", "data", "repodata_labeled_undirected")
else:
    path_process_graph = os.path.join(file_dir, "..", "data", "repodata_directed")
    path_inject_anom = os.path.join(file_dir, "..", "data", "repodata_directed_anom")
    path_labeled_anom = os.path.join(file_dir, "..", "data", "repodata_labeled_directed")

if not os.path.isdir(os.path.join(file_dir, "..", "tmp")):
    os.system('mkdir ' + os.path.join(file_dir, "..", "tmp"))

''' Generating Dataset from Git Repos '''
urls_file_path =  os.path.join(file_dir, "..", "data", "urls_lists", "urls_list_90.txt")
with open(urls_file_path) as f:
        urls_list = f.read().split("\n")


gen_graph = False
process_graph = False
inject_anom = False
type_str = "_2_3_4"       # empty for all anomalies, _1 for type1, _2 for type2, _3 for type3, _4 for type4 

if gen_graph or process_graph or inject_anom:
    gen_dataset_out_f = open(os.path.join(file_dir, "..", "tmp", "gen_dataset_output_"+str(len(urls_list))+"_"+name+"_"+str(datetime.date.today())+".txt"), "a")
    gen_repo_dataset(urls_file_path, gen_graph=gen_graph, path_gen_graph=path_gen_graph, \
                        process_graph=process_graph, path_process_graph=path_process_graph, \
                        inject_anom=inject_anom, path_inject_anom=path_inject_anom, \
                        p_anomals1=0.015, p_anomals2=0.02, p_anomals3=0.02, p_anomals4=0.02, \
                        all_types=False, type1=False, type2=True, type3=True, type4=True, type5=False, undirected=undirected, out_f=gen_dataset_out_f)

''' Loading already generated Dataset with injected anomalies '''
data_list =[]
for url in urls_list:
    data_list.append(path_inject_anom + "/repo_graph_anom"+type_str+"_"+url.split('/')[-1]+".pkl")


# data = Repo_Dataset()
data = Repo_Dataset_IM()
data._data_list = data_list
data.load()

print_stats = True
b_min = False
train = True
test = True
plots = True
test_octopus = False
test_malicious = False
test_all_malicious = True

tvt_multiple = False

if print_stats:
    N = data.len()
    GR_GAD_df \
        = pd.DataFrame(columns=['Repository', 'Branches', 'Developers', 'Commits', \
                                'Files', 'Methods', 'No. of Anomalous Nodes', \
                                'Proportion of Anomalous Nodes (%)', 'No. of Existing Edges', \
                                'No. of Possible Edges', 'Sparsity (%)'])

    n_features, n_nodes, anom_ratio, t_edges, n_edges, sparsity, n_edge_features \
        = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
    for g, (url, graph) in enumerate(zip(urls_list, data)):
        GR_GAD_df, n_features[g], n_nodes[g], anom_ratio[g], t_edges[g], n_edges[g], sparsity[g], n_edge_features[g] \
            = print_graph_stats(GR_GAD_df, graph.node_type, graph.node_features, graph.edge_features, graph.targets, g, url, _print=False)
    GR_GAD_df.to_csv(path_inject_anom + "/GR-GAD_Statistics.csv")  

    print("\nNumber of node features: ", int(np.mean(n_features)))
    print("Total number of nodes: ", int(np.sum(n_nodes)))
    print("Average number of nodes: ", int(np.mean(n_nodes)))
    print("Average percentage of anomalous nodes: ", np.mean(anom_ratio))
    print("Minimum percentage of anomalous nodes in a repo: ", np.min(anom_ratio))
    print("Maximum percentage of anomalous nodes in a repo: ", np.max(anom_ratio))
    print("Total number of possible edges: ", int(np.sum(t_edges)))
    print("Total number of existing edges: ", int(np.sum(n_edges)))
    print("Average sparsity: ", np.mean(sparsity))
    print("Number of edge features: ", int(np.mean(n_edge_features)))

run_dir = os.path.join(file_dir, "..", "tmp", str(datetime.date.today()))
# run_dir = os.path.join(file_dir, "..", "tmp", "2022-01-12")
if not os.path.isdir(run_dir):
    os.system('mkdir ' + run_dir)

if not tvt_multiple:  
    data_c = copy.deepcopy(data)  
    r_state = 2
    tv_state = 32

    N = data_c.len()
    tv_len, test_len = int(0.8 * N), N
    tv_data, test_data = random_split(data_c, [tv_len, N-tv_len], generator=torch.Generator().manual_seed(r_state))

    if b_min:
        ''' Performing Bayesian Minimization '''    
        hp_space_GEN = [Categorical([1e-4, 1e-3], name='learn_r'), 
                        Categorical(np.arange(128, 160, 32), name='hidden_features'),
                        Categorical(np.arange(2, 3, 1), name='n_layers'),
                        Categorical(np.arange(1, 4, 1), name='n_aggr'),
                        Categorical([1e-5, 1e-4, 1e-3, 1e-2], name='alpha'),
                        Categorical(np.arange(0., 0.25, 0.25), name='dropout')]
        
        b_epoch_GEN = 101
        n_calls_GEN = 25

        bm_dir = os.path.join(file_dir, "..", "tmp", "bm", str(datetime.date.today()))
        if not os.path.isdir(bm_dir):
            os.system('mkdir ' + bm_dir)

        out_f = open(bm_dir + "/bm_output_"+str(b_epoch_GEN)+"_"+str(datetime.date.today())+"_"+name+"_"+"GENConv.txt", "a")

        hp_GEN = bayes_opt(train_GEN_Classifier, tv_data, b_epoch_GEN, \
                        b_min=b_min, out_f=out_f, hp_space_GEN=hp_space_GEN, \
                            name=name, n_calls_GEN=n_calls_GEN, bm_dir=bm_dir)
    else:            
        ''' Loading optimum hyperparameters obtained from Bayesian Minimization ''' 
        # bm_dir = os.path.join(file_dir, "..", "tmp", "bm", str(datetime.date.today()))
        # with open (bm_dir + 'best_hp_'+name+'_GEN.txt', 'rb') as F:
        #     hp_GEN = pickle.load(F)
        
        hp_GEN = [1e-3, 128, 2, 2, 1e-2, 0.0]

    n_epoch_GEN = 200
    
    out_f = open(run_dir + "/output_"+str(n_epoch_GEN)+"_"+str(hp_GEN)+"_"+name+"_"+str(datetime.date.today())+"_"+"GENConv.txt", "a")
    # out_f = open(run_dir + "/output_201_[0.001, 128, 2, 2, 0.01, 0.0]_repo_2022-01-12_GENConv.txt", "a")


    print("\nNo. of epochs: ", n_epoch_GEN, "\n", datetime.datetime.now(), "\n", file=out_f)

    if train:
        
        probs = train_GEN_Classifier(tv_data, n_epoch_GEN, b_min=False, out_f=out_f, name=name, model_path=None, hp=hp_GEN, tv_state=tv_state, save_dir=run_dir)

        
    # if plots:            
    #     ''' Simple Scatter Plot '''
    #     plt.figure(figsize=(10,10))
    #     tv_probs = torch.load(root+'tmp/'+str(datetime.date.today())+"/tv_probs_"+str(n_epoch_GEN)+"_"+str(hp_GEN)+"_"+name+"_"+str(datetime.date.today())+".pt")
    #     tv_targets = torch.load(root+'tmp/'+str(datetime.date.today())+"/targets_"+str(n_epoch_GEN)+"_"+str(hp_GEN)+"_"+name+"_"+str(datetime.date.today())+".pt")
    #     for r, repo in enumerate(tv_data):
    #         plt.subplot(6, 6, r+1)

    #         probs = tv_probs[r].detach().cpu().numpy()
    #         targets = tv_targets[r].detach().cpu().numpy()        
    #         # probs = torch.tanh(dist)
            
    #         x_probs_pos = np.where(targets == 0)[0]
    #         probs_pos = probs[targets == 0]
    #         # probs_pos = np.log(probs[targets == 0])
    #         plt.scatter(x_probs_pos, probs_pos, label=('Normal'))  
    #         x_probs_neg = np.where(targets == 1)[0]
    #         probs_neg = probs[targets == 1]
    #         # probs_neg = np.log(probs[targets == 1])
    #         plt.scatter(x_probs_neg, probs_neg, label=('Anomalous'))  

    #     plt.legend()
    #     plt.tight_layout()
    #     plt.savefig("train_val_distances_"+name)
    #     plt.show()
        ''' --------------------------------------- '''
        


    ''''''''''''' Test '''''''''''''
    if test:
        
        model_path_GEN = run_dir+"/model_"+str(n_epoch_GEN)+"_"+str(hp_GEN)+"_"+name+"_"+str(datetime.date.today())+".pt"

        test_recall, test_precision, test_f_score, test_roc_auc_score \
            = test_GEN_Classifier(test_data, name, model_path_GEN=model_path_GEN, hp_GEN=hp_GEN, plots=plots, out_f=out_f)

    if test_octopus or test_malicious or test_all_malicious:
       
        if test_octopus:
            urls_file_path =  os.path.join(file_dir, "..", "data", "urls_lists", "urls_list_Octopus.txt")        
        elif test_malicious:
            urls_file_path =  os.path.join(file_dir, "..", "data", "urls_lists", "urls_list_malicious.txt")    
        elif test_all_malicious:
            urls_file_path =  os.path.join(file_dir, "..", "data", "urls_lists", "urls_list_all_malicious.txt")    
        with open(urls_file_path) as f:
            urls_list = f.read().split("\n")

        test_data_list =[]
        for url in urls_list:
            test_data_list.append(path_labeled_anom + "/labeled_repo_graph_"+url.split('/')[-1]+".pkl")


        # test_data = Repo_Dataset()
        test_data = Repo_Dataset_IM()
        test_data._data_list = test_data_list
        test_data.load()
        
        model_path_GEN = run_dir+"/model_"+str(n_epoch_GEN)+"_"+str(hp_GEN)+"_"+name+"_"+str(datetime.date.today())+".pt"
        # model_path_GEN = run_dir+"/model_201_[0.001, 128, 2, 2, 0.01, 0.0]_repo_2022-01-12.pt"

        test_recall, test_precision, test_f_score, test_roc_auc_score \
            = test_GEN_Classifier(test_data, name, model_path_GEN=model_path_GEN, hp_GEN=hp_GEN, plots=plots, out_f=out_f)


elif tvt_multiple:
    random_states = [2, 22, 32, 42]
    t_size = 0.8
    hp_GEN = [1e-3, 128, 2, 2, 1e-2, 0]

    n_epoch_GEN = 200

    urls_file_path =  os.path.join(file_dir, "..", "data", "urls_lists", "urls_list_all_malicious.txt")   
    with open(urls_file_path) as f:
        urls_list = f.read().split("\n")
    mal_data_list =[]
    for url in urls_list:
        mal_data_list.append(path_labeled_anom + "/labeled_repo_graph_"+url.split('/')[-1]+".pkl")
    # test_data = Repo_Dataset()
    mal_data = Repo_Dataset_IM()
    mal_data._data_list = mal_data_list
    mal_data.load()

    run_dir = os.path.join(file_dir, "..", "tmp", str(datetime.date.today()))
    out_f = open(run_dir + "/tvt_output_"+str(random_states)+"_"+str(hp_GEN)+"_"+name+"_"+str(datetime.date.today())+"_"+"GENConv.txt", "a")
    recall, precision, f_score, roc_auc_score_, mal_recall, mal_precision, mal_f_score, mal_roc_auc_score_  \
        = train_test_multiple(data, mal_data, n_epoch_GEN, out_f, \
                                name, hp_GEN, random_states, t_size, save_dir=run_dir)
    
    print(f'Average_Recall: {np.mean(recall)}, Average_Precision: {np.mean(precision)}, \
            Average_F_Score: {np.mean(f_score)}, Average_ROC_AUC_Score: {np.mean(roc_auc_score_)}')
    print(f'Average_Recall: {np.mean(recall)}, Average_Precision: {np.mean(precision)}, \
            Average_F_Score: {np.mean(f_score)}, Average_ROC_AUC_Score: {np.mean(roc_auc_score_)}', file=out_f)
    print(f'Recall: {recall}, Precision: {precision}, \
            F_Score: {f_score}, ROC_AUC_Score: {roc_auc_score_}', file=out_f)
    
    print(f'Average_Recall Malicious: {np.mean(mal_recall)}, Average_Precision Malicious: {np.mean(mal_precision)}, \
            Average_F_Score Malicious: {np.mean(mal_f_score)}, Average_ROC_AUC_Score Malicious: {np.mean(mal_roc_auc_score_)}')
    print(f'Average_Recall Malicious: {np.mean(mal_recall)}, Average_Precision Malicious: {np.mean(mal_precision)}, \
            Average_F_Score Malicious: {np.mean(mal_f_score)}, Average_ROC_AUC_Score Malicious: {np.mean(mal_roc_auc_score_)}', file=out_f)
    print(f'Recall Malicious: {mal_recall}, Precision Malicious: {mal_precision}, \
            F_Score Malicious: {mal_f_score}, ROC_AUC_Score Malicious: {mal_roc_auc_score_}', file=out_f)