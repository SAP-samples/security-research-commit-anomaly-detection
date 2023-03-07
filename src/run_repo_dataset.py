import numpy as np
import pandas as pd
import torch
from src.gen_repo_dataset import *
from torch.utils.data.dataset import random_split
from src.utils.graph_analysis import print_graph_stats 
from sklearn.model_selection import train_test_split
from torch_geometric.utils import subgraph
import os, datetime, copy
from skopt.space import Real, Categorical
from src.train_test_repo_dataset import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



''''''''''' Main '''''''''

name = 'repo'
file_dir = os.path.dirname(os.path.realpath(__file__))
undirected = True
path_gen_graph = os.path.join(file_dir, "..", "data", "repodata")
if undirected:
    path_process_graph = os.path.join(file_dir, "..", "data", "paper", "repodata_undirected")
    path_inject_anom = os.path.join(file_dir, "..", "data", "paper", "repodata_undirected_anom")
    path_labeled_anom = os.path.join(file_dir, "..", "data", "paper", "repodata_labeled_undirected")
else:
    path_process_graph = os.path.join(file_dir, "..", "data", "repodata_directed")
    path_inject_anom = os.path.join(file_dir, "..", "data", "repodata_directed_anom")
    path_labeled_anom = os.path.join(file_dir, "..", "data", "repodata_labeled_directed")
    
if not os.path.isdir(os.path.join(file_dir, "..", "tmp")):
    os.system('mkdir ' + os.path.join(file_dir, "..", "tmp"))

''' Generating Dataset from Git Repos '''
#urls_file_path =  os.path.join(file_dir, "..", "data", "urls_lists", "urls_list_95.txt")
urls_file_path = os.path.join(file_dir, "..", "data", "urls_lists", "toms.txt")
with open(urls_file_path) as f:
        urls_list = f.read().split("\n")
urls_list = list(filter(lambda x: len(x) >2, urls_list))

gen_graph = True
process_graph = True
inject_anom = False
sampling = False
part = "None"
if sampling:
    type_str = "_s_2_3_4"       # empty for all anomalies, _1 for type1, _2 for type2, _3 for type3, _4 for type4, _5 for type5 
else:
    type_str = "_2_3_4"       # empty for all anomalies, _1 for type1, _2 for type2, _3 for type3, _4 for type4, _5 for type5 


if gen_graph or process_graph or inject_anom:
    gen_dataset_out_f = open(file_dir + "/gen_dataset_output_"+str(len(urls_list))+"_"+name+"_"+str(datetime.date.today())+".txt", "a")
    gen_repo_dataset(urls_file_path, gen_graph=gen_graph, path_gen_graph=path_gen_graph, \
                        process_graph=process_graph, path_process_graph=path_process_graph, \
                        inject_anom=inject_anom, path_inject_anom=path_inject_anom, \
                        p_anomals1=0.015, p_anomals2=0.02, p_anomals3=0.02, p_anomals4=0.02, p_anomals5=0.05, \
                        all_types=False, type1=False, type2=True, type3=False, type4=False, type5=False, \
                        undirected=undirected, sampling=sampling, part=part, out_f=gen_dataset_out_f)

''' Loading already generated Dataset with injected anomalies '''
data_list =[]
for url in urls_list:
    data_list.append(path_inject_anom + "/repo_graph_anom"+type_str+"_"+url.split('/')[-1]+".pkl")
    # data_list.append(path_labeled_anom + "/labeled_repo_graph_"+url.split('/')[-1]+".pkl")


# data = Repo_Dataset(len(urls_list))
data = Repo_Dataset_IM(len(urls_list))
data._data_list = data_list
data.load()

print_stats = True
b_min = False
train = False
test = True
plots = False
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
    # GR_GAD_df.to_csv(path_labeled_anom + "/GR-GAD_Statistics.csv")  
    
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
if not os.path.isdir(run_dir):
    os.system('mkdir ' + run_dir)

if not tvt_multiple:  
    # data_c = copy.deepcopy(data)  
    tt_state = 2
    tv_state = 2

    train_data, val_data, test_data = train_val_test_split(data, tt_state, tv_state)
   
    if b_min:
        ''' Performing Bayesian Minimization '''    
        hp_space_GVAE = [[1e-3], 
                        Categorical(np.arange(128, 160, 32), name='hidden_features'),
                        Categorical(np.arange(2, 3, 1), name='n_layers'),
                        Categorical(np.arange(2, 4, 1), name='n_aggr'),
                        Categorical([1e-3, 1e-4, 1e-5], name='alpha'),
                        Categorical(np.arange(0., 0.25, 0.25), name='dropout')]
        hp_space_SAD = [[1e-3], 
                        hp_space_GVAE[2],
                        hp_space_GVAE[3],
                        hp_space_GVAE[4],
                        Categorical([1e-2, 2e-2], name='alpha'),
                        Categorical(np.arange(0., 0.25, 0.25), name='dropout'),
                        Categorical(np.arange(10, 130, 40), name='eta'),
                        Categorical(np.arange(0.05, 0.20, 0.05), name='nrml_th'),
                        Categorical(np.arange(0.05, 0.65, 0.15), name='anm_th')]
        b_epoch_GVAE = 101
        b_epoch_SAD = 101
        n_calls_GVAE = 10
        n_calls_SAD = 50

        bm_dir = os.path.join(file_dir, "..", "tmp", "bm", str(datetime.date.today()))
        if not os.path.isdir(bm_dir):
            os.system('mkdir ' + bm_dir)

        out_f = open(bm_dir + "/bm_output_"+str(b_epoch_GVAE)+"_"+str(b_epoch_SAD)+"_"+str(datetime.date.today())+"_"+name+".txt", "a")

        hp_GVAE, hp_SAD = bayes_opt(train_GVAE, train_deepSAD, train_data, val_data, b_epoch_GVAE, b_epoch_SAD, \
                        b_min=b_min, out_f=out_f, hp_space_GVAE=hp_space_GVAE, hp_space_SAD= hp_space_SAD, \
                            name=name, n_calls_GVAE=n_calls_GVAE, n_calls_SAD=n_calls_SAD, bm_dir=bm_dir)
    else:            
        ''' Loading optimum hyperparameters obtained from Bayesian Minimization ''' 
        # bm_dir = os.path.join(file_dir, "..", "tmp", "bm", str(datetime.date.today()))
        # with open (bm_dir + 'best_hp_'+name+'_GVAE.txt', 'rb') as F:
        #     hp_GVAE = pickle.load(F)
        # with open (bm_dir + 'best_hp_'+name+'_SAD.txt', 'rb') as F:
        #     hp_SAD = pickle.load(F)

        # hp_GVAE = [1e-3, 128, 2, 2, 1e-5, 0.0]
        # hp_SAD = [1e-3, 128, 2, 2, 1e-2, 0.0, 50, 0.05, 0.5]

        # hp_GVAE = [1e-3, 128, 2, 2, 1e-3, 0.0]
        # hp_SAD = [1e-3, 128, 2, 2, 1e-2, 0.0, 50, 0.2, 0.35]

        hp_GVAE = [1e-3, 128, 2, 2, 1e-5, 0]
        hp_SAD = [1e-3, 128, 2, 2, 1e-2, 0.0, 50, 0.05, 0.5]

    n_epoch_GVAE = 100
    n_epoch_SAD = 100
    
    out_f = open(run_dir + "/output_"+str(n_epoch_GVAE)+"_"+str(hp_GVAE)+"_"+name+"_"+str(datetime.date.today())+".txt", "a")

    print("\nNo. of epochs: ", n_epoch_GVAE, "\n", datetime.datetime.now(), "\n", file=out_f)

    if train:

        center, stdev, state = train_GVAE(train_data, val_data, n_epoch_GVAE, b_min=False, out_f=out_f, name=name, hp=hp_GVAE, tv_state=tv_state, save_dir=run_dir)

        center = torch.load(run_dir + "/center_"+str(n_epoch_GVAE)+"_"+str(hp_GVAE)+"_"+name+"_"+str(datetime.date.today())+".pt")

        stdev = torch.load(run_dir + "/stdev_"+str(n_epoch_GVAE)+"_"+str(hp_GVAE)+"_"+name+"_"+str(datetime.date.today())+".pt")

        model_path = run_dir + "/model_"+str(n_epoch_GVAE)+"_"+str(hp_GVAE)+"_"+name+"_"+str(datetime.date.today())+".pt"
        dists = train_deepSAD(train_data, val_data, n_epoch_SAD, b_min=False, out_f=out_f, name=name, center=center, stdev=stdev, model_path=model_path, hp=hp_SAD, tv_state=tv_state, save_dir=run_dir)

        
    # if plots:            
    #     ''' Simple Scatter Plot '''
    #     plt.figure(figsize=(10,10))
    #     tv_dists = torch.load(root+'tmp/'+str(datetime.date.today())+"/dists_"+str(n_epoch_SAD)+"_"+str(hp_SAD)+"_"+name+"_"+str(datetime.date.today())+".pt")
    #     tv_targets = torch.load(root+'tmp/'+str(datetime.date.today())+"/targets_"+str(n_epoch_SAD)+"_"+str(hp_SAD)+"_"+name+"_"+str(datetime.date.today())+".pt")
    #     for r, repo in enumerate(tv_data):
    #         plt.subplot(6, 6, r+1)

    #         dists = tv_dists[r].detach().cpu().numpy()
    #         targets = tv_targets[r].detach().cpu().numpy()        
    #         # dists = torch.tanh(dist)
            
    #         x_dists_pos = np.where(targets == 0)[0]
    #         dists_pos = dists[targets == 0]
    #         # dists_pos = np.log(dists[targets == 0])
    #         plt.scatter(x_dists_pos, dists_pos, label=('Normal'))  
    #         x_dists_neg = np.where(targets == 1)[0]
    #         dists_neg = dists[targets == 1]
    #         # dists_neg = np.log(dists[targets == 1])
    #         plt.scatter(x_dists_neg, dists_neg, label=('Anomalous'))  

    #     plt.legend()
    #     plt.tight_layout()
    #     plt.savefig("train_val_distances_"+name)
    #     plt.show()
        ''' --------------------------------------- '''
        


    ''''''''''''' Test '''''''''''''
    if test:
        center = torch.load(run_dir+"/center_"+str(n_epoch_GVAE)+"_"+str(hp_GVAE)+"_"+name+"_"+str(datetime.date.today())+".pt")

        stdev = torch.load(run_dir+"/stdev_"+str(n_epoch_GVAE)+"_"+str(hp_GVAE)+"_"+name+"_"+str(datetime.date.today())+".pt")

        model_path_SAD = run_dir+"/model_"+str(n_epoch_SAD)+"_"+str(hp_SAD)+"_"+name+"_"+str(datetime.date.today())+".pt"

        test_recall, test_precision, test_f_score, test_roc_auc_score, test_dists \
            = test_GVAESAD(test_data, center, stdev, name, model_path_SAD=model_path_SAD, hp_SAD=hp_SAD, plots=plots, out_f=out_f, ret_dist=True)

    if test_octopus or test_malicious or test_all_malicious:
        torch.cuda.empty_cache()
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
        test_data = Repo_Dataset_IM(len(urls_list))
        test_data._data_list = test_data_list
        test_data.load()


        center = torch.load(run_dir+"/center_"+str(n_epoch_GVAE)+"_"+str(hp_GVAE)+"_"+name+"_"+str(datetime.date.today())+".pt")

        stdev = torch.load(run_dir+"/stdev_"+str(n_epoch_GVAE)+"_"+str(hp_GVAE)+"_"+name+"_"+str(datetime.date.today())+".pt")

        model_path_SAD = run_dir+"/model_"+str(n_epoch_SAD)+"_"+str(hp_SAD)+"_"+name+"_"+str(datetime.date.today())+".pt"

        test_recall, test_precision, test_f_score, test_roc_auc_score, test_dists \
            = test_GVAESAD(test_data, center, stdev, name, model_path_SAD=model_path_SAD, hp_SAD=hp_SAD, plots=plots, out_f=out_f, malicious='malicious', ret_dist=True)

        for r, repo in enumerate(test_data):
            print("\nRepo: ", test_data._data_list[r], " No:", r+1)
            print("\nRepo: ", test_data._data_list[r], " No:", r+1, file=out_f)
            dists = test_dists[r]
            anom_nodes = np.where(repo.targets == 1)[0]
            for an in anom_nodes:
                print("Anomalous Node name: "+str(repo.node_names[an].item())+" Distance: "+str(dists[an].item()))
                print("Anomalous Node name: "+str(repo.node_names[an].item())+" Distance: "+str(dists[an].item()), file=out_f)
            print("Detected Anomalous Nodes: ")
            print("Detected Anomalous Nodes: ", file=out_f)
            for n in range(5):
                t_dists = dists[repo.node_type == n]
                if len(t_dists) != 0:
                    print("Type "+str(n)+": "+str(len(t_dists[t_dists >= 1.])), str(len(t_dists)))
                    print("Type "+str(n)+": "+str(len(t_dists[t_dists >= 1.])), str(len(t_dists)), file=out_f)



elif tvt_multiple:
    random_states = [2, 22, 32, 42]
    t_size = 0.8
    hp_GVAE = [1e-3, 128, 2, 2, 1e-5, 0.0]
    hp_SAD = [1e-3, 128, 2, 2, 1e-2, 0.0, 50, 0.05, 0.35]
    # hp_GVAE = [1e-3, 128, 2, 2, 1e-3, 0]
    # hp_SAD = [1e-3, 128, 2, 2, 1e-2, 0, 50, 0.2, 0.35]

    n_epoch_GVAE = 100
    n_epoch_SAD = 100

    urls_file_path =  os.path.join(file_dir, "..", "data", "urls_lists", "urls_list_all_malicious.txt")   
    with open(urls_file_path) as f:
        urls_list = f.read().split("\n")
    mal_data_list =[]
    for url in urls_list:
        mal_data_list.append(path_labeled_anom + "/labeled_repo_graph_"+url.split('/')[-1]+".pkl")
    # test_data = Repo_Dataset()
    mal_data = Repo_Dataset_IM(len(urls_list))
    mal_data._data_list = mal_data_list
    mal_data.load()

    run_dir = os.path.join(file_dir, "..", "tmp", str(datetime.date.today()))
    out_f = open(run_dir + "/tvt_output_"+str(random_states)+"_"+str(hp_GVAE)+"_"+str(hp_SAD)+"_"+name+"_"+str(datetime.date.today())+".txt", "a")
    recall, precision, f_score, roc_auc_score_, mal_recall, mal_precision, mal_f_score, mal_roc_auc_score_ \
        = train_test_multiple(data, mal_data, n_epoch_GVAE, n_epoch_SAD, out_f, \
                                name, hp_GVAE, hp_SAD, random_states, t_size, save_dir=run_dir)
    
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
