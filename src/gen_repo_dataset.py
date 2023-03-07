import torch
import numpy as np
from src.utils.gen_graph_dataset import gen_dataset, process_dataset 
from src.utils.anomaly_injection import inject_anomalies
import sys, os, time
from tqdm import tqdm
np.set_printoptions(threshold=sys.maxsize)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import warnings
warnings.simplefilter(action='ignore')

def gen_repo_dataset(urls_file_path, gen_graph=False, path_gen_graph=None, \
                    process_graph=False, path_process_graph=None, \
                    inject_anom=False, path_inject_anom=None, \
                    p_anomals1=0.015, p_anomals2=0.05, p_anomals3=0.05, p_anomals4=0.05, p_anomals5=0.05, \
                    all_types=True, type1=False, type2=False, type3=False, type4=False, type5=False, \
                    undirected=True, sampling=False, part="None", out_f = "data/repodata/gen_dataset_output.txt"):

    with open(urls_file_path) as f:
            urls_list = f.read().split("\n")

    if gen_graph:
        assert path_gen_graph is not None
        ''' Generating Graphs from Git Repos '''        
        st = time.time()
        gen_dataset(urls_file_path=urls_file_path, \
                    config_path="examples/configs/current_repo.yml", \
                    data_dir=path_gen_graph, part=part, out_f=out_f)
        print("\nTime taken to extract "+str(len(urls_list))+" repos", np.round((time.time() - st)/60), " minutes\n")

    if process_graph:
        assert path_gen_graph is not None and path_process_graph is not None
        ''' Processing already generated graphs '''
        process_dataset(urls_list=urls_list, path_gen_graph=path_gen_graph, path_process_graph=path_process_graph, undirected=undirected)

    if inject_anom:
        assert path_process_graph is not None and path_inject_anom is not None
        ''' Injecting Anomalies'''
        inject_anomalies(urls_list=urls_list, path_process_graph=path_process_graph, path_inject_anom=path_inject_anom, \
                            p_anomals1=p_anomals1, p_anomals2=p_anomals2, p_anomals3=p_anomals3, p_anomals4=p_anomals4, p_anomals5=p_anomals5, \
                            all_types=all_types, type1=type1, type2=type2, type3=type3, type4=type4, type5=type5, sampling=sampling)
    
    

