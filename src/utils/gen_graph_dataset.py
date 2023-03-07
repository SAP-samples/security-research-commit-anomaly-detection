from cmath import pi
import pickle, math
from networkx import convert
from src.utils.gen_git_graph import *
from src.utils.gen_node_edge_features_all import *
from torch_geometric.utils import sort_edge_index
from src.utils.graph_analysis import *
import yaml, copy
import subprocess
import torch.nn.functional as F
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
# nltk.download('punkt')
import gensim
from gensim.models import Word2Vec
from Levenshtein import distance as lev
import warnings
warnings.simplefilter(action='ignore')

def sort_timestamp(df):
    df = df.sort_values(by='timestamp')
    df = df.reset_index()
    df = df.drop(columns="index")
    return df


''' Converting non_numeric features to numeric '''
def convert_df(df, node, cols_numeric, cols_string):
    start_year = 2005    
    if node:
        df.dev_account_age = [float(d) if not pd.isnull(d) else np.nan for d in df.dev_account_age] 
    if "timestamp" in df.columns:
        df.timestamp = pd.to_datetime(df.timestamp, unit='s').astype('object')
    if not node and "parent_timestamp" in df.columns:
        df.parent_timestamp = pd.to_datetime(df.parent_timestamp, unit='s').astype('object')

    # float_df = df.select_dtypes(include='float64')
    float_df = df[cols_numeric].astype('float64')
    # object_df = df.select_dtypes(include='object')    
    object_df = df[cols_string].astype('object')
    if "timestamp" in df.columns:
        object_df['day_name'] = [d.day_name() if not pd.isnull(d) else pd.NA for d in object_df.timestamp] 
        object_df['month_name'] = [d.month_name() if not pd.isnull(d) else pd.NA for d in object_df.timestamp] 

        float_df['year'] = [d.year if not pd.isnull(d) else np.nan for d in object_df.timestamp] 
        float_df['day_of_year'] = [d.dayofyear if not pd.isnull(d) else np.nan for d in object_df.timestamp] 
        float_df['day'] = np.floor((float_df['year'] - start_year) / 4) * 1461 + \
                            ((float_df['year'] - start_year) % 4) * 365 + float_df['day_of_year']
        float_df = float_df.drop(columns=["year", "day_of_year"])

        float_df['hour'] = [d.hour if not pd.isnull(d) else np.nan for d in object_df.timestamp] 
        float_df['minute'] = [d.minute if not pd.isnull(d) else np.nan for d in object_df.timestamp] 
        float_df['second'] = [d.second if not pd.isnull(d) else np.nan for d in object_df.timestamp] 
        float_df['time'] = (float_df['hour'] + ((float_df['minute'] + (float_df['second'] / 60)) / 60)) / 24
        float_df['time'] = 1 - np.cos(2 * pi * float_df['time'])
        float_df = float_df.drop(columns=["hour", "minute", "second"])  

    if node:        
        float_df = float_df.drop(columns="commit_index")
        if "timestamp" in object_df.columns:
            object_df = object_df.drop(columns=["timestamp"])

        object_df["dev_name"][object_df["dev_name"] == 0] = ""
        object_df["dev_login"][object_df["dev_login"] == 0] = ""
        object_df["dev_location"][object_df["dev_location"] == 0] = ""
        object_df["branch_name"][object_df["branch_name"] == 0] = ""
        object_df["file_name"][object_df["file_name"] == 0] = ""
        object_df["file_type"][object_df["file_type"] == 0] = ""
        object_df["method_name"][object_df["method_name"] == 0] = ""

        float_df["branch_idx"] = np.random.choice(np.arange(1, 1e6, 2), len(object_df.branch_name), replace=False)
        float_df["dev_idx"] = np.random.choice(np.arange(2, 1e6, 2), len(object_df.dev_login), replace=False)

        float_df["lev_branch_hash"] = [float(lev(n,l)) if not pd.isnull(n) and not pd.isnull(l) else pd.NA for (n,l) in zip(object_df.branch_name.values, object_df.branch_hash.values)] 
        float_df["lev_branch_project"] = [float(lev(n,l)) if not pd.isnull(n) and not pd.isnull(l) else pd.NA for (n,l) in zip(object_df.branch_name.values, object_df.project_id.values)] 
        float_df["lev_branch_hash_project"] = [float(lev(n,l)) if not pd.isnull(n) and not pd.isnull(l) else pd.NA for (n,l) in zip(object_df.branch_hash.values, object_df.project_id.values)]         
        
        float_df["lev_dev_name_login"] = [float(lev(n,l)) if not pd.isnull(n) and not pd.isnull(l) else pd.NA for (n,l) in zip(object_df.dev_name.values, object_df.dev_login.values)] 
        float_df["lev_dev_name_location"] = [float(lev(n,l)) if not pd.isnull(n) and not pd.isnull(l) else pd.NA for (n,l) in zip(object_df.dev_name.values, object_df.dev_location.values)] 
        float_df["lev_dev_login_location"] = [float(lev(n,l)) if not pd.isnull(n) and not pd.isnull(l) else pd.NA for (n,l) in zip(object_df.dev_login.values, object_df.dev_location.values)] 
        
        float_df["lev_file_hash"] = [float(lev(n,l)) if not pd.isnull(n) and not pd.isnull(l) else pd.NA for (n,l) in zip(object_df.file_name.values, object_df.file_hash.values)] 
        float_df["lev_file_name_type"] = [float(lev(n,l)) if not pd.isnull(n) and not pd.isnull(l) else pd.NA for (n,l) in zip(object_df.file_name.values, object_df.file_type.values)] 
        float_df["lev_file_hash_type"] = [float(lev(n,l)) if not pd.isnull(n) and not pd.isnull(l) else pd.NA for (n,l) in zip(object_df.file_hash.values, object_df.file_type.values)] 

        float_df["lev_file_method"] = [float(lev(n,l)) if not pd.isnull(n) and not pd.isnull(l) else pd.NA for (n,l) in zip(object_df.file_name.values, object_df.method_name.values)] 
        float_df["lev_method_hash"] = [float(lev(n,l)) if not pd.isnull(n) and not pd.isnull(l) else pd.NA for (n,l) in zip(object_df.method_name.values, object_df.method_hash.values)] 
        float_df["lev_file_method_hash"] = [float(lev(n,l)) if not pd.isnull(n) and not pd.isnull(l) else pd.NA for (n,l) in zip(object_df.file_name.values, object_df.method_hash.values)] 

        object_df = object_df.drop(columns=["branch_name", "branch_hash", "dev_hash", "dev_login", \
                                            "dev_name", "dev_email", "dev_location", "commit_hash", \
                                            "commit_commit_hash", "file_name", "file_hash", "file_type", "method_hash", "method_name", "project_id"])
        # float_df = float_df.drop(columns=['commit_dmm_unit_complexity', 'commit_dmm_unit_interfacing', \
        #                                     'commit_dmm_unit_size'])        

    else:
        if "parent_timestamp" in object_df.columns:                        
            object_df['parent_day_name'] = [d.day_name() if not pd.isnull(d) else pd.NA for d in object_df.parent_timestamp] 
            object_df['parent_month_name'] = [d.month_name() if not pd.isnull(d) else pd.NA for d in object_df.parent_timestamp] 

            float_df['parent_year'] = [d.year if not pd.isnull(d) else np.nan for d in object_df.parent_timestamp] 
            float_df['parent_day_of_year'] = [d.dayofyear if not pd.isnull(d) else np.nan for d in object_df.parent_timestamp] 
            float_df['parent_day'] = np.floor((float_df['parent_year'] - start_year) / 4) * 1461 + \
                                ((float_df['parent_year'] - start_year) % 4) * 365 + float_df['parent_day_of_year']
            float_df = float_df.drop(columns=["parent_year", "parent_day_of_year"])

            float_df['parent_hour'] = [d.hour if not pd.isnull(d) else np.nan for d in object_df.parent_timestamp] 
            float_df['parent_minute'] = [d.minute if not pd.isnull(d) else np.nan for d in object_df.parent_timestamp] 
            float_df['parent_second'] = [d.second if not pd.isnull(d) else np.nan for d in object_df.parent_timestamp] 
            float_df['parent_time'] = (float_df['parent_hour'] + ((float_df['parent_minute'] + (float_df['parent_second'] / 60)) / 60)) / 24
            float_df['parent_time'] = 1 - np.cos(2 * pi * float_df['parent_time'])
            float_df = float_df.drop(columns=["parent_hour", "parent_minute", "parent_second"]) 

            object_df = object_df.drop(columns=["parent_timestamp"])

        object_df["file_name"][object_df["file_name"] == 0] = ""
        float_df["lev_file_project"] = [float(lev(n,l)) if not pd.isnull(n) and not pd.isnull(l) else pd.NA for (n,l) in zip(object_df.file_name.values, object_df.project_id.values)]         
        object_df = object_df.drop(columns=['file_name', "project_id", "timestamp"])

    float_df = float_df.fillna(0)   
    # object_df = object_df.fillna('')  

    w2vec_df = pd.DataFrame()
    # for col in object_df.columns:
        # print(pd.to_numeric(object_df[col].fillna(0), errors='coerce').notnull().all())
        # if not pd.to_numeric(object_df[col].fillna(0), errors='coerce').notnull().all() \
        #     or (pd.to_numeric(object_df[col].fillna(0), errors='coerce') == 0).all():
    w2vec = []
    object_df["commit_message"] = object_df["commit_message"].fillna('')
    object_df["commit_message"][object_df["commit_message"] == 0] = ""
    m_sent = [[[j.lower() for j in word_tokenize(i)] for i in sent_tokenize(m)][0] for m in object_df["commit_message"] if m!='' and str(m)]
    if m_sent !=  []:
        model = Word2Vec(m_sent, min_count = 1, vector_size = 100, window = 5, sg = 0)
        for sent in object_df["commit_message"]:
            if sent != '':
                sent = [[j.lower() for j in word_tokenize(i)] for i in sent_tokenize(sent)][0]                
                w2vec.append(model.wv[sent].mean(axis=0))
            else:
                w2vec.append(np.zeros(100))
    # elif m_sent !=  []:
    #     model = gensim.models.Word2Vec(m_sent, min_count = 1, vector_size = 10, window = 1, sg = 0)                
    #     for sent in object_df[col]:
    #         if sent != '':
    #             sent = [[j.lower() for j in word_tokenize(i)] for i in sent_tokenize(sent)][0]                
    #             w2vec.append(model.wv[sent].mean(axis=0))
    #         else:
    #             w2vec.append(np.zeros(10))
    else:
        w2vec = np.zeros((object_df.shape[0],100))
        
    w2vec_df = pd.DataFrame(w2vec)
        # w2vec_df = pd.concat([w2vec_df, w2vec_df_one], axis=1)
        # else:
        #     float_df[col] = object_df[col].fillna(0).astype(np.float32)


    object_df = object_df.drop(columns=["commit_message"])
    onehot_df = pd.get_dummies(object_df, drop_first=True)

    float_df = float_df.reset_index()
    float_df = float_df.drop(columns="index")
    w2vec_df = w2vec_df.reset_index()
    w2vec_df = w2vec_df.drop(columns="index")
    onehot_df = onehot_df.reset_index()
    onehot_df = onehot_df.drop(columns="index")

    w2vec_df = w2vec_df.fillna(0)   
    # features = torch.tensor(pd.concat([float_df, w2vec_df], axis=1).values.astype(np.float32))
    float_features = torch.tensor(float_df.values.astype(np.float32))
    w2vec_features = torch.tensor(w2vec_df.values.astype(np.float32)) 
    if node:   
        onehot_features = torch.zeros(onehot_df.shape[0], 19)    
        onehot_features[:,:onehot_df.shape[1]] = torch.tensor(onehot_df.values.astype(np.float32))    
    else:
        onehot_features = torch.zeros(onehot_df.shape[0], 38+6)    
        onehot_features[:,:onehot_df.shape[1]] = torch.tensor(onehot_df.values.astype(np.float32))    
        
    # norms = torch.linalg.norm(float_features, dim=0) + 1e-12
    # float_features = float_features / norms
    float_features = F.normalize(float_features)

    features = torch.cat((onehot_features, float_features, w2vec_features), dim=1)
    print(float_df.shape, object_df.shape, onehot_df.shape, w2vec_df.shape, features.shape)

    return features

class Graph_Dataset():
    def __init__(self, data_nodes, data_node_names, data_node_type, data_edge_indices, \
                    data_adj, data_edge_features, data_node_labels=None, data_targets=None):
        self.data_nodes = data_nodes
        self.data_node_names = data_node_names
        self.data_node_type = data_node_type
        self.data_edge_indices = data_edge_indices
        self.data_adj = data_adj
        self.data_edge_features = data_edge_features 
        self.data_node_labels = data_node_labels 
        self.data_targets = data_targets 
        
class Repo_Graph():
    def __init__(self, node_features=None, node_names=None, node_type=None, edge_indices=None, \
                    adj=None, edge_features=None, node_labels=None, targets=None):
        self.node_features = node_features
        self.node_names = node_names
        self.node_type = node_type
        self.edge_indices = edge_indices
        self.adj = adj
        self.edge_features = edge_features 
        self.node_labels = node_labels 
        self.targets = targets 

def gen_one(config_path = "examples/configs/pydriller.yml"):    

    ''' Generating graphs for each node type using GraphRepo and Pydriller '''
    branches_graph_df, commits_parents_df, commits_files_df, \
        commits_methods_df, devs_graph_df, files_graph_df, \
            methods_graph_df = gen_git_graph(config_path)

    ''' Extracting node features, edge indices, edge names and edge features 
        from the graphs generated above '''
    nodes_df, edge_indices, edge_names, edge_features_df \
        = gen_nodes_edges(branches_graph_df, commits_parents_df, commits_files_df, \
                            commits_methods_df, devs_graph_df, files_graph_df, \
                                methods_graph_df)

    ''' Extracting heterogeneous edge indices for plotting '''
    bc_edge_indices_hetero, dc_edge_indices_hetero, cc_edge_indices_hetero, \
    cf_edge_indices_hetero, cm_edge_indices_hetero, fm_edge_indices_hetero \
        = cei_to_hei(nodes_df, edge_indices)

    ''' Plotting '''
    plot_simple_graph(nodes_df, edge_indices)
    plot_hetero_graph(nodes_df, bc_edge_indices_hetero, dc_edge_indices_hetero, \
                        cc_edge_indices_hetero, cf_edge_indices_hetero, \
                            cm_edge_indices_hetero, fm_edge_indices_hetero)
    
    nodes_df.to_pickle("working/graphdata/nodes_df.pkl")
    np.save("working/graphdata/edge_indices.npy", edge_indices)
    np.save("working/graphdata/edge_names.npy", edge_names)
    edge_features_df.to_pickle("working/graphdata/edge_features_df.pkl")


def gen_dataset(urls_file_path:str = "data/repodata/urls_list.txt", \
                config_path:str = "examples/configs/current_repo.yml", \
                data_dir:str = "data/repodata", part:str = "None", \
                out_f:str = "data/repodata/gen_dataset_output.txt"):    

    with open(urls_file_path) as f:
        urls_list = f.read().split("\n")
    urls_split = list(filter(lambda x: len(x) >2, urls_list))
    conf = {'neo': {'db_url': 'localhost', 
                    'port': 7687, 
                    'db_user': 'neo4j', 
                    'db_pwd': 'neo4jj', 
                    'batch_size': 50}, 
            'project': {'repo': '', 
                        'start_date': None, 
                        'end_date': None, 
                        'project_id': '', 
                        'index_code': False, 
                        'index_developer_email': True}}        

    '''To stop dockerd:
        ps -aux | grep dockerd
        sudo kill -9 <id>'''
    subprocess.Popen('sudo dockerd & sleep 10', shell=True, stdout=subprocess.PIPE).wait()
    subprocess.Popen('sudo docker container stop $(sudo docker container list -q)', shell=True).wait()
    subprocess.Popen('sudo docker rm $(sudo docker ps -a -q)', shell=True).wait()
    subprocess.Popen('sudo rm -rf ../neo4j/data', shell=True).wait()
    subprocess.Popen('sudo rm -rf ../neo4j/plugins', shell=True).wait()    
    subprocess.Popen('sudo docker run -p 7474:7474 -p 7687:7687 -v /Users/i534627/neo4j/data:/data -v /Users/i534627/neo4j/plugins:/plugins  -e NEO4JLABS_PLUGINS=\'["apoc"]\'   -e NEO4J_AUTH=neo4j/neo4jj neo4j:3.5.11 & sleep 30', shell=True).wait()    
        
    for url in urls_list:
        project_id = url.split('/')[-1]       
        if len(project_id) < 2:
            continue
#        if part == "_1":
#            if project_id == 'php-src':
#                conf['project']['start_date'] = "15 March, 2021 00:00"   
#                conf['project']['end_date'] = "27 March, 2021 23:59" 
#                project_id = project_id + part      
#            if project_id == 'thegreatsuspender':
#                conf['project']['start_date'] = "01 January, 2005 00:00"   
#                conf['project']['end_date'] = "16 October, 2020 23:59"      
#                project_id = project_id + part      
#            if project_id == 'event-stream':
#                conf['project']['start_date'] = "01 January, 2005 00:00"   
#                conf['project']['end_date'] = "8 September, 2018 23:59"      
#                project_id = project_id + part      
#            if project_id == 'minimap':
#                conf['project']['start_date'] = "01 January, 2005 00:00"   
#                conf['project']['end_date'] = "19 April, 2017 23:59"      
#                project_id = project_id + part 
#            if project_id == 'KeseQul-Desktop-Alpha':
#                conf['project']['start_date'] = "15 September, 2019 00:00"   
#                conf['project']['end_date'] = "24 February, 2021 23:59"      
#                project_id = project_id + part 
#            if project_id == 'gentoo':
#                conf['project']['start_date'] = "15 June, 2018 00:00"   
#                conf['project']['end_date'] = "29 June, 2018 23:59"      
#                project_id = project_id + part      
#            if project_id == 'systemd':
#                conf['project']['start_date'] = "15 June, 2018 00:00"   
#                conf['project']['end_date'] = "29 June, 2018 23:59"      
#                project_id = project_id + part      
#        if part == "_2":
        if project_id == 'php-src':
            conf['project']['start_date'] = "28 March, 2021 00:00"   
            conf['project']['end_date'] = "15 April, 2021 23:59"       
            project_id = project_id      
        if project_id == 'thegreatsuspender':
            conf['project']['start_date'] = "17 October, 2020 00:00"   
            conf['project']['end_date'] = "25 February, 2022 23:59"       
            project_id = project_id      
        if project_id == 'event-stream':
            conf['project']['start_date'] = "9 September, 2018 00:00"   
            conf['project']['end_date'] = "25 February, 2022 23:59"       
            project_id = project_id      
        if project_id == 'minimap':
            conf['project']['start_date'] = "20 April, 2017 00:00"   
            conf['project']['end_date'] = "25 February, 2022 23:59"       
            project_id = project_id 
        if project_id == 'KeseQul-Desktop-Alpha':
            conf['project']['start_date'] = "01 January, 2005 00:00"   
            conf['project']['end_date'] = "14 September, 2019 23:59"      
            project_id = project_id 
        if project_id == 'gentoo':
            conf['project']['start_date'] = "25 June, 2018 00:00"   
            conf['project']['end_date'] = "29 June, 2018 23:59"      
            project_id = project_id      
        if project_id == 'systemd':
            conf['project']['start_date'] = "25 June, 2018 00:00"   
            conf['project']['end_date'] = "29 June, 2018 23:59"   
            project_id = project_id      

        print('Generating graph for project: '+project_id)
        print('Generating graph for project: '+project_id, file=out_f)

        conf['project']['repo'] = url
        conf['project']['project_id'] = project_id


        with open('examples/configs/current_repo.yml', 'w') as y:
            data = yaml.dump(conf, y)    

        ''' Generating graphs for each node type using GraphRepo and Pydriller '''
        print(url, project_id)
        branches_commits_df, branches_files_df, branches_methods_df, \
            devs_commits_df, devs_files_df, devs_methods_df, \
                commits_parents_df, commits_files_df, commits_methods_df, \
                    files_graph_df, methods_graph_df = gen_git_graph(config_path, project_id, url)        
        print('Data extraction from Gihub complete for project: '+project_id, file=out_f)
        
        ''' Extracting node features, edge indices, edge names and edge features 
            from the graphs generated above '''        
        nodes_df, node_names, node_type, edge_indices, edge_features_df \
            = gen_nodes_edges(branches_commits_df, branches_files_df, branches_methods_df, \
                                devs_commits_df, devs_files_df, devs_methods_df, \
                                    commits_parents_df, commits_files_df, commits_methods_df, \
                                        files_graph_df, methods_graph_df)        
        print('Nodes, Edges processing complete for project: '+project_id, file=out_f)
        # if project_id == "ncp":
        #     ''' Extracting heterogeneous edge indices for plotting '''
        #     bc_edge_indices_hetero, dc_edge_indices_hetero, cc_edge_indices_hetero, \
        #     cf_edge_indices_hetero, cm_edge_indices_hetero, fm_edge_indices_hetero \
        #         = cei_to_hei(nodes_df, edge_indices)

        #     ''' Plotting '''
        #     plot_simple_graph(nodes_df, edge_indices)
        #     plot_hetero_graph(nodes_df, bc_edge_indices_hetero, dc_edge_indices_hetero, \
        #                         cc_edge_indices_hetero, cf_edge_indices_hetero, \
        #                             cm_edge_indices_hetero, fm_edge_indices_hetero)      

        node_type = torch.tensor(node_type, dtype=torch.int8)
        edge_indices = torch.tensor(edge_indices)
        adj = torch.zeros((nodes_df.shape[0], nodes_df.shape[0]), dtype=torch.bool)
        adj[edge_indices[0,:], edge_indices[1,:]] = 1
        print(edge_indices.size(), torch.sum(adj)) 

        init_repo_graph = Repo_Graph(nodes_df, node_names, node_type, edge_indices, \
                                adj, edge_features_df)
        pickle.dump(init_repo_graph,  open(data_dir + "/init_repo_graph_"+project_id+".pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
        print('Graph saved for project: '+project_id, file=out_f)


    print('....... All graphs extracted and saved .......')

def convert_to_bi_edges(edge_index, edge_attr, adj, node_type):
    # edge_index, edge_attr = sort_edge_index(edge_index=edge_index, edge_attr=edge_attr)
    # type_adj = torch.zeros(adj.shape)
    # type_adj[node_type.numpy() == 1, :] = 2
    # type_adj[adj == 0] = 0
    # dev_edge_indices = np.argwhere(type_adj == 2)    
    # _, counts = torch.unique(torch.cat((edge_index, dev_edge_indices), dim=1), dim=1, return_counts=True)
    # edge_index_swap, edge_attr_swap = dev_edge_indices.clone(), edge_attr[counts == 2, :].clone()
    # edge_index_swap[0,:] = dev_edge_indices[1,:]
    # edge_index_swap[1,:] = dev_edge_indices[0,:]
    
    edge_index_swap, edge_attr_swap = edge_index.clone(), edge_attr.clone()
    edge_index_swap[0,:] = edge_index[1,:]
    edge_index_swap[1,:] = edge_index[0,:]
    
    edge_index_bi = torch.cat([edge_index, edge_index_swap], dim=1)
    edge_attr_bi = torch.cat([edge_attr, edge_attr_swap], dim=0)
    adj[edge_index_bi[0,:], edge_index_bi[1,:]] = 1
    return edge_index_bi, edge_attr_bi, adj

def add_self_loops(edge_index, edge_attr, adj, node_type):
    self_node_indices = torch.tensor(np.where(node_type.numpy() != np.inf)[0])
    self_edge_index = torch.zeros(2, edge_index.shape[1] + self_node_indices.shape[0], dtype=torch.int64)
    self_edge_index[:, :edge_index.shape[1]] = edge_index.clone()
    self_edge_index[0, edge_index.shape[1]:] = self_node_indices
    self_edge_index[1, edge_index.shape[1]:] = self_node_indices
    self_edge_attr = torch.zeros(edge_index.shape[1] + self_node_indices.shape[0], edge_attr.shape[1])
    self_edge_attr[:edge_index.shape[1], :] = edge_attr.clone()
    self_adj = torch.zeros(adj.shape, dtype=torch.bool)
    self_adj[self_edge_index[0,:], self_edge_index[1,:]] = 1
    return self_edge_index, self_edge_attr, self_adj

def process_dataset(urls_list, path_gen_graph, path_process_graph, undirected=True):    
    ''' Converting non_numeric features to numeric '''
    nodes_numeric = ['no_of_commits', 'no_of_merge_commits', 'no_of_non_merge_commits', 
                        'dev_public_repos', 'dev_public_gists', 'dev_exists', 'dev_account_age',
                        'commit_dmm_unit_complexity', 'commit_is_merge', 'commit_dmm_unit_interfacing', 
                        'commit_dmm_unit_size', 'commit_index', 'no_of_current_and_past_methods']
    nodes_string = ['branch_hash', 'branch_name', 'project_id', 'dev_hash', 'dev_name', 'dev_email', 
                    'dev_login', 'dev_location', 'commit_message', 'commit_commit_hash', 'commit_hash', 
                    'timestamp', 'file_name', 'file_type', 'file_hash', 'method_name', 'method_hash']

    edges_numeric = ['commit_dmm_unit_complexity', 'commit_is_merge', 'commit_dmm_unit_interfacing', 
                        'commit_dmm_unit_size', 'fileupdate_complexity', 'fileupdate_nloc', 'fileupdate_added',
                        'fileupdate_token_count', 'fileupdate_removed', 'methodupdate_complexity', 'methodupdate_nloc',
                        'methodupdate_token_count', 'methodupdate_length', 'methodupdate_fan_in', 'methodupdate_start_line',
                        'methodupdate_general_fan_out', 'methodupdate_end_line','methodupdate_fan_out']
    edges_string = ['timestamp', 'project_id', 'parent_timestamp', 'commit_message', 'fileupdate_type', 'file_name']
    
    for url in urls_list:
        if len(url) < 2 or len(url.split('/')) < 2:
            continue
        repo_graph = pickle.load(open(path_gen_graph + "/init_repo_graph_"+url.split('/')[-1]+".pkl", "rb"))

        repo_graph.node_features = convert_df(repo_graph.node_features, True, nodes_numeric, nodes_string)
        repo_graph.edge_features = convert_df(repo_graph.edge_features, False, edges_numeric, edges_string)

        n_nodes = len(repo_graph.node_features)
        repo_graph.node_labels = torch.zeros(n_nodes, dtype=torch.int8)        
        repo_graph.targets = torch.zeros(n_nodes, dtype=torch.int8)

        if undirected:
            repo_graph.edge_indices, repo_graph.edge_features, repo_graph.adj \
               = convert_to_bi_edges(repo_graph.edge_indices, repo_graph.edge_features, repo_graph.adj, repo_graph.node_type)
        
        repo_graph.edge_indices, repo_graph.edge_features, repo_graph.adj \
             = add_self_loops(repo_graph.edge_indices, repo_graph.edge_features, repo_graph.adj, repo_graph.node_type)
        
        pickle.dump(repo_graph, open(path_process_graph + "/repo_graph_"+url.split('/')[-1]+".pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    
    print('....... All graphs processed and saved .......')
    


def get_node_labels(latent, adj, adj_pred):
    adj_pred_copy = np.where(adj == 1, adj_pred.detach().cpu().numpy(), 100)
    outliers = torch.tensor(np.argwhere((adj_pred_copy <= 0.5) & (adj_pred_copy != 100)).T)
    n_nodes = latent.shape[0]
    node_labels = torch.zeros(n_nodes, dtype=torch.int64)
    for n in range(n_nodes):
        if ((n not in outliers[0,:]) & (n not in outliers[1,:])):
            node_labels[n] = 1
    return node_labels


''' Converting non_numeric features to numeric '''
def convert_df_hetero(df, node, node_type, edge_type):
    float_df = df.select_dtypes(include=['int64', 'float64'])
    object_df = df.select_dtypes(include='object')    
    object_df = object_df.drop(columns=["project_id"])
    
    if node:
        if node_type == "commit":
            df.timestamp = pd.to_datetime(df.timestamp, unit='s').astype('object')
            object_df['day_name'] = [d.day_name() if not pd.isnull(d) else pd.NA for d in df.timestamp] 
            object_df['month_name'] = [d.month_name() if not pd.isnull(d) else pd.NA for d in df.timestamp] 
            object_df['time'] = [d.ctime() if not pd.isnull(d) else pd.NA for d in df.timestamp] 
            object_df = object_df.drop(columns=["commit_hash", "commit_commit_hash"])
            # object_df = object_df.fillna('')    
            
            # m_sent = [[[j.lower() for j in word_tokenize(i)] for i in sent_tokenize(m)][0] for m in object_df.commit_message if m!='']
            # model = gensim.models.Word2Vec(m_sent, min_count = 1, vector_size = 100, window = 5)
            # w2vec = []
            # for message in object_df.commit_message:
            #     if message != '':
            #         sent = [[j.lower() for j in word_tokenize(i)] for i in sent_tokenize(message)][0]                
            #         w2vec.append(model.wv[sent].mean(axis=0))
            #     else:
            #         w2vec.append(np.zeros(100))
            # w2vec_df = pd.DataFrame(w2vec)
            # object_df = object_df.drop(columns=["commit_message"])
        else:
            object_df = object_df.drop(columns=[str(node_type)+"_hash"])
    else:
        if edge_type != "fm" and edge_type != "cc":
            df.timestamp = pd.to_datetime(df.timestamp, unit='s').astype('object')
            object_df['day_name'] = [d.day_name() if not pd.isnull(d) else pd.NA for d in df.timestamp] 
            object_df['month_name'] = [d.month_name() if not pd.isnull(d) else pd.NA for d in df.timestamp] 
            object_df['time'] = [d.ctime() if not pd.isnull(d) else pd.NA for d in df.timestamp] 
            object_df = object_df.drop(columns=["timestamp"])            
        if edge_type == "cc":
            df.parent_timestamp = pd.to_datetime(df.parent_timestamp, unit='s').astype('object')
            object_df['parent_day_name'] = [d.day_name() if not pd.isnull(d) else pd.NA for d in df.parent_timestamp] 
            object_df['parent_month_name'] = [d.month_name() if not pd.isnull(d) else pd.NA for d in df.parent_timestamp] 
            object_df['parent_time'] = [d.ctime() if not pd.isnull(d) else pd.NA for d in df.parent_timestamp] 
            object_df = object_df.drop(columns=["parent_timestamp"])
        # object_df = object_df.fillna('')       
    
    w2vec_df = pd.DataFrame()
    for col in object_df.columns:
        # print(pd.to_numeric(object_df[col].fillna(0), errors='coerce').notnull().all())
        if not pd.to_numeric(object_df[col].fillna(0), errors='coerce').notnull().all():
            object_df[col] = object_df[col].fillna('')
            m_sent = [[[j.lower() for j in word_tokenize(i)] for i in sent_tokenize(m)][0] for m in object_df[col] if m!='']
            model = gensim.models.Word2Vec(m_sent, min_count = 1, vector_size = 100, window = 5)
            w2vec = []
            for sent in object_df[col]:
                if sent != '':
                    sent = [[j.lower() for j in word_tokenize(i)] for i in sent_tokenize(sent)][0]                
                    w2vec.append(model.wv[sent].mean(axis=0))
                else:
                    w2vec.append(np.zeros(100))
            w2vec_df_one = pd.DataFrame(w2vec)
            w2vec_df = pd.concat([w2vec_df, w2vec_df_one], axis=1)
        else:
            float_df[col] = object_df[col].fillna(0).astype(np.float32)

    float_df = float_df.fillna(0)    
    
    # onehot_df = pd.get_dummies(object_df, drop_first=True)

    float_df = float_df.reset_index()
    float_df = float_df.drop(columns="index")
    w2vec_df = w2vec_df.reset_index()
    w2vec_df = w2vec_df.drop(columns="index")
    # onehot_df = onehot_df.reset_index()
    # onehot_df = onehot_df.drop(columns="index")

    # if node and node_type == "commit":
    features = torch.tensor(pd.concat([float_df, w2vec_df], axis=1).to_numpy(), dtype=float)
    # else:
    #     features = torch.tensor(pd.concat([float_df, onehot_df], axis=1).to_numpy(), dtype=float)
    print(float_df.shape, object_df.shape, w2vec_df.shape, features.shape)

    features = features / torch.linalg.norm(features)

    return features


def create_adj(nodes_df_row, nodes_df_col, edge_indices):
    adj = torch.zeros((nodes_df_row.shape[0], nodes_df_col.shape[0]), dtype=int)
    adj[edge_indices[0,:], edge_indices[1,:]] = 1
    return adj

def gen_hetero(urls_file_path:str = "working/graphdata/urls_list.txt", \
                config_path:str = "examples/configs/current_repo.yml"):    

    with open(urls_file_path) as f:
        urls_list = f.read().split("\n")

    conf = {'neo': {'db_url': 'localhost', 
                    'port': 7687, 
                    'db_user': 'neo4j', 
                    'db_pwd': 'neo4jj', 
                    'batch_size': 50}, 
            'project': {'repo': '', 
                        'start_date': None, 
                        'end_date': None, 
                        'project_id': '', 
                        'index_code': False, 
                        'index_developer_email': True}}        

    '''To stop dockerd:
        ps -aux | grep dockerd
        sudo kill -9 <id>'''
    subprocess.Popen('sudo dockerd & sleep 10', shell=True, stdout=subprocess.PIPE).wait()
    subprocess.Popen('sudo docker container stop $(sudo docker container list -q)', shell=True).wait()
    subprocess.Popen('sudo docker rm $(sudo docker ps -a -q)', shell=True).wait()
    subprocess.Popen('sudo rm -rf ../neo4j/data', shell=True).wait()
    subprocess.Popen('sudo rm -rf ../neo4j/plugins', shell=True).wait()    
    subprocess.Popen('sudo docker run -p 7474:7474 -p 7687:7687 -v $HOME/neo4j/data:/data -v $HOME/neo4j/plugins:/plugins  -e NEO4JLABS_PLUGINS=\[\"apoc\"\]   -e NEO4J_AUTH=neo4j/neo4jj neo4j:3.5.11 & sleep 30', shell=True).wait()    
    
    cum_branch_nodes, cum_dev_nodes, cum_commit_nodes, cum_file_nodes, cum_method_nodes, \
    cum_bc_edge_features, cum_dc_edge_features, cum_cc_edge_features, \
    cum_cf_edge_features, cum_cm_edge_features, cum_fm_edge_features, \
    data_node_names, data_adj, data_bc_edge_indices, data_cb_edge_indices, \
    data_dc_edge_indices, data_cd_edge_indices, data_cc_edge_indices, \
    data_cf_edge_indices, data_cm_edge_indices, data_fm_edge_indices, \
    data_bc_adj, data_cb_adj, data_dc_adj, data_cd_adj, data_cc_adj, data_cf_adj, data_cm_adj, data_fm_adj, \
    data_branch_names, data_dev_names, data_commit_names, data_file_names, data_method_names \
        = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], \
            [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
    
    for url in urls_list:
        project_id = url.split('/')[-1]
        print('Generating graph for project: '+project_id)
        conf['project']['repo'] = url
        conf['project']['project_id'] = project_id
        with open('examples/configs/current_repo.yml', 'w') as y:
            data = yaml.dump(conf, y)    

        ''' Generating graphs for each node type using GraphRepo and Pydriller '''
        branches_graph_df, commits_parents_df, commits_files_df, \
            commits_methods_df, devs_graph_df, files_graph_df, \
                methods_graph_df = gen_git_graph(config_path, project_id)

        ''' Extracting node features, edge indices, edge names and edge features 
            from the graphs generated above '''
        nodes_df, node_names, \
        branch_nodes_df, branch_names, \
        dev_nodes_df, dev_names, \
        commit_nodes_df, commit_names, \
        file_nodes_df, file_names, \
        method_nodes_df, method_names, \
        edge_indices, edge_features_df, \
        bc_edge_indices, bc_edge_features_df, \
        dc_edge_indices, dc_edge_features_df, \
        cc_edge_indices, cc_edge_features_df, \
        cf_edge_indices, cf_edge_features_df, \
        cm_edge_indices, cm_edge_features_df, \
        fm_edge_indices, fm_edge_features_df = gen_nodes_edges(branches_graph_df, commits_parents_df, commits_files_df, \
                                commits_methods_df, devs_graph_df, files_graph_df, \
                                    methods_graph_df, hetero=True)        

        edge_indices = torch.tensor(edge_indices)
        adj = torch.zeros((nodes_df.shape[0], nodes_df.shape[0]), dtype=int)
        adj[edge_indices[0,:], edge_indices[1,:]] = 1
        print(edge_indices.size(), torch.sum(adj)) 
        data_adj.append(adj)
        data_node_names.append(node_names)        

        cb_edge_indices = bc_edge_indices.copy()
        cb_edge_indices[[1,0],:] = bc_edge_indices[[0,1],:]
        cd_edge_indices = dc_edge_indices.copy()
        cd_edge_indices[[1,0],:] = dc_edge_indices[[0,1],:]

        data_bc_adj.append(create_adj(branch_nodes_df, commit_nodes_df, bc_edge_indices))
        data_cb_adj.append(create_adj(commit_nodes_df, branch_nodes_df, cb_edge_indices))
        data_dc_adj.append(create_adj(dev_nodes_df, commit_nodes_df, dc_edge_indices))
        data_cd_adj.append(create_adj(commit_nodes_df, dev_nodes_df, cd_edge_indices))
        data_cc_adj.append(create_adj(commit_nodes_df, commit_nodes_df, cc_edge_indices))
        data_cf_adj.append(create_adj(commit_nodes_df, file_nodes_df, cf_edge_indices))
        data_cm_adj.append(create_adj(commit_nodes_df, method_nodes_df, cm_edge_indices))
        data_fm_adj.append(create_adj(file_nodes_df, method_nodes_df, fm_edge_indices))

        cum_branch_nodes.append(branch_nodes_df)
        cum_dev_nodes.append(dev_nodes_df)
        cum_commit_nodes.append(commit_nodes_df)
        cum_file_nodes.append(file_nodes_df)
        cum_method_nodes.append(method_nodes_df)

        cum_bc_edge_features.append(bc_edge_features_df)
        cum_dc_edge_features.append(dc_edge_features_df)
        cum_cc_edge_features.append(cc_edge_features_df)
        cum_cf_edge_features.append(cf_edge_features_df)
        cum_cm_edge_features.append(cm_edge_features_df)
        cum_fm_edge_features.append(fm_edge_features_df)

        data_bc_edge_indices.append(bc_edge_indices)
        data_cb_edge_indices.append(cb_edge_indices)
        data_dc_edge_indices.append(dc_edge_indices)
        data_cd_edge_indices.append(cd_edge_indices)
        data_cc_edge_indices.append(cc_edge_indices)
        data_cf_edge_indices.append(cf_edge_indices)
        data_cm_edge_indices.append(cm_edge_indices)
        data_fm_edge_indices.append(fm_edge_indices)
        
        data_branch_names.append(branch_names)
        data_dev_names.append(dev_names)
        data_commit_names.append(commit_names)
        data_file_names.append(file_names)
        data_method_names.append(method_names)

    torch.save(data_node_names, "working/graphdata/data_node_names"+str(len(urls_list))+"_hetero.pt")
    torch.save(data_adj, "working/graphdata/data_adj"+str(len(urls_list))+"_hetero.pt")

    torch.save(cum_branch_nodes, "working/graphdata/cum_branch_nodes"+str(len(urls_list))+"_hetero.pt")
    torch.save(cum_dev_nodes, "working/graphdata/cum_dev_nodes"+str(len(urls_list))+"_hetero.pt")
    torch.save(cum_commit_nodes, "working/graphdata/cum_commit_nodes"+str(len(urls_list))+"_hetero.pt")
    torch.save(cum_file_nodes, "working/graphdata/cum_file_nodes"+str(len(urls_list))+"_hetero.pt")
    torch.save(cum_method_nodes, "working/graphdata/cum_method_nodes"+str(len(urls_list))+"_hetero.pt")

    torch.save(cum_bc_edge_features, "working/graphdata/cum_bc_edge_features"+str(len(urls_list))+"_hetero.pt")
    torch.save(cum_dc_edge_features, "working/graphdata/cum_dc_edge_features"+str(len(urls_list))+"_hetero.pt")
    torch.save(cum_cc_edge_features, "working/graphdata/cum_cc_edge_features"+str(len(urls_list))+"_hetero.pt")
    torch.save(cum_cf_edge_features, "working/graphdata/cum_cf_edge_features"+str(len(urls_list))+"_hetero.pt")
    torch.save(cum_cm_edge_features, "working/graphdata/cum_cm_edge_features"+str(len(urls_list))+"_hetero.pt")
    torch.save(cum_fm_edge_features, "working/graphdata/cum_fm_edge_features"+str(len(urls_list))+"_hetero.pt")

    torch.save(data_bc_edge_indices, "working/graphdata/data_bc_edge_indices"+str(len(urls_list))+"_hetero.pt")
    torch.save(data_cb_edge_indices, "working/graphdata/data_cb_edge_indices"+str(len(urls_list))+"_hetero.pt")
    torch.save(data_dc_edge_indices, "working/graphdata/data_dc_edge_indices"+str(len(urls_list))+"_hetero.pt")
    torch.save(data_cd_edge_indices, "working/graphdata/data_cd_edge_indices"+str(len(urls_list))+"_hetero.pt")
    torch.save(data_cc_edge_indices, "working/graphdata/data_cc_edge_indices"+str(len(urls_list))+"_hetero.pt")
    torch.save(data_cf_edge_indices, "working/graphdata/data_cf_edge_indices"+str(len(urls_list))+"_hetero.pt")
    torch.save(data_cm_edge_indices, "working/graphdata/data_cm_edge_indices"+str(len(urls_list))+"_hetero.pt")
    torch.save(data_fm_edge_indices, "working/graphdata/data_fm_edge_indices"+str(len(urls_list))+"_hetero.pt")

    torch.save(data_bc_adj, "working/graphdata/data_bc_adj"+str(len(urls_list))+"_hetero.pt")
    torch.save(data_cb_adj, "working/graphdata/data_cb_adj"+str(len(urls_list))+"_hetero.pt")
    torch.save(data_dc_adj, "working/graphdata/data_dc_adj"+str(len(urls_list))+"_hetero.pt")
    torch.save(data_cd_adj, "working/graphdata/data_cd_adj"+str(len(urls_list))+"_hetero.pt")
    torch.save(data_cc_adj, "working/graphdata/data_cc_adj"+str(len(urls_list))+"_hetero.pt")
    torch.save(data_cf_adj, "working/graphdata/data_cf_adj"+str(len(urls_list))+"_hetero.pt")
    torch.save(data_cm_adj, "working/graphdata/data_cm_adj"+str(len(urls_list))+"_hetero.pt")
    torch.save(data_fm_adj, "working/graphdata/data_fm_adj"+str(len(urls_list))+"_hetero.pt")

    torch.save(data_branch_names, "working/graphdata/data_branch_names"+str(len(urls_list))+"_hetero.pt")
    torch.save(data_dev_names, "working/graphdata/data_dev_names"+str(len(urls_list))+"_hetero.pt")
    torch.save(data_commit_names, "working/graphdata/data_commit_names"+str(len(urls_list))+"_hetero.pt")
    torch.save(data_file_names, "working/graphdata/data_file_names"+str(len(urls_list))+"_hetero.pt")
    torch.save(data_method_names, "working/graphdata/data_method_names"+str(len(urls_list))+"_hetero.pt")



def process_hetero(urls_list, 
                    cum_branch_nodes, cum_dev_nodes, cum_commit_nodes, 
                    cum_file_nodes, cum_method_nodes, 
                    cum_bc_edge_features, cum_dc_edge_features, cum_cc_edge_features, 
                    cum_cf_edge_features, cum_cm_edge_features, cum_fm_edge_features,
                    data_bc_edge_indices, data_cb_edge_indices, data_dc_edge_indices, \
                    data_cd_edge_indices, data_cc_edge_indices,
                    data_cf_edge_indices, data_cm_edge_indices, data_fm_edge_indices):    
    ''' Converting non_numeric features to numeric '''
    # cum_branch_node_features = convert_df_hetero(cum_branch_nodes_df, True, "branch", "")
    # cum_dev_node_features = convert_df_hetero(cum_dev_nodes_df, True, "dev", "")
    # cum_commit_node_features = convert_df_hetero(cum_commit_nodes_df, True, "commit", "")
    # cum_file_node_features = convert_df_hetero(cum_file_nodes_df, True, "file", "")
    # cum_method_node_features = convert_df_hetero(cum_method_nodes_df, True, "method", "")

    # cum_bc_edge_features = convert_df_hetero(cum_bc_edge_features_df, False, "", "bc")
    # cum_dc_edge_features = convert_df_hetero(cum_dc_edge_features_df, False, "", "dc")
    # cum_cc_edge_features = convert_df_hetero(cum_cc_edge_features_df, False, "", "cc")
    # cum_cf_edge_features = convert_df_hetero(cum_cf_edge_features_df, False, "", "cf")
    # cum_cm_edge_features = convert_df_hetero(cum_cm_edge_features_df, False, "", "cm")
    # cum_fm_edge_features = convert_df_hetero(cum_fm_edge_features_df, False, "", "fm")

    dataset = []
    for u, url in enumerate(urls_list):
        project_id = url.split('/')[-1]

        data = HeteroData()

        data["branch"].x = convert_df_hetero(cum_branch_nodes[u], True, "branch", "")
        data["dev"].x = convert_df_hetero(cum_dev_nodes[u], True, "dev", "")
        data["commit"].x = convert_df_hetero(cum_commit_nodes[u], True, "commit", "")
        data["file"].x = convert_df_hetero(cum_file_nodes[u], True, "file", "")
        data["method"].x = convert_df_hetero(cum_method_nodes[u], True, "method", "")

        data["branch", "authors", "commit"].edge_attr = convert_df_hetero(cum_bc_edge_features[u], False, "", "bc")
        data["commit", "authored by", "branch"].edge_attr = data["branch", "authors", "commit"].edge_attr 
        data["dev", "authors", "commit"].edge_attr = convert_df_hetero(cum_dc_edge_features[u], False, "", "dc")
        data["commit", "authored by", "dev"].edge_attr = data["dev", "authors", "commit"].edge_attr
        data["commit", "parent", "commit"].edge_attr = convert_df_hetero(cum_cc_edge_features[u], False, "", "cc")
        data["commit", "changes", "file"].edge_attr = convert_df_hetero(cum_cf_edge_features[u], False, "", "cf")
        data["commit", "changes", "method"].edge_attr = convert_df_hetero(cum_cm_edge_features[u], False, "", "cm")
        data["file", "contains", "method"].edge_attr = convert_df_hetero(cum_fm_edge_features[u], False, "", "fm")

        data["branch", "authors", "commit"].edge_index = torch.tensor(data_bc_edge_indices[u])
        data["commit", "authored by", "branch"].edge_index = torch.tensor(data_cb_edge_indices[u])
        data["dev", "authors", "commit"].edge_index = torch.tensor(data_dc_edge_indices[u])
        data["commit", "authored by", "dev"].edge_index = torch.tensor(data_cd_edge_indices[u])
        data["commit", "parent", "commit"].edge_index = torch.tensor(data_cc_edge_indices[u])
        data["commit", "changes", "file"].edge_index = torch.tensor(data_cf_edge_indices[u])
        data["commit", "changes", "method"].edge_index = torch.tensor(data_cm_edge_indices[u])
        data["file", "contains", "method"].edge_index = torch.tensor(data_fm_edge_indices[u])        

        dataset.append(data)
        
    torch.save(dataset, "working/graphdata/dataset"+str(len(urls_list))+"_hetero.pt")

    return dataset

    

   
