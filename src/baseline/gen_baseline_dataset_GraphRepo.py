import yaml, pandas as pd
import numpy as np
import subprocess, time, os, sys
from graphrepo.miners import MineManager
from IPython.display import display
file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(file_dir, '..', ))
from src.utils.gen_git_graph import gen_commit_graph, gen_dev_graph
from pydriller import RepositoryMining

def gen_commit_reports(ocfactors_df, sensfiles_df, filehistory_df, \
                        devs_commits_df, commits_files_df, rules):
    ocfactors_df.timestamp = pd.to_datetime(ocfactors_df.timestamp, unit='s')
    devs_commits_df.timestamp = pd.to_datetime(devs_commits_df.timestamp, unit='s')
    repo_mean_change_prop = ocfactors_df.mean().to_dict()#['fileupdate_nloc', 'fileupdate_added', 
                                # 'fileupdate_token_count', 'fileupdate_removed',
                                # 'no_of_files_modified', 'no_filetypes_modified'].values
    for c_hash in ocfactors_df.commit_hash:
        commit_dict = {}
        commit_dict["commit_hash"] = c_hash
        commit_dict["authors/committers"] = devs_commits_df[devs_commits_df.commit_hash == c_hash].dev_name.values
        commit_dict["authored on/committed on"] = ocfactors_df[ocfactors_df.commit_hash == c_hash].timestamp.values
        commit_dict["commit_message"] = ocfactors_df[ocfactors_df.commit_hash == c_hash].commit_message.values
        commit_dict["no_of_files_modified"] = ocfactors_df[ocfactors_df.commit_hash == c_hash].no_of_files_modified.values

        ''' Generating Outlier Change Properties Rules Values '''
        commit_mean_change_prop = ocfactors_df[ocfactors_df.commit_hash == c_hash].mean().to_dict()
        for (kr, vr), (kc, vc) in zip(repo_mean_change_prop.items(), commit_mean_change_prop.items()):
            if np.abs(vc - vr) > rules["OCP Threshold"]:
                commit_dict[kc] = True
            else:
                commit_dict[kc] = False

        ''' Generating Contrib. Trust Rules Values '''
        curr_devs = devs_commits_df[devs_commits_df.commit_hash == c_hash].dev_hash.values
        for d_hash in curr_devs:
            dev_df = devs_commits_df[devs_commits_df.dev_hash == d_hash]
            dev_df = dev_df.sort_values(by=['timestamp']).reset_index() 
            print(dev_df[dev_df.commit_hash == c_hash].timestamp - dev_df.timestamp.iloc[0])
            if dev_df[dev_df.commit_hash == c_hash].timestamp - dev_df.timestamp.iloc[0] == 0:
                commit_dict["CT_T4_First Commit"] = True


def gen_commit_factors(devs_commits_df, commits_files_df):
    
    ''' Outlier Change Factors:
        'fileupdate_nloc', 'fileupdate_added', 'fileupdate_token_count', 
        'fileupdate_removed', 'no_of_files_modified', 'no_filetypes_modified' '''

    ocfactors_df = commits_files_df.groupby(['project_id', 'commit_is_merge', 'commit_message', 
                        'commit_commit_hash', 'commit_hash', 'timestamp']).count()
    ocfactors_df = ocfactors_df.reset_index()
    ocfactors_df = ocfactors_df[['project_id', 'commit_is_merge', 'commit_message', 
                        'commit_commit_hash', 'commit_hash', 'timestamp', 'file_name']]
    ocfactors_df.rename(columns = {'file_name':'no_of_files_modified'}, inplace = True)

    ocfactors_df['no_filetypes_modified'] = 0
    for h, hash in enumerate(ocfactors_df.commit_hash):
        ocfactors_df['no_filetypes_modified'][h] = commits_files_df[commits_files_df.commit_hash == hash].file_type.nunique()

    commits_files_df = commits_files_df.fillna(0)
    commit_merge_df = commits_files_df.groupby(['project_id', 'commit_is_merge', 'commit_message', 
                        'commit_commit_hash', 'commit_hash', 'timestamp']).sum()
    commit_merge_df = commit_merge_df.reset_index()

    if set(['fileupdate_nloc', 'fileupdate_added', 
                'fileupdate_token_count', 'fileupdate_removed']).issubset(commit_merge_df.columns):        
        ocfactors_df[['fileupdate_nloc', 'fileupdate_added', 
                        'fileupdate_token_count', 'fileupdate_removed']] \
                                = commit_merge_df[['fileupdate_nloc', 'fileupdate_added', 
                                                    'fileupdate_token_count', 'fileupdate_removed']]

    ''' Sensitive Files:
        .xml, .json, .jar, .ini, .dat, .cnf, .yml, .toml, 
        .gradle, .bin, .config, .exe, .properties, .cmd, .build '''
    sensfiles_df = commits_files_df.groupby(['project_id', 'commit_is_merge', 'commit_message', 
                        'commit_commit_hash', 'commit_hash', 'timestamp', 'file_type']).count()
    sensfiles_df = sensfiles_df.reset_index()
    sensfiles_df = sensfiles_df[['project_id', 'commit_is_merge', 'commit_message', 
                        'commit_commit_hash', 'commit_hash', 'timestamp', 'file_type']]

    ''' File History:
        No of commits that modified the file, 
        No of commits from each developer that modified the file'''
    devs_commits_files_df = pd.merge(devs_commits_df, commits_files_df, \
                            on=['commit_dmm_unit_complexity', 'project_id', 'commit_is_merge',
                                'commit_dmm_unit_interfacing', 'commit_message', 'commit_dmm_unit_size',
                                'commit_commit_hash', 'commit_hash', 'timestamp'])
    devs_commits_files_df \
        = devs_commits_files_df[['dev_name', 'dev_email', 'dev_hash', \
                                'project_id', 'commit_message', 'commit_commit_hash', \
                                'commit_hash', 'timestamp', 'file_name', 'file_type', 'file_hash']]
    filehistory_df \
        = devs_commits_files_df.groupby(['dev_name', 'dev_email', 'dev_hash', \
                                'project_id', 'file_name', 'file_type', 'file_hash']).count()
    filehistory_df = filehistory_df.reset_index()
    filehistory_df = filehistory_df[['dev_name', 'dev_email', 'dev_hash', \
                                'project_id', 'file_name', 'file_type', 'file_hash', 'commit_hash']]
    filehistory_df.rename(columns = {'commit_hash':'no_of_commits_per_dev'}, inplace = True)
    filehistory_df = filehistory_df[filehistory_df.file_hash != 0]
    filehistory_df = filehistory_df.reset_index()

    n_commits_df = commits_files_df.groupby(['file_name', 'file_type', 'project_id', 'file_hash']).count()
    n_commits_df = n_commits_df.reset_index()
    n_commits_df.rename(columns = {'commit_hash':'no_of_commits'}, inplace = True)
    n_commits_df = n_commits_df[n_commits_df.file_hash != 0]
    n_commits_df = n_commits_df.reset_index()

    filehistory_df["no_of_commits"] = 0
    for h, hash in enumerate(filehistory_df.file_hash):
        if filehistory_df.file_hash[h] == n_commits_df.file_hash[n_commits_df.file_hash == hash].iloc[0]:
            filehistory_df["no_of_commits"][h] = n_commits_df[n_commits_df.file_hash == hash].no_of_commits.iloc[0]
    filehistory_df = filehistory_df.sort_values(by=['file_hash'])     

    return ocfactors_df, sensfiles_df, filehistory_df

def gen_git_features(config_path, project_id):
    subprocess.Popen("python3 -m examples.index_all --config="+config_path, shell=True).wait()
    # os.system("python3 -m examples.mine_all --config="+config_path)
    
    """ initialize mine manager """
    miner = MineManager(config_path=config_path)
    
    devs_commits_df, devs_files_df, devs_methods_df = gen_dev_graph(miner)
    commits_parents_df, commits_files_df, commits_methods_df = gen_commit_graph(miner)

    return devs_commits_df[devs_commits_df.project_id == project_id], \
            devs_files_df, devs_methods_df, \
                commits_parents_df[commits_parents_df.project_id == project_id], \
                commits_files_df[commits_files_df.project_id == project_id], \
                commits_methods_df[commits_methods_df.project_id == project_id], 


def gen_dataset(urls_file_path:str = "working/graphdata/urls_list.txt", \
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
    
    data_ocfactors_df, data_sensfiles_df, data_filehistory_df, \
        data_devs_commits_df, data_commits_files_df = [], [], [], [], []
    
    for url in urls_list:
        project_id = url.split('/')[-1]
        print('Generating data for project: '+project_id)
        conf['project']['repo'] = url
        conf['project']['project_id'] = project_id
        with open('examples/configs/current_repo.yml', 'w') as y:
            data = yaml.dump(conf, y)    

        # for commit in RepositoryMining(url).traverse_commits():
        #     print(commit.author.name)
        
        ''' Generating graphs for each node type using GraphRepo and Pydriller '''        
        devs_commits_df, devs_files_df, devs_methods_df, \
            commits_parents_df, commits_files_df, commits_methods_df, \
                = gen_git_features(config_path, project_id)

        ''' Extracting factors from the features generated above '''
        ocfactors_df, sensfiles_df, filehistory_df \
            = gen_commit_factors(devs_commits_df, commits_files_df)

        data_ocfactors_df.append(ocfactors_df)
        data_sensfiles_df.append(sensfiles_df) 
        data_filehistory_df.append(filehistory_df)      
        data_devs_commits_df.append(devs_commits_df)      
        data_commits_files_df.append(commits_files_df)      

    np.save("working/Baseline/data_ocfactors_df.npy", np.array(data_ocfactors_df, dtype=object))
    np.save("working/Baseline/data_sensfiles_df.npy", np.array(data_sensfiles_df, dtype=object))
    np.save("working/Baseline/data_filehistory_df.npy", np.array(data_filehistory_df, dtype=object))
    np.save("working/Baseline/data_devs_commits_df.npy", np.array(data_devs_commits_df, dtype=object))
    np.save("working/Baseline/data_commits_files_df.npy", np.array(data_commits_files_df, dtype=object))
    

''' Generating Dataset from Git Repos '''
root = os.path.dirname(os.path.realpath(__file__)) 
urls_file_path = os.path.join(root, "..", "urls_list_5.txt")
with open(urls_file_path) as f:
        urls_list = f.read().split("\n")
# st = time.time()
# gen_dataset(urls_file_path = urls_file_path, \
#             config_path = "examples/configs/current_repo.yml")
# print("\nTime taken to extract "+str(len(urls_list))+" repos", np.round((time.time() - st)/60), " minutes\n")

rules = {
            "DMR Threshold": 0.5,
            "OCP Threshold": 0.5,
            "CT Min. Time as Contributor Threshold": 7,
            "CTR Threshold": 0.5,
            "CT Few Commits Threshold": 0.05,
            "CT Same Day Commit Threshold": 0.2,
            "File History Excluded Files": ["README", ".gitignore"],
            "File History Consider First Commit to File": True,
            "Exclude History for New Contribs.": True,
            "Sensitive Files Threshold": 1,
            "Ownership Excluded Files": [".class", ".md", ".gitignore"],
            "Ownership Consider Major Contributors.": True,
            "Ownership Majority Contrib. Threshold": 0.00,
            "Ownership Min. un-Owned/Majority Files": 0.75
        }

data_ocfactors_df = np.load("working/Baseline/data_ocfactors_df.npy", allow_pickle=True)
data_sensfiles_df = np.load("working/Baseline/data_sensfiles_df.npy", allow_pickle=True)
data_filehistory_df = np.load("working/Baseline/data_filehistory_df.npy", allow_pickle=True)
data_devs_commits_df = np.load("working/Baseline/data_devs_commits_df.npy", allow_pickle=True)
data_commits_files_df = np.load("working/Baseline/data_commits_files_df.npy", allow_pickle=True)

for r in range(len(data_ocfactors_df)):
    gen_commit_reports(data_ocfactors_df[r], data_sensfiles_df[r], \
                        data_filehistory_df[r], data_devs_commits_df[r], \
                            data_commits_files_df[r], rules)