from re import M
from IPython.core.display import display
import numpy as np
import pandas as pd
import sys, datetime
import warnings
warnings.simplefilter(action='ignore')

from graphrepo.miners import branch

def sort_timestamp(df):
    df = df.sort_values(by='timestamp')
    df = df.reset_index()
    df = df.drop(columns="index")
    return df

'''
    Generating Node Features
                            '''

def gen_branch_nodes(branches_graph_df):
    print('\nGenerating Branch Nodes Features ... ')
    branch_nodes_df = branches_graph_df.groupby(["branch_hash", "branch_name", "project_id"]).count()
    branch_nodes_df = branch_nodes_df.reset_index()
    branch_nodes_df = branch_nodes_df[["branch_hash", "branch_name", "timestamp", "project_id"]]
    # branch_nodes_df["project_id"] = branches_graph_df.project_id.iloc[0]
    branch_nodes_df.rename(columns = {'timestamp':'no_of_commits'}, inplace = True)

    branches_graph_df = branches_graph_df.fillna(0)
    branch_merge_df = branches_graph_df.groupby(["branch_hash", "branch_name", "project_id"]).sum()
    branch_merge_df = branch_merge_df.reset_index()
    branch_nodes_df["no_of_merge_commits"] = branch_merge_df["commit_is_merge"]
    branch_nodes_df["no_of_non_merge_commits"] = branch_nodes_df["no_of_commits"] - branch_merge_df["commit_is_merge"]

    # print("\nBranch node features: \n", branch_nodes_df.columns)
    # display(branch_nodes_df)
    branch_node_type = np.zeros(branch_nodes_df.shape[0], dtype=int)
    branch_names = ['branch_' + b for b in branch_nodes_df.branch_name.tolist()]
    # '''reduced features'''
    # branch_nodes_df = branch_nodes_df[["branch_hash", "branch_name", "project_id"]]
    return branch_nodes_df, branch_names, branch_node_type


def gen_dev_nodes(devs_graph_df):
    print('\nGenerating Developer Nodes Features ... ')
    devs_graph_df = devs_graph_df.fillna(0)
    dev_nodes_df = devs_graph_df.groupby(["dev_hash", "dev_name", "dev_email", "project_id", \
                                            "dev_login", "dev_location", "dev_created_at", \
                                                "dev_public_repos", "dev_public_gists"]).count()
    dev_nodes_df = dev_nodes_df.reset_index()
    dev_nodes_df = dev_nodes_df[["dev_hash", "dev_name", "dev_email", "timestamp", "project_id", \
                                    "dev_login", "dev_location", "dev_created_at", \
                                        "dev_public_repos", "dev_public_gists"]]
    # dev_nodes_df["project_id"] = devs_graph_df.project_id.iloc[0]
    dev_nodes_df.rename(columns = {'timestamp':'no_of_commits'}, inplace = True)

    dev_merge_df = devs_graph_df.groupby(["dev_hash", "dev_name", "dev_email", "project_id", \
                                            "dev_login", "dev_location", "dev_created_at", \
                                                "dev_public_repos", "dev_public_gists"]).sum()
    dev_merge_df = dev_merge_df.reset_index()
    dev_nodes_df["no_of_merge_commits"] = dev_merge_df["commit_is_merge"]
    dev_nodes_df["no_of_non_merge_commits"] = dev_nodes_df["no_of_commits"] - dev_merge_df["commit_is_merge"]

    dev_nodes_df.dev_created_at = pd.to_datetime(dev_nodes_df.dev_created_at)
    dev_nodes_df["dev_exists"] \
        = [1 if not pd.isnull(d) else 0 for d in dev_nodes_df.dev_created_at]
    dev_nodes_df["dev_account_age"] \
        = [(datetime.datetime.utcnow() - d).days if not pd.isnull(d) else pd.NaT for d in dev_nodes_df.dev_created_at]
    dev_nodes_df = dev_nodes_df.drop(columns="dev_created_at")
    # print("\nDeveloper node features: \n", dev_nodes_df.columns)
    # display(dev_nodes_df)
    dev_node_type = np.ones(dev_nodes_df.shape[0], dtype=int)
    dev_names = ['dev_' + b for b in dev_nodes_df.dev_name.tolist()]
    # '''reduced features'''
    # dev_nodes_df = dev_nodes_df[["dev_hash", "dev_name", "dev_email", "project_id"]]
    return dev_nodes_df, dev_names, dev_node_type


def gen_commit_nodes(commits_parents_df, commits_files_df, commits_methods_df):
    print('\nGenerating Commit Nodes Features ... ')
    # print(commits_files_df.columns)
    # display(commits_graph_df)
    commit_nodes_df = commits_files_df.groupby(['commit_dmm_unit_complexity', 'project_id', 'commit_is_merge',
       'commit_dmm_unit_interfacing', 'commit_message', 'commit_dmm_unit_size',
       'commit_commit_hash', 'commit_hash', 'timestamp']).count()
    commit_nodes_df = commit_nodes_df.reset_index()
    # display(commit_nodes_df)
    commit_nodes_df = commit_nodes_df[['commit_dmm_unit_complexity', 'project_id', 'commit_is_merge',
       'commit_dmm_unit_interfacing', 'commit_message', 'commit_dmm_unit_size', 'commit_commit_hash', 'commit_hash', 'timestamp', 'file_name']]
    commit_nodes_df.rename(columns = {'file_name':'no_of_files'}, inplace = True)
    # display(commit_nodes_df)

    # commits_files_df = commits_files_df.fillna(0)
    # commit_merge_df = commits_files_df.groupby(['commit_dmm_unit_complexity', 'project_id', 'commit_is_merge',
    #    'commit_dmm_unit_interfacing', 'commit_message', 'commit_dmm_unit_size',
    #    'commit_commit_hash', 'commit_hash', 'timestamp']).sum()
    # commit_merge_df = commit_merge_df.reset_index()

    # if set(['fileupdate_complexity', 'fileupdate_nloc', 'fileupdate_added', 
    #             'fileupdate_token_count', 'fileupdate_removed']).issubset(commit_merge_df.columns):        
    #     commit_nodes_df[['fileupdate_complexity', 'fileupdate_nloc', 'fileupdate_added', 
    #                         'fileupdate_token_count', 'fileupdate_removed']] \
    #                             = commit_merge_df[['fileupdate_complexity', 'fileupdate_nloc', 
    #                                     'fileupdate_added','fileupdate_token_count', 'fileupdate_removed']]
    # display(commit_nodes_df)

    
    # print(commits_methods_df.columns)
    # display(commits_methods_df)
    # method_count_df = commits_methods_df.groupby(['commit_dmm_unit_complexity', 'project_id', 'commit_is_merge',
    #    'commit_dmm_unit_interfacing', 'commit_message', 'commit_dmm_unit_size',
    #    'commit_commit_hash', 'commit_hash', 'timestamp']).count()
    # method_count_df = method_count_df.reset_index()
    
    # if 'method_name' in method_count_df.columns:
    #     commit_nodes_df["no_of_methods"] = method_count_df[['method_name']]
    # # display(commit_nodes_df)

    # commits_methods_df = commits_methods_df.fillna(0)
    # commit_merge_df = commits_methods_df.groupby(['commit_dmm_unit_complexity', 'project_id', 'commit_is_merge',
    #    'commit_dmm_unit_interfacing', 'commit_message', 'commit_dmm_unit_size',
    #    'commit_commit_hash', 'commit_hash', 'timestamp']).sum()
    # commit_merge_df = commit_merge_df.reset_index()

    # if set(['methodupdate_complexity', 'methodupdate_nloc',
    #    'methodupdate_token_count', 'methodupdate_length',
    #    'methodupdate_fan_in', 'methodupdate_start_line',
    #    'methodupdate_general_fan_out', 'methodupdate_end_line', 
    #    'methodupdate_fan_out']).issubset(commit_merge_df.columns):
    #     commit_nodes_df[['methodupdate_complexity', 'methodupdate_nloc',
    #         'methodupdate_token_count', 'methodupdate_length',
    #         'methodupdate_fan_in', 'methodupdate_start_line',
    #         'methodupdate_general_fan_out', 'methodupdate_end_line', 
    #         'methodupdate_fan_out']] = commit_merge_df[['methodupdate_complexity', 'methodupdate_nloc',
    #                                                             'methodupdate_token_count', 'methodupdate_length',
    #                                                             'methodupdate_fan_in', 'methodupdate_start_line',
    #                                                             'methodupdate_general_fan_out', 'methodupdate_end_line', 'methodupdate_fan_out']]
    
    # print("\nCommit node features: \n", commit_nodes_df.columns)
    # display(commit_nodes_df)
    commit_nodes_df = sort_timestamp(commit_nodes_df)
    commit_nodes_df["commit_index"] = commit_nodes_df.index
    commit_node_type = 2 * np.ones(commit_nodes_df.shape[0], dtype=int)

    commit_names = ['commit_' + b for b in commit_nodes_df.commit_commit_hash.tolist()]
    '''reduced features'''
    commit_nodes_df = commit_nodes_df[['commit_dmm_unit_complexity', 'project_id', 'commit_is_merge',
       'commit_dmm_unit_interfacing', 'commit_message', 'commit_dmm_unit_size', 'commit_commit_hash', 
       'commit_hash', 'timestamp', 'commit_index']]
    return commit_nodes_df, commit_names, commit_node_type


def gen_file_nodes(files_graph_df):
    print('\nGenerating File Nodes Features ... ')
    # print(files_graph_df.columns)  
    # display(files_graph_df)
    file_nodes_df = files_graph_df.groupby(['file_name', 'file_type', 'project_id', 'file_hash']).count()
    file_nodes_df = file_nodes_df.reset_index()

    if 'method_name' in file_nodes_df.columns:
        file_nodes_df = file_nodes_df[["file_name", "file_type", "project_id", 'file_hash', "method_name"]]
        file_nodes_df.rename(columns = {'method_name':'no_of_current_and_past_methods'}, inplace = True) 
    else:
        file_nodes_df = file_nodes_df[["file_name", "file_type", "project_id", 'file_hash']]
    # print("\nFile node features: \n", file_nodes_df.columns)
    # display(file_nodes_df)
    file_node_type = 3 * np.ones(file_nodes_df.shape[0], dtype=int)
    file_names = ['file_' + b for b in file_nodes_df.file_name.tolist()]
    # '''reduced features'''
    # file_nodes_df = file_nodes_df[['file_name', 'file_type', 'project_id', 'file_hash']]
    return file_nodes_df, file_names, file_node_type


def gen_method_nodes(methods_graph_df):
    print('\nGenerating Method Nodes Features ... ')
    # print(methods_graph_df.columns)  
    # display(methods_graph_df)
    method_nodes_df = methods_graph_df.groupby(['method_name', 'file_name', 'project_id', 'method_hash']).count()
    method_nodes_df = method_nodes_df.reset_index()
    method_nodes_df = method_nodes_df[['method_name', 'file_name', 'project_id', 'method_hash']]
    # method_nodes_df.rename(columns = {'method_file_name':'file_name'}, inplace = True) 
    # print("\nMethod node features: \n", method_nodes_df.columns)
    # display(method_nodes_df)
    method_node_type = 4 * np.ones(method_nodes_df.shape[0], dtype=int)
    method_names = ['method_' + b for b in method_nodes_df.method_name.tolist()]
    # '''reduced features'''
    # method_nodes_df = method_nodes_df[['method_name', 'method_file_name', 'project_id', 'method_hash']]
    return method_nodes_df, method_names, method_node_type


def gen_nodes(branches_commits_df, devs_commits_df, \
                    commits_parents_df, commits_files_df, commits_methods_df, \
                        files_graph_df, methods_graph_df, hetero=False):

    branch_nodes_df, branch_names, branch_node_type = gen_branch_nodes(branches_commits_df)
    dev_nodes_df, dev_names, dev_node_type = gen_dev_nodes(devs_commits_df)
    commit_nodes_df, commit_names, commit_node_type = gen_commit_nodes(commits_parents_df, commits_files_df, commits_methods_df)
    file_nodes_df, file_names,file_node_type = gen_file_nodes(files_graph_df)
    method_nodes_df, method_names, method_node_type = gen_method_nodes(methods_graph_df)

    branch_nodes_df.to_csv("working/graphdata/branch_nodes_df.csv")
    dev_nodes_df.to_csv("working/graphdata/dev_nodes_df.csv")
    commit_nodes_df.to_csv("working/graphdata/commit_nodes_df.csv")
    file_nodes_df.to_csv("working/graphdata/file_nodes_df.csv")
    method_nodes_df.to_csv("working/graphdata/method_nodes_df.csv")    

    nodes_df = pd.concat([branch_nodes_df, dev_nodes_df, commit_nodes_df, file_nodes_df, method_nodes_df])
    nodes_df = nodes_df.reset_index()
    nodes_df = nodes_df.drop(columns="index")
    
    nodes_df.to_csv("working/graphdata/nodes_df.csv")    
    # print("\nAll node features: \n", nodes_df.columns)
    # display(nodes_df)    
    node_type = np.concatenate((branch_node_type, dev_node_type, commit_node_type, file_node_type, method_node_type), axis=None)
    node_names = np.concatenate((branch_names, dev_names, commit_names, file_names, method_names), axis=None)

    if hetero:
        return nodes_df, node_names, \
                branch_nodes_df, branch_names, branch_node_type, \
                dev_nodes_df, dev_names, dev_node_type, \
                commit_nodes_df, commit_names, commit_node_type, \
                file_nodes_df, file_names,file_node_type, \
                method_nodes_df, method_names, method_node_type
    else:
        return nodes_df, node_names, node_type


'''
    Generating Edge Indices and Features
                                        '''

def gen_bc_edges(nodes_df, branches_graph_df):
    print('\nGenerating Branch-->Commit Edges Indices, Names and Features ... ')
    # print(branches_graph_df.columns)
    # bc_adj_df = branches_graph_df[["branch_hash", "commit_hash", "timestamp", "project_id"]]
    bc_edge_features_list = ["timestamp", "project_id"]
    bc_edge_indices = np.empty((2,len(branches_graph_df)), dtype=int)
    # bc_edge_names = np.empty((2,len(branches_graph_df)), dtype=object)
    bc_edge_features = np.empty((len(branches_graph_df), len(bc_edge_features_list)), dtype=object)
    for i in range(len(branches_graph_df)):
        ''' Getting edge indices '''
        bc_edge_indices[0,i] = np.where(nodes_df.branch_hash == branches_graph_df.branch_hash.iloc[i])[0]
        bc_edge_indices[1,i] = np.where(nodes_df.commit_hash == branches_graph_df.commit_hash.iloc[i])[0]
        # ''' Getting edge names '''
        # bc_edge_names[0,i] = branches_graph_df.project_id.iloc[i]+': branch_'+branches_graph_df.branch_name.iloc[i]
        # bc_edge_names[1,i] = branches_graph_df.project_id.iloc[i]+': commit_'+branches_graph_df.commit_commit_hash.iloc[i]
        ''' Getting edge features '''        
        bc_edge_features[i,:] = branches_graph_df[bc_edge_features_list].iloc[i]   
    bc_edge_features_df = pd.DataFrame(bc_edge_features, columns=bc_edge_features_list)
    # print("\nBranch-Commit edge features: \n", bc_edge_features_df.columns)
    # display(bc_edge_features_df)

    return bc_edge_indices, bc_edge_features_df

def gen_bf_edges(nodes_df, branches_files_df):
    print('\nGenerating Branch-->File Edges Indices, Names and Features ... ')
    
    bf_edge_features_list = ["branch_name", "project_id"]
    bf_edge_indices = np.empty((2,len(branches_files_df)), dtype=int)
    bf_edge_features = np.empty((len(branches_files_df), len(bf_edge_features_list)), dtype=object)
    # branches_files_df.file_hash = branches_files_df.file_hash.fillna('')
    for i in range(len(branches_files_df)):
        ''' Getting edge indices '''
        bf_edge_indices[0,i] = np.where(nodes_df.branch_hash == branches_files_df.branch_hash.iloc[i])[0]
        bf_edge_indices[1,i] = np.where(nodes_df.file_hash == branches_files_df.file_hash.iloc[i])[0]
        
        ''' Getting edge features '''        
        bf_edge_features[i,:] = branches_files_df[bf_edge_features_list].iloc[i]   
    bf_edge_features_df = pd.DataFrame(bf_edge_features, columns=bf_edge_features_list)

    return bf_edge_indices, bf_edge_features_df

def gen_bm_edges(nodes_df, branches_methods_df):
    print('\nGenerating Branch-->method Edges Indices, Names and Features ... ')
    
    bm_edge_features_list = ["branch_name", "project_id"]
    bm_edge_indices = np.empty((2,len(branches_methods_df)), dtype=int)
    bm_edge_features = np.empty((len(branches_methods_df), len(bm_edge_features_list)), dtype=object)
    for i in range(len(branches_methods_df)):
        ''' Getting edge indices '''
        bm_edge_indices[0,i] = np.where(nodes_df.branch_hash == branches_methods_df.branch_hash.iloc[i])[0]
        bm_edge_indices[1,i] = np.where(nodes_df.method_hash == branches_methods_df.method_hash.iloc[i])[0]
        
        ''' Getting edge features '''        
        bm_edge_features[i,:] = branches_methods_df[bm_edge_features_list].iloc[i]   
    bm_edge_features_df = pd.DataFrame(bm_edge_features, columns=bm_edge_features_list)

    return bm_edge_indices, bm_edge_features_df


def gen_dc_edges(nodes_df, devs_graph_df):
    print('\nGenerating Developer-->Commit Edges Indices, Names and Features ... ')
    # print(devs_graph_df.columns)

    # dc_adj_df = devs_graph_df[["dev_hash", "commit_hash", "timestamp", "project_id"]]
    dc_edge_features_list = ["timestamp", "project_id"]
    dc_edge_indices = np.empty((2,len(devs_graph_df)), dtype=int)
    # dc_edge_names = np.empty((2,len(devs_graph_df)), dtype=object)
    dc_edge_features = np.empty((len(devs_graph_df), len(dc_edge_features_list)), dtype=object)
    for i in range(len(devs_graph_df)):
        ''' Getting edge indices '''
        dc_edge_indices[0,i] = np.where(nodes_df.dev_hash == devs_graph_df.dev_hash.iloc[i])[0]
        dc_edge_indices[1,i] = np.where(nodes_df.commit_hash == devs_graph_df.commit_hash.iloc[i])[0]
        # ''' Getting edge names '''
        # dc_edge_names[0,i] = devs_graph_df.project_id.iloc[i]+': dev_'+devs_graph_df.dev_name.iloc[i]
        # dc_edge_names[1,i] = devs_graph_df.project_id.iloc[i]+': commit_'+devs_graph_df.commit_commit_hash.iloc[i]
        ''' Getting edge features '''        
        dc_edge_features[i,:] = devs_graph_df[dc_edge_features_list].iloc[i]   
    dc_edge_features_df = pd.DataFrame(dc_edge_features, columns=dc_edge_features_list)
    # print("\nDeveloper-Commit edge features: \n", dc_edge_features_df.columns)
    # display(dc_edge_features_df)

    return dc_edge_indices, dc_edge_features_df

def gen_df_edges(nodes_df, devs_files_df):
    print('\nGenerating Developer-->File Edges Indices, Names and Features ... ')

    df_edge_features_list = ["dev_name", "dev_email", "project_id"]
    df_edge_indices = np.empty((2,len(devs_files_df)), dtype=int)
    df_edge_features = np.empty((len(devs_files_df), len(df_edge_features_list)), dtype=object)
    for i in range(len(devs_files_df)):
        ''' Getting edge indices '''
        df_edge_indices[0,i] = np.where(nodes_df.dev_hash == devs_files_df.dev_hash.iloc[i])[0]
        df_edge_indices[1,i] = np.where(nodes_df.file_hash == devs_files_df.file_hash.iloc[i])[0]
        ''' Getting edge features '''
        df_edge_features[i,:] = devs_files_df[df_edge_features_list].iloc[i]
    df_edge_features_df = pd.DataFrame(df_edge_features, columns=df_edge_features_list)

    return df_edge_indices, df_edge_features_df

def gen_dm_edges(nodes_df, devs_methods_df):
    print('\nGenerating Developer-->method Edges Indices, Names and Features ... ')

    dm_edge_features_list = ["dev_name", "dev_email", "project_id"]
    dm_edge_indices = np.empty((2,len(devs_methods_df)), dtype=int)
    dm_edge_features = np.empty((len(devs_methods_df), len(dm_edge_features_list)), dtype=object)
    for i in range(len(devs_methods_df)):
        ''' Getting edge indices '''
        dm_edge_indices[0,i] = np.where(nodes_df.dev_hash == devs_methods_df.dev_hash.iloc[i])[0]
        dm_edge_indices[1,i] = np.where(nodes_df.method_hash == devs_methods_df.method_hash.iloc[i])[0]
        ''' Getting edge features '''
        dm_edge_features[i,:] = devs_methods_df[dm_edge_features_list].iloc[i]
    dm_edge_features_df = pd.DataFrame(dm_edge_features, columns=dm_edge_features_list)

    return dm_edge_indices, dm_edge_features_df


def gen_cc_edges(nodes_df, commits_parents_df):
    print('\nGenerating Parent Commit-->Commit Edges Indices, Names and Features ... ')

    # cc_adj_df = commits_parents_df[["commit_hash", "commit_commit_hash", "parent_hash", "project_id"]]
    cc_edge_features_list = ['project_id', 'parent_timestamp', 'parent_message', 'parent_dmm_unit_complexity', 'parent_is_merge',
       'parent_dmm_unit_interfacing', 'parent_dmm_unit_size']
    cc_edge_indices = np.empty((2,len(commits_parents_df)), dtype=int)
    # cc_edge_names = np.empty((2,len(commits_parents_df)), dtype=object)
    cc_edge_features = np.empty((len(commits_parents_df),len(cc_edge_features_list)), dtype=object)
    # cc_edge_features[:,0] = 1
    commits_parents_df.parent_hash = commits_parents_df.parent_hash.fillna('')
    commits_parents_df = sort_timestamp(commits_parents_df)
    for i in range(len(commits_parents_df)):
        if commits_parents_df.parent_hash.iloc[i]:
            ''' Getting edge indices '''
            cc_edge_indices[0,i] = np.where(nodes_df.commit_hash == commits_parents_df.commit_hash.iloc[i])[0]
            cc_edge_indices[1,i] = np.where(nodes_df.commit_hash == commits_parents_df.parent_hash.iloc[i])[0]
            # ''' Getting edge names '''
            # cc_edge_names[0,i] = commits_parents_df.project_id.iloc[i]+': commit_'+commits_parents_df.parent_commit_hash.iloc[i]
            # cc_edge_names[1,i] = commits_parents_df.project_id.iloc[i]+': commit_'+commits_parents_df.commit_commit_hash.iloc[i]
            ''' Getting edge features ''' 
            # cc_edge_features[i,0] = nodes_df.project_id.iloc[i]
            cc_edge_features[i,:] = commits_parents_df[cc_edge_features_list].iloc[i]
        else:
            cc_edge_indices[:,i] = sys.maxsize
            # cc_edge_names[:,i] = sys.maxsize

    cc_edge_features = cc_edge_features[(cc_edge_indices != sys.maxsize).any(axis=0),:]   
    cc_edge_indices = cc_edge_indices[:, (cc_edge_indices != sys.maxsize).any(axis=0)]   
    # cc_edge_names = cc_edge_names[:, (cc_edge_names != sys.maxsize).any(axis=0)]   
    ''' Getting edge features '''                
    cc_edge_features_df = pd.DataFrame(cc_edge_features, columns=cc_edge_features_list)
    cc_edge_features_df.rename(columns = {'parent_message':'commit_message', 'parent_dmm_unit_complexity':'commit_dmm_unit_complexity', 'parent_is_merge':'commit_is_merge', 'parent_dmm_unit_interfacing':'commit_dmm_unit_interfacing', 'parent_dmm_unit_size':'commit_dmm_unit_size'}, inplace = True)

    # print("\nParentCommit-Commit edge features: \n", cc_edge_features_df.columns)
    # display(cc_edge_features_df)

    return cc_edge_indices, cc_edge_features_df


def gen_cf_edges(nodes_df, commits_files_df):
    print('\nGenerating Commit-->File Edges Indices, Names and Features ... ')
    # cf_adj_df = commits_files_df[['project_id', 'commit_hash', 'timestamp', 'file_name',
    #    'file_hash', 'fileupdate_complexity', 'fileupdate_nloc', 'fileupdate_added', 
    #    'fileupdate_type', 'fileupdate_token_count', 'fileupdate_removed']]
    # print(cf_adj_df.shape)
    cf_edge_features_list = ['project_id', 'timestamp', 'fileupdate_complexity',
       'fileupdate_nloc', 'fileupdate_added', 'fileupdate_type',
       'fileupdate_token_count', 'fileupdate_removed']#, 'file_name', 'file_type']
    cf_edge_indices = np.empty((2,len(commits_files_df)), dtype=int)
    # cf_edge_names = np.empty((2,len(commits_files_df)), dtype=object)
    cf_edge_features = np.empty((len(commits_files_df), len(cf_edge_features_list)), dtype=object)
    commits_files_df.file_hash = commits_files_df.file_hash.fillna('')
    for i in range(len(commits_files_df)):
        if commits_files_df.file_hash.iloc[i]:
            ''' Getting edge indices '''
            cf_edge_indices[0,i] = np.where(nodes_df.commit_hash == commits_files_df.commit_hash.iloc[i])[0]
            cf_edge_indices[1,i] = np.where(nodes_df.file_hash == commits_files_df.file_hash.iloc[i])[0]
            # ''' Getting edge names '''
            # cf_edge_names[0,i] = commits_files_df.project_id.iloc[i]+': commit_'+commits_files_df.commit_commit_hash.iloc[i]
            # cf_edge_names[1,i] = commits_files_df.project_id.iloc[i]+': file_'+commits_files_df.file_name.iloc[i]
            ''' Getting edge features ''' 
            cf_edge_features[i,:] = commits_files_df[cf_edge_features_list].iloc[i]   
        else:
            cf_edge_indices[:,i] = sys.maxsize
            # cf_edge_names[:,i] = sys.maxsize

    cf_edge_features= cf_edge_features[(cf_edge_indices != sys.maxsize).any(axis=0),:]   
    cf_edge_indices= cf_edge_indices[:, (cf_edge_indices != sys.maxsize).any(axis=0)]   
    # cf_edge_names= cf_edge_names[:, (cf_edge_names != sys.maxsize).any(axis=0)]   

    cf_edge_features_df = pd.DataFrame(cf_edge_features, columns=cf_edge_features_list)
    # print("\nCommit-Files edge features: \n", cf_edge_features_df.columns)
    # display(cf_edge_features_df)

    return cf_edge_indices, cf_edge_features_df


def gen_cm_edges(nodes_df, commits_methods_df):
    print('\nGenerating Commit-->Method Edges Indices, Names and Features ... ')
    # cm_adj_df = commits_methods_df[['project_id', 'commit_hash', 'timestamp', 'method_name',
    #    'method_file_name', 'method_hash', 'methodupdate_complexity', 'methodupdate_nloc',
    #    'methodupdate_token_count', 'methodupdate_length', 'methodupdate_fan_in', 'methodupdate_start_line',
    #    'methodupdate_general_fan_out', 'methodupdate_end_line', 'methodupdate_fan_out']]
    # print(cm_adj_df.shape)
    cm_edge_features_list = ['project_id', 'timestamp', 'methodupdate_complexity', 'methodupdate_nloc',
       'methodupdate_token_count', 'methodupdate_length', 'methodupdate_fan_in', 'methodupdate_start_line',
       'methodupdate_general_fan_out', 'methodupdate_end_line', 'methodupdate_fan_out']#, 'method_name', 'file_name']
    cm_edge_indices = np.empty((2,len(commits_methods_df)), dtype=int)
    # cm_edge_names = np.empty((2,len(commits_methods_df)), dtype=object)
    cm_edge_features = np.empty((len(commits_methods_df), len(cm_edge_features_list)), dtype=object)
    commits_methods_df.method_hash = commits_methods_df.method_hash.fillna('')
    for i in range(len(commits_methods_df)):
        # print(commits_methods_df.method_hash.iloc[i], np.sum(commits_methods_df.method_hash == 0), np.sum(nodes_df.method_hash == commits_methods_df.method_hash.iloc[i]))
        if commits_methods_df.method_hash.iloc[i]:
            ''' Getting edge indices '''
            cm_edge_indices[0,i] = np.where(nodes_df.commit_hash == commits_methods_df.commit_hash.iloc[i])[0]
            cm_edge_indices[1,i] = np.where(nodes_df.method_hash == commits_methods_df.method_hash.iloc[i])[0]
            # ''' Getting edge names '''
            # cm_edge_names[0,i] = commits_methods_df.project_id.iloc[i]+': commit_'+commits_methods_df.commit_commit_hash.iloc[i]
            # cm_edge_names[1,i] = commits_methods_df.project_id.iloc[i]+': method_'+commits_methods_df.method_name.iloc[i]
            ''' Getting edge features ''' 
            cm_edge_features[i,:] = commits_methods_df[cm_edge_features_list].iloc[i]   
        else:
            cm_edge_indices[:,i] = sys.maxsize
            # cm_edge_names[:,i] = sys.maxsize
    cm_edge_features= cm_edge_features[(cm_edge_indices != sys.maxsize).any(axis=0),:]   
    cm_edge_indices= cm_edge_indices[:, (cm_edge_indices != sys.maxsize).any(axis=0)]
    # cm_edge_names= cm_edge_names[:, (cm_edge_names != sys.maxsize).any(axis=0)]

    cm_edge_features_df = pd.DataFrame(cm_edge_features, columns=cm_edge_features_list)
    # print("\nCommit-Methods edge features: \n", cm_edge_features_df.columns)
    # display(cm_edge_features_df)

    return cm_edge_indices, cm_edge_features_df


def gen_fm_edges(nodes_df, files_graph_df):
    print('\nGenerating File-->Method Edges Indices, Names and Features ... ')
    
    # fm_adj_df = files_graph_df[['file_name', 'file_type', 'project_id', 'file_hash',
    #                             'method_name', 'method_hash']]
    fm_edge_features_list = ["project_id", 'file_name']
    fm_edge_indices = np.empty((2,len(files_graph_df)), dtype=int)
    # fm_edge_names = np.empty((2,len(files_graph_df)), dtype=object)
    fm_edge_features = np.empty((len(files_graph_df), len(fm_edge_features_list)), dtype=object)
    # fm_edge_features[:,0] = 1
    files_graph_df.method_hash = files_graph_df.method_hash.fillna('')
    for i in range(len(files_graph_df)):
        if files_graph_df.method_hash.iloc[i]:
            ''' Getting edge indices '''
            fm_edge_indices[0,i] = np.where(nodes_df.file_hash == files_graph_df.file_hash.iloc[i])[0]
            fm_edge_indices[1,i] = np.where(nodes_df.method_hash == files_graph_df.method_hash.iloc[i])[0]  
            # ''' Getting edge names '''
            # fm_edge_names[0,i] = files_graph_df.project_id.iloc[i]+': file_'+files_graph_df.file_name.iloc[i]
            # fm_edge_names[1,i] = files_graph_df.project_id.iloc[i]+': method_'+files_graph_df.method_name.iloc[i]   
            ''' Getting edge features '''
            fm_edge_features[i,:] = files_graph_df[fm_edge_features_list].iloc[i]   
            # fm_edge_features[i,1] = nodes_df.project_id.iloc[i]          
        else:
            fm_edge_indices[:,i] = sys.maxsize
            # fm_edge_names[:,i] = sys.maxsize

    fm_edge_features = fm_edge_features[(fm_edge_indices != sys.maxsize).any(axis=0),:]   
    fm_edge_indices= fm_edge_indices[:, (fm_edge_indices != sys.maxsize).any(axis=0)]
    # fm_edge_names= fm_edge_names[:, (fm_edge_names != sys.maxsize).any(axis=0)]
    
    fm_edge_features_df = pd.DataFrame(fm_edge_features, columns=fm_edge_features_list)
    # print("\nFile-Methods edge features: \n", fm_edge_features_df.columns)
    # display(fm_edge_features_df)

    return fm_edge_indices, fm_edge_features_df


def gen_edges(nodes_df, branches_commits_df, branches_files_df, branches_methods_df, \
                devs_commits_df, devs_files_df, devs_methods_df, \
                    commits_parents_df, commits_files_df, commits_methods_df, files_graph_df):

    bc_edge_indices, bc_edge_features_df = gen_bc_edges(nodes_df, branches_commits_df)
    # bf_edge_indices, bf_edge_features_df = gen_bf_edges(nodes_df, branches_files_df)
    # bm_edge_indices, bm_edge_features_df = gen_bm_edges(nodes_df, branches_methods_df)
    dc_edge_indices, dc_edge_features_df = gen_dc_edges(nodes_df, devs_commits_df)
    # df_edge_indices, df_edge_features_df = gen_df_edges(nodes_df, devs_files_df)
    # dm_edge_indices, dm_edge_features_df = gen_dm_edges(nodes_df, devs_methods_df)
    cc_edge_indices, cc_edge_features_df = gen_cc_edges(nodes_df, commits_parents_df)
    cf_edge_indices, cf_edge_features_df = gen_cf_edges(nodes_df, commits_files_df)
    cm_edge_indices, cm_edge_features_df = gen_cm_edges(nodes_df, commits_methods_df)
    fm_edge_indices, fm_edge_features_df = gen_fm_edges(nodes_df, files_graph_df)

    bc_edge_features_df.to_csv("working/graphdata/bc_edge_features_df.csv")
    # bf_edge_features_df.to_csv("working/graphdata/bf_edge_features_df.csv")
    # bm_edge_features_df.to_csv("working/graphdata/bm_edge_features_df.csv")
    dc_edge_features_df.to_csv("working/graphdata/dc_edge_features_df.csv")
    # df_edge_features_df.to_csv("working/graphdata/df_edge_features_df.csv")
    # dm_edge_features_df.to_csv("working/graphdata/dm_edge_features_df.csv")
    cc_edge_features_df.to_csv("working/graphdata/cc_edge_features_df.csv")
    cf_edge_features_df.to_csv("working/graphdata/cf_edge_features_df.csv")
    cm_edge_features_df.to_csv("working/graphdata/cm_edge_features_df.csv")
    fm_edge_features_df.to_csv("working/graphdata/fm_edge_features_df.csv")

    edge_indices = np.concatenate((bc_edge_indices, #bf_edge_indices, bm_edge_indices, 
                                    dc_edge_indices, #df_edge_indices, dm_edge_indices, \
                                        cc_edge_indices, cf_edge_indices, cm_edge_indices, fm_edge_indices), axis=1)
    edge_features_df = pd.concat([bc_edge_features_df, #bf_edge_features_df, bm_edge_features_df, \
                                    dc_edge_features_df, #df_edge_features_df, dm_edge_features_df, \
                                        cc_edge_features_df, cf_edge_features_df, cm_edge_features_df, fm_edge_features_df])

    edge_features_df.to_csv("working/graphdata/edge_features_df.csv")
    # print("\nAll edge features: \n", edge_features_df.columns)
    # display(edge_features_df)    
    
    return edge_indices, edge_features_df 


'''
    Generating Heterogeneous Edge Indices and Features
                                        '''

def gen_bc_edges_hetero(branch_nodes_df, commit_nodes_df, branches_graph_df):
    print('\nGenerating Branch-->Commit Edges Indices, Names and Features ... ')
    # print(branches_graph_df.columns)
    bc_edge_features_list = ["timestamp", "project_id"]
    bc_edge_indices = np.empty((2,len(branches_graph_df)), dtype=int)
    bc_edge_features = np.empty((len(branches_graph_df), len(bc_edge_features_list)), dtype=object)
    for i in range(len(branches_graph_df)):
        ''' Getting edge indices '''
        bc_edge_indices[0,i] = np.where(branch_nodes_df.branch_hash == branches_graph_df.branch_hash.iloc[i])[0]
        bc_edge_indices[1,i] = np.where(commit_nodes_df.commit_hash == branches_graph_df.commit_hash.iloc[i])[0]
        ''' Getting edge features '''        
        bc_edge_features[i,:] = branches_graph_df[bc_edge_features_list].iloc[i]   
    bc_edge_features_df = pd.DataFrame(bc_edge_features, columns=bc_edge_features_list)
    # print("\nBranch-Commit edge features: \n", bc_edge_features_df.columns)
    # display(bc_edge_features_df)

    return bc_edge_indices, bc_edge_features_df


def gen_dc_edges_hetero(dev_nodes_df, commit_nodes_df, devs_graph_df):
    print('\nGenerating Developer-->Commit Edges Indices, Names and Features ... ')
    # print(devs_graph_df.columns)

    dc_edge_features_list = ["timestamp", "project_id"]
    dc_edge_indices = np.empty((2,len(devs_graph_df)), dtype=int)
    dc_edge_features = np.empty((len(devs_graph_df), len(dc_edge_features_list)), dtype=object)
    for i in range(len(devs_graph_df)):
        ''' Getting edge indices '''
        dc_edge_indices[0,i] = np.where(dev_nodes_df.dev_hash == devs_graph_df.dev_hash.iloc[i])[0]
        dc_edge_indices[1,i] = np.where(commit_nodes_df.commit_hash == devs_graph_df.commit_hash.iloc[i])[0]
        ''' Getting edge features '''        
        dc_edge_features[i,:] = devs_graph_df[dc_edge_features_list].iloc[i]   
    dc_edge_features_df = pd.DataFrame(dc_edge_features, columns=dc_edge_features_list)
    # print("\nDeveloper-Commit edge features: \n", dc_edge_features_df.columns)
    # display(dc_edge_features_df)

    return dc_edge_indices, dc_edge_features_df


def gen_cc_edges_hetero(commit_nodes_df, commits_parents_df):
    print('\nGenerating Parent Commit-->Commit Edges Indices, Names and Features ... ')

    cc_edge_features_list = ['project_id', 'parent_timestamp', 'parent_message', 'parent_dmm_unit_complexity', 'parent_is_merge',
       'parent_dmm_unit_interfacing', 'parent_dmm_unit_size']
    cc_edge_indices = np.empty((2,len(commits_parents_df)), dtype=int)
    cc_edge_features = np.empty((len(commits_parents_df),len(cc_edge_features_list)), dtype=object)
    commits_parents_df.parent_hash = commits_parents_df.parent_hash.fillna('')
    commits_parents_df = sort_timestamp(commits_parents_df)
    for i in range(len(commits_parents_df)):
        if commits_parents_df.parent_hash.iloc[i]:
            ''' Getting edge indices '''
            cc_edge_indices[0,i] = np.where(commit_nodes_df.commit_hash == commits_parents_df.commit_hash.iloc[i])[0]
            cc_edge_indices[1,i] = np.where(commit_nodes_df.commit_hash == commits_parents_df.parent_hash.iloc[i])[0]
            ''' Getting edge features ''' 
            cc_edge_features[i,:] = commits_parents_df[cc_edge_features_list].iloc[i]
        else:
            cc_edge_indices[:,i] = sys.maxsize

    cc_edge_features = cc_edge_features[(cc_edge_indices != sys.maxsize).any(axis=0),:]   
    cc_edge_indices = cc_edge_indices[:, (cc_edge_indices != sys.maxsize).any(axis=0)]   
    ''' Getting edge features '''                
    cc_edge_features_df = pd.DataFrame(cc_edge_features, columns=cc_edge_features_list)
    cc_edge_features_df.rename(columns = {'parent_message':'commit_message', 'parent_dmm_unit_complexity':'commit_dmm_unit_complexity', 'parent_is_merge':'commit_is_merge', 'parent_dmm_unit_interfacing':'commit_dmm_unit_interfacing', 'parent_dmm_unit_size':'commit_dmm_unit_size'}, inplace = True)
    # print("\nParentCommit-Commit edge features: \n", cc_edge_features_df.columns)
    # display(cc_edge_features_df)

    return cc_edge_indices, cc_edge_features_df


def gen_cf_edges_hetero(commit_nodes_df, file_nodes_df, commits_files_df):
    print('\nGenerating Commit-->File Edges Indices, Names and Features ... ')    
    # print(cf_adj_df.shape)
    cf_edge_features_list = ['project_id', 'timestamp', 'fileupdate_complexity',
       'fileupdate_nloc', 'fileupdate_added', 'fileupdate_type',
       'fileupdate_token_count', 'fileupdate_removed']#, 'file_name', 'file_type']
    cf_edge_indices = np.empty((2,len(commits_files_df)), dtype=int)
    cf_edge_features = np.empty((len(commits_files_df), len(cf_edge_features_list)), dtype=object)
    commits_files_df.file_hash = commits_files_df.file_hash.fillna('')
    for i in range(len(commits_files_df)):
        if commits_files_df.file_hash.iloc[i]:
            ''' Getting edge indices '''
            cf_edge_indices[0,i] = np.where(commit_nodes_df.commit_hash == commits_files_df.commit_hash.iloc[i])[0]
            cf_edge_indices[1,i] = np.where(file_nodes_df.file_hash == commits_files_df.file_hash.iloc[i])[0]
            ''' Getting edge features ''' 
            cf_edge_features[i,:] = commits_files_df[cf_edge_features_list].iloc[i]   
        else:
            cf_edge_indices[:,i] = sys.maxsize

    cf_edge_features= cf_edge_features[(cf_edge_indices != sys.maxsize).any(axis=0),:]   
    cf_edge_indices= cf_edge_indices[:, (cf_edge_indices != sys.maxsize).any(axis=0)]   

    cf_edge_features_df = pd.DataFrame(cf_edge_features, columns=cf_edge_features_list)
    # print("\nCommit-Files edge features: \n", cf_edge_features_df.columns)
    # display(cf_edge_features_df)

    return cf_edge_indices, cf_edge_features_df


def gen_cm_edges_hetero(commit_nodes_df, method_nodes_df, commits_methods_df):
    print('\nGenerating Commit-->Method Edges Indices, Names and Features ... ')
    # print(cm_adj_df.shape)
    cm_edge_features_list = ['project_id', 'timestamp', 'methodupdate_complexity', 'methodupdate_nloc',
       'methodupdate_token_count', 'methodupdate_length', 'methodupdate_fan_in', 'methodupdate_start_line',
       'methodupdate_general_fan_out', 'methodupdate_end_line', 'methodupdate_fan_out']#, 'method_name', 'file_name']
    cm_edge_indices = np.empty((2,len(commits_methods_df)), dtype=int)
    cm_edge_features = np.empty((len(commits_methods_df), len(cm_edge_features_list)), dtype=object)
    commits_methods_df.method_hash = commits_methods_df.method_hash.fillna('')
    for i in range(len(commits_methods_df)):
        if commits_methods_df.method_hash.iloc[i]:
            ''' Getting edge indices '''
            cm_edge_indices[0,i] = np.where(commit_nodes_df.commit_hash == commits_methods_df.commit_hash.iloc[i])[0]
            cm_edge_indices[1,i] = np.where(method_nodes_df.method_hash == commits_methods_df.method_hash.iloc[i])[0]
            ''' Getting edge features '''
            cm_edge_features[i,:] = commits_methods_df[cm_edge_features_list].iloc[i]   
        else:
            cm_edge_indices[:,i] = sys.maxsize
    cm_edge_features= cm_edge_features[(cm_edge_indices != sys.maxsize).any(axis=0),:]   
    cm_edge_indices= cm_edge_indices[:, (cm_edge_indices != sys.maxsize).any(axis=0)]

    cm_edge_features_df = pd.DataFrame(cm_edge_features, columns=cm_edge_features_list)
    # print("\nCommit-Methods edge features: \n", cm_edge_features_df.columns)
    # display(cm_edge_features_df)

    return cm_edge_indices, cm_edge_features_df


def gen_fm_edges_hetero(file_nodes_df, method_nodes_df, files_graph_df):
    print('\nGenerating File-->Method Edges Indices, Names and Features ... ')    
    fm_edge_features_list = ["project_id", 'method_name', 'file_name']
    fm_edge_indices = np.empty((2,len(files_graph_df)), dtype=int)
    fm_edge_features = np.empty((len(files_graph_df), len(fm_edge_features_list)), dtype=object)
    # fm_edge_features[:,0] = 1
    files_graph_df.method_hash = files_graph_df.method_hash.fillna('')
    for i in range(len(files_graph_df)):
        if files_graph_df.method_hash.iloc[i]:
            ''' Getting edge indices '''
            fm_edge_indices[0,i] = np.where(file_nodes_df.file_hash == files_graph_df.file_hash.iloc[i])[0]
            fm_edge_indices[1,i] = np.where(method_nodes_df.method_hash == files_graph_df.method_hash.iloc[i])[0]  
            ''' Getting edge features '''
            fm_edge_features[i,:] = files_graph_df[fm_edge_features_list].iloc[i]         
        else:
            fm_edge_indices[:,i] = sys.maxsize

    fm_edge_features = fm_edge_features[(fm_edge_indices != sys.maxsize).any(axis=0),:]   
    fm_edge_indices= fm_edge_indices[:, (fm_edge_indices != sys.maxsize).any(axis=0)]
    
    fm_edge_features_df = pd.DataFrame(fm_edge_features, columns=fm_edge_features_list)
    # print("\nFile-Methods edge features: \n", fm_edge_features_df.columns)
    # display(fm_edge_features_df)

    return fm_edge_indices, fm_edge_features_df


def gen_edges_hetero(branch_nodes_df, dev_nodes_df, commit_nodes_df, 
                        file_nodes_df, method_nodes_df, branches_graph_df, devs_graph_df, 
                        commits_parents_df, commits_files_df, commits_methods_df, files_graph_df):

    bc_edge_indices, bc_edge_features_df = gen_bc_edges_hetero(branch_nodes_df, commit_nodes_df, branches_graph_df)
    dc_edge_indices, dc_edge_features_df = gen_dc_edges_hetero(dev_nodes_df, commit_nodes_df, devs_graph_df)
    cc_edge_indices, cc_edge_features_df = gen_cc_edges_hetero(commit_nodes_df, commits_parents_df)
    cf_edge_indices, cf_edge_features_df = gen_cf_edges_hetero(commit_nodes_df, file_nodes_df, commits_files_df)
    cm_edge_indices, cm_edge_features_df = gen_cm_edges_hetero(commit_nodes_df, method_nodes_df, commits_methods_df)
    fm_edge_indices, fm_edge_features_df = gen_fm_edges_hetero(file_nodes_df, method_nodes_df, files_graph_df)

    bc_edge_features_df.to_csv("working/graphdata/bc_edge_features_df.csv")
    dc_edge_features_df.to_csv("working/graphdata/dc_edge_features_df.csv")
    cc_edge_features_df.to_csv("working/graphdata/cc_edge_features_df.csv")
    cf_edge_features_df.to_csv("working/graphdata/cf_edge_features_df.csv")
    cm_edge_features_df.to_csv("working/graphdata/cm_edge_features_df.csv")
    fm_edge_features_df.to_csv("working/graphdata/fm_edge_features_df.csv")      

    edge_indices = np.concatenate((bc_edge_indices, dc_edge_indices, cc_edge_indices, cf_edge_indices, cm_edge_indices, fm_edge_indices), axis=1)
    edge_features_df = pd.concat([bc_edge_features_df, dc_edge_features_df, cc_edge_features_df, cf_edge_features_df, cm_edge_features_df, fm_edge_features_df]) 
    
    return edge_indices, edge_features_df, \
            bc_edge_indices, bc_edge_features_df, \
                dc_edge_indices, dc_edge_features_df, \
                    cc_edge_indices, cc_edge_features_df, \
                        cf_edge_indices, cf_edge_features_df, \
                            cm_edge_indices, cm_edge_features_df, \
                                fm_edge_indices, fm_edge_features_df  


def gen_nodes_edges(branches_commits_df, branches_files_df, branches_methods_df, \
                        devs_commits_df, devs_files_df, devs_methods_df, \
                            commits_parents_df, commits_files_df, commits_methods_df, \
                                files_graph_df, methods_graph_df, hetero=False):

    # devs_graph_df = sort_timestamp(devs_graph_df)
    # commits_parents_df = sort_timestamp(commits_parents_df)
    # commits_files_df = sort_timestamp(commits_files_df)
    if hetero:
        nodes_df, node_names, \
        branch_nodes_df, branch_names, branch_node_type, \
        dev_nodes_df, dev_names, dev_node_type, \
        commit_nodes_df, commit_names, commit_node_type, \
        file_nodes_df, file_names,file_node_type, \
        method_nodes_df, method_names, method_node_type = gen_nodes(branches_commits_df, devs_commits_df, commits_parents_df, commits_files_df, commits_methods_df, files_graph_df, methods_graph_df, hetero=hetero)

        edge_indices, edge_features_df, \
            bc_edge_indices, bc_edge_features_df, \
                dc_edge_indices, dc_edge_features_df, \
                    cc_edge_indices, cc_edge_features_df, \
                        cf_edge_indices, cf_edge_features_df, \
                            cm_edge_indices, cm_edge_features_df, \
                                fm_edge_indices, fm_edge_features_df \
                                = gen_edges_hetero(branch_nodes_df, dev_nodes_df, commit_nodes_df, 
                                                    file_nodes_df, method_nodes_df, branches_commits_df, devs_commits_df, 
                                                    commits_parents_df, commits_files_df, commits_methods_df, files_graph_df)
        print("\n ------------- Done ------------- \n")

        return nodes_df, node_names, \
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
                fm_edge_indices, fm_edge_features_df
    else:
        nodes_df, node_names, node_type = gen_nodes(branches_commits_df, devs_commits_df, \
                                                        commits_parents_df, commits_files_df, commits_methods_df, \
                                                            files_graph_df, methods_graph_df, hetero=hetero)

        edge_indices, edge_features_df = gen_edges(nodes_df, branches_commits_df, branches_files_df, branches_methods_df, \
                                                        devs_commits_df, devs_files_df, devs_methods_df, \
                                                            commits_parents_df, commits_files_df, commits_methods_df, files_graph_df)
        
        print("\n ------------- Done ------------- \n")

        return nodes_df, node_names, node_type, edge_indices, edge_features_df


'''
    Converter for collective edge indices to heterogenous edge indices
                                                                       '''

def cei_to_hei(nodes_df, edge_indices):
    branch_nodes_df = nodes_df[nodes_df.branch_hash.notnull()]
    dev_nodes_df = nodes_df[nodes_df.dev_hash.notnull()]
    commit_nodes_df = nodes_df[nodes_df.commit_hash.notnull()]
    file_nodes_df = nodes_df[nodes_df.file_hash.notnull()]
    method_nodes_df = nodes_df[nodes_df.method_hash.notnull()] 

    n_branches = len(branch_nodes_df)
    n_devs = len(dev_nodes_df)
    n_commits = len(commit_nodes_df)
    n_files = len(file_nodes_df)
    n_methods = len(method_nodes_df)

    bc_edge_indices_hetero = \
        edge_indices[:, (edge_indices[0,:] < n_branches)]
    bc_edge_indices_hetero[1,:] = bc_edge_indices_hetero[1,:] - (n_branches + n_devs)

    dc_edge_indices_hetero = \
        edge_indices[:, (np.array(edge_indices[0,:] >= n_branches) & \
                            np.array(edge_indices[0,:] < (n_branches + n_devs)))]
    dc_edge_indices_hetero[0,:] = dc_edge_indices_hetero[0,:] - n_branches
    dc_edge_indices_hetero[1,:] = dc_edge_indices_hetero[1,:] - (n_branches + n_devs)

    cc_edge_indices_hetero = \
        edge_indices[:, (np.array(edge_indices[0,:] >= (n_branches + n_devs)) & \
                            np.array(edge_indices[0,:] < (n_branches + n_devs + n_commits)) & \
                                np.array(edge_indices[1,:] >= (n_branches + n_devs)) & \
                                    np.array(edge_indices[1,:] < (n_branches + n_devs + n_commits)))]
    cc_edge_indices_hetero[0,:] = cc_edge_indices_hetero[0,:] - (n_branches + n_devs)
    cc_edge_indices_hetero[1,:] = cc_edge_indices_hetero[1,:] - (n_branches + n_devs)

    cf_edge_indices_hetero = \
        edge_indices[:, (np.array(edge_indices[0,:] >= (n_branches + n_devs)) & \
                            np.array(edge_indices[0,:] < (n_branches + n_devs + n_commits)) & \
                                np.array(edge_indices[1,:] >= (n_branches + n_devs + n_commits)) & \
                                    np.array((edge_indices[1,:] < (n_branches + n_devs + n_commits + n_files))))]
    cf_edge_indices_hetero[0,:] = cf_edge_indices_hetero[0,:] - (n_branches + n_devs)
    cf_edge_indices_hetero[1,:] = cf_edge_indices_hetero[1,:] - (n_branches + n_devs + n_commits)

    cm_edge_indices_hetero = \
        edge_indices[:, (np.array(edge_indices[0,:] >= (n_branches + n_devs)) & \
                            np.array(edge_indices[0,:] < (n_branches + n_devs + n_commits)) & \
                                np.array(edge_indices[1,:] >= (n_branches + n_devs + n_commits + n_files)) & \
                                    np.array(edge_indices[1,:] < (n_branches + n_devs + n_commits + n_files + n_methods)))]
    cm_edge_indices_hetero[0,:] = cm_edge_indices_hetero[0,:] - (n_branches + n_devs)
    cm_edge_indices_hetero[1,:] = cm_edge_indices_hetero[1,:] - (n_branches + n_devs + n_commits + n_files)

    fm_edge_indices_hetero = \
        edge_indices[:, (np.array(edge_indices[0,:] >= (n_branches + n_devs + n_commits)) & \
                            np.array(edge_indices[0,:] < (n_branches + n_devs + n_commits + n_files)) & \
                                np.array(edge_indices[1,:] >= (n_branches + n_devs + n_commits + n_files)) & \
                                    np.array(edge_indices[1,:] < (n_branches + n_devs + n_commits + n_files + n_methods)))]
    fm_edge_indices_hetero[0,:] = fm_edge_indices_hetero[0,:] - (n_branches + n_devs + n_commits)
    fm_edge_indices_hetero[1,:] = fm_edge_indices_hetero[1,:] - (n_branches + n_devs + n_commits + n_files)

    return bc_edge_indices_hetero, dc_edge_indices_hetero, cc_edge_indices_hetero, \
            cf_edge_indices_hetero, cm_edge_indices_hetero, fm_edge_indices_hetero


'''________main________'''

# branches_path = "working/graphdata/branches_graph_df.pkl"
# commits_parents_path = "working/graphdata/commits_parents_df.pkl"
# commits_files_path = "working/graphdata/commits_files_df.pkl"
# commits_methods_path = "working/graphdata/commits_methods_df.pkl"
# devs_path = "working/graphdata/devs_graph_df.pkl"
# files_path = "working/graphdata/files_graph_df.pkl"
# methods_path = "working/graphdata/methods_graph_df.pkl"

# branches_graph_df = pd.read_pickle(branches_path)
# devs_graph_df = pd.read_pickle(devs_path)   
# commits_parents_df = pd.read_pickle(commits_parents_path)
# commits_files_df = pd.read_pickle(commits_files_path)   
# commits_methods_df = pd.read_pickle(commits_methods_path)   
# files_graph_df = pd.read_pickle(files_path) 
# methods_graph_df = pd.read_pickle(methods_path) 

    
# nodes_df, edge_indices, edge_names, edge_features_df \
#     = gen_nodes_edges(branches_graph_df, commits_parents_df, commits_files_df, \
                            # commits_methods_df, devs_graph_df, files_graph_df, \
                                # methods_graph_df)

# nodes_df.to_pickle("working/graphdata/nodes_df.pkl")
# np.save("working/graphdata/edge_indices.npy", edge_indices)
# np.save("working/graphdata/edge_names.npy", edge_names)
# edge_features_df.to_pickle("working/graphdata/edge_features_df.pkl")

# bc_edge_indices_hetero, dc_edge_indices_hetero, cc_edge_indices_hetero, \
# cf_edge_indices_hetero, cm_edge_indices_hetero, fm_edge_indices_hetero = cei_to_hei(nodes_df, edge_indices)

# np.save("working/graphdata/bc_edge_indices_hetero.npy", bc_edge_indices_hetero)
# np.save("working/graphdata/dc_edge_indices_hetero.npy", dc_edge_indices_hetero)
# np.save("working/graphdata/cc_edge_indices_hetero.npy", cc_edge_indices_hetero)
# np.save("working/graphdata/cf_edge_indices_hetero.npy", cf_edge_indices_hetero)
# np.save("working/graphdata/cm_edge_indices_hetero.npy", cm_edge_indices_hetero)
# np.save("working/graphdata/fm_edge_indices_hetero.npy", fm_edge_indices_hetero)



