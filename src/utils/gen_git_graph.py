import os
from graphrepo.miners import MineManager
import pandas as pd
import numpy as np
from IPython.display import display
import subprocess
from github import Github
import warnings
warnings.simplefilter(action='ignore')

def gen_branch_graph(miner):
    """ The branches miner - get_all """
    print('\nGenerating Branches Graph ... ')
    branches = miner.branch_miner.get_all()
    branches_df = pd.DataFrame(branches)
    branches_df = branches_df.add_prefix("branch_")
    branches_df.rename(columns = {'branch_project_id':'project_id'}, inplace = True)
    # print("\nBranches_df\n", branches_df.columns)
    # display(branches_df)    

    """ The branches miner - get_branch_commits """
    branches_commits_df = pd.DataFrame()
    branches_files_df = pd.DataFrame()
    branches_methods_df = pd.DataFrame()
    for h in list(branches_df.branch_hash):
        branch_commits = miner.branch_miner.get_commits(h)
        branch_commits_df = pd.DataFrame(branch_commits)
        branch_commits_df = branch_commits_df.add_prefix("commit_")
        branch_commits_df.rename(columns = {'commit_project_id':'project_id', 'commit_timestamp':'timestamp'}, inplace = True)        

        if branch_commits_df.empty:
            branch_commits_df = branches_df[branches_df.branch_hash == h]
        else:
            branch_commits_df = pd.merge(branches_df[branches_df.branch_hash == h], branch_commits_df, on=["project_id"])
        branches_commits_df = pd.concat([branches_commits_df, branch_commits_df])
        # display(branches_commits_df)

        # branch_files = miner.branch_miner.get_files(h)
        # branch_files_df = pd.DataFrame(branch_files)
        # branch_files_df = branch_files_df.add_prefix("file_")
        # branch_files_df.rename(columns = {'file_project_id':'project_id'}, inplace = True)        

        # if branch_files_df.empty:
        #     branch_files_df = branches_df[branches_df.branch_hash == h]
        # else:
        #     branch_files_df = pd.merge(branches_df[branches_df.branch_hash == h], branch_files_df, on=["project_id"])
        # branches_files_df = pd.concat([branches_files_df, branch_files_df])

        # branch_methods = miner.branch_miner.get_methods(h)
        # branch_methods_df = pd.DataFrame(branch_methods)
        # branch_methods_df = branch_methods_df.add_prefix("method_")
        # branch_methods_df.rename(columns = {'method_project_id':'project_id'}, inplace = True)        

        # if branch_methods_df.empty:
        #     branch_methods_df = branches_df[branches_df.branch_hash == h]
        # else:
        #     branch_methods_df = pd.merge(branches_df[branches_df.branch_hash == h], branch_methods_df, on=["project_id"])
        # branches_methods_df = pd.concat([branches_methods_df, branch_methods_df])

    # print("\nBranch_commits_df\n", branches_commits_df.columns)
    # display(branches_commits_df)    
    branches_commits_df.branch_hash = branches_commits_df.project_id  + branches_commits_df.branch_hash
    branches_commits_df.commit_hash = branches_commits_df.project_id  + branches_commits_df.commit_hash
    # branches_files_df.branch_hash = branches_files_df.project_id  + branches_files_df.branch_hash
    # branches_files_df.file_hash = branches_files_df.project_id  + branches_files_df.file_hash
    # branches_methods_df.branch_hash = branches_methods_df.project_id  + branches_methods_df.branch_hash
    # branches_methods_df.method_hash = branches_methods_df.project_id  + branches_methods_df.method_hash

    return branches_commits_df, branches_files_df, branches_methods_df


def gen_dev_graph(miner, url):
    """ The devs miner - get_all """
    print('\nGenerating Developers Graph ... ', url)
    devs = miner.dev_miner.get_all()
    devs_df = pd.DataFrame(devs)

    #g = Github("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")
    g = Github("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")
    reponame = "/".join(url.split("/")[-2:])
    repo = g.get_repo(reponame)
    git_devs_r = repo.get_contributors()
    np.save("working/git_devs.npy", np.array(git_devs_r, dtype=object))
    git_devs = np.load("working/git_devs.npy", allow_pickle=True)
    devs_df["login"] = ""
    devs_df["location"] = ""
    devs_df["public_repos"] = 0
    devs_df["public_gists"] = 0
    devs_df["created_at"] = ""
    for dev in git_devs.item():
        name = dev.name
        email = dev.email
        if name != None or email != None:
            devs_df["login"][(devs_df.name == name) | (devs_df.email == email)] = dev.login 
            devs_df["location"][(devs_df.name == name) | (devs_df.email == email)] = dev.location 
            devs_df["public_repos"][(devs_df.name == name) | (devs_df.email == email)] = dev.public_repos 
            devs_df["public_gists"][(devs_df.name == name) | (devs_df.email == email)] = dev.public_gists 
            devs_df["created_at"][(devs_df.name == name) | (devs_df.email == email)] = str(dev.created_at) 
    devs_df = devs_df.add_prefix("dev_")
    # print("\nDevs_df\n", devs_df.columns)
    # display(devs_df)    

    """ The devs miner - get_dev_commits """
    devs_commits_df = pd.DataFrame()
    devs_files_df = pd.DataFrame()
    devs_methods_df = pd.DataFrame()
    for h in list(devs_df.dev_hash):
        dev_commits = miner.dev_miner.get_commits(h)
        dev_commits_df = pd.DataFrame(dev_commits)
        dev_commits_df = dev_commits_df.add_prefix("commit_")
        dev_commits_df.rename(columns = {'commit_project_id':'project_id', 'commit_timestamp':'timestamp'}, inplace = True)        

        if dev_commits_df.empty:
            dev_commits_df = devs_df[devs_df.dev_hash == h]
        else:
            dev_commits_df = dev_commits_df.join(devs_df[devs_df.dev_hash == h])
            dev_commits_df.loc[:,list(devs_df.columns)] = devs_df[devs_df.dev_hash == h].iloc[0].tolist()
        devs_commits_df = pd.concat([devs_commits_df, dev_commits_df])
        # display(devs_commits_df)

        # dev_files = miner.dev_miner.get_files(h)
        # dev_files_df = pd.DataFrame(dev_files)
        # dev_files_df = dev_files_df.add_prefix("file_")
        # dev_files_df.rename(columns = {'file_project_id':'project_id'}, inplace = True)        

        # if dev_files_df.empty:
        #     dev_files_df = devs_df[devs_df.dev_hash == h]
        # else:
        #     dev_files_df = dev_files_df.join(devs_df[devs_df.dev_hash == h])
        #     dev_files_df.loc[:,list(devs_df.columns)] = devs_df[devs_df.dev_hash == h].iloc[0].tolist()
        # devs_files_df = pd.concat([devs_files_df, dev_files_df])

        # dev_methods = miner.dev_miner.get_methods(h)
        # dev_methods_df = pd.DataFrame(dev_methods)
        # dev_methods_df = dev_methods_df.add_prefix("method_")
        # dev_methods_df.rename(columns = {'method_project_id':'project_id'}, inplace = True)        

        # if dev_methods_df.empty:
        #     dev_methods_df = devs_df[devs_df.dev_hash == h]
        # else:
        #     dev_methods_df = dev_methods_df.join(devs_df[devs_df.dev_hash == h])
        #     dev_methods_df.loc[:,list(devs_df.columns)] = devs_df[devs_df.dev_hash == h].iloc[0].tolist()
        # devs_methods_df = pd.concat([devs_methods_df, dev_methods_df])

    # print("\nDev_commits_df\n", devs_commits_df.columns)
    # display(devs_commits_df)    
    devs_commits_df.dev_hash = devs_commits_df.project_id  + devs_commits_df.dev_hash
    devs_commits_df.commit_hash = devs_commits_df.project_id  + devs_commits_df.commit_hash
    # devs_files_df.dev_hash = devs_files_df.project_id  + devs_files_df.dev_hash
    # devs_files_df.file_hash = devs_files_df.project_id  + devs_files_df.file_hash
    # devs_methods_df.dev_hash = devs_methods_df.project_id  + devs_methods_df.dev_hash
    # devs_methods_df.method_hash = devs_methods_df.project_id  + devs_methods_df.method_hash

    devs_commits_df = devs_commits_df.reset_index().drop(columns="index")

    return devs_commits_df, devs_files_df, devs_methods_df


def gen_commit_graph(miner):
    """ The commits miner - get_all """
    print('\nGenerating Commits Graph ... ')
    commits = miner.commit_miner.get_all()
    commits_df = pd.DataFrame(commits)
    commits_df = commits_df.add_prefix("commit_")
    commits_df.rename(columns = {'commit_timestamp':'timestamp', 'commit_project_id':'project_id'}, inplace = True)
    # print("\nCommits_df\n", commits_df.columns)
    # display(commits_df)
    
    commits_parents_df = pd.DataFrame()
    commits_files_df = pd.DataFrame()    
    commits_methods_df = pd.DataFrame()
    for h in list(commits_df.commit_hash):
        ''' The commits miner - get_parents, get_commit_files, get_commit_file_updates '''
        commit_parents = miner.commit_miner.get_parents(h)
        commit_parents_df = pd.DataFrame(commit_parents)
        commit_parents_df = commit_parents_df.add_prefix("parent_")
        commit_parents_df.rename(columns = {'parent_project_id':'project_id'}, inplace = True)
        if commit_parents_df.empty:
            commit_df = commits_df[commits_df.commit_hash == h]
        else:
            commit_df = pd.merge(commits_df[commits_df.commit_hash == h], commit_parents_df, on=["project_id"])
        commits_parents_df = pd.concat([commits_parents_df, commit_df])
        # display(commits_parents_df)

        commit_files = miner.commit_miner.get_commit_files(h)
        commit_files_df = pd.DataFrame(commit_files)
        commit_files_df = commit_files_df.add_prefix("file_")
        commit_files_df.rename(columns = {'file_project_id':'project_id'}, inplace = True)
        # display(commit_files_df)

        commit_file_updates = miner.commit_miner.get_commit_file_updates(h)
        commit_file_updates_df = pd.DataFrame(commit_file_updates)
        commit_file_updates_df = commit_file_updates_df.add_prefix("fileupdate_")
        commit_file_updates_df.rename(columns = {'fileupdate_timestamp':'timestamp'}, inplace = True)
        # if not commit_file_updates_df.empty:
        #     commit_file_updates_df["file_name"] = np.where(commit_file_updates_df.fileupdate_old_path == "", \
        #                                             commit_file_updates_df.fileupdate_path.str.split("/").str[-1], \
        #                                             commit_file_updates_df.fileupdate_old_path.str.split("/").str[-1])
        # display(commit_file_updates_df)

        if commit_files_df.empty and commit_file_updates_df.empty:
            commit_df = commits_df[commits_df.commit_hash == h]
        else:
            # commit_files_df = pd.merge(commit_files_df, commit_file_updates_df, on="file_name")
            commit_files_df = commit_files_df.join(commit_file_updates_df)
            commit_df = pd.merge(commits_df[commits_df.commit_hash == h], commit_files_df, on=["timestamp", "project_id"])
        commits_files_df = pd.concat([commits_files_df, commit_df])
    
        # The commits miner - get_commit_methods, get_commit_method_updates
        commit_methods = miner.commit_miner.get_commit_methods(h)
        commit_methods_df = pd.DataFrame(commit_methods)
        commit_methods_df = commit_methods_df.add_prefix("method_")
        commit_methods_df.rename(columns = {'method_project_id':'project_id', 'method_file_name':'file_name'}, inplace = True)
        # if not commit_methods_df.empty:
        #     display(commit_methods_df[['method_hash','method_name']].to_markdown())

        commit_method_updates = miner.commit_miner.get_commit_method_updates(h)
        commit_method_updates_df = pd.DataFrame(commit_method_updates)
        commit_method_updates_df = commit_method_updates_df.add_prefix("methodupdate_")
        commit_method_updates_df.rename(columns = {'methodupdate_timestamp':'timestamp', 'method_file_name':'file_name'}, inplace = True)
        # if not commit_method_updates_df.empty:
        #     commit_method_updates_df["method_name"] = commit_method_updates_df.method_name.str.split("(").str[0]
            #   display(commit_method_updates_df[['timestamp', 'methodupdate_nloc']].to_markdown())

        if commit_methods_df.empty and commit_method_updates_df.empty:
            commit_df = commits_df[commits_df.commit_hash == h]
        else:
            # commit_methods_df = pd.merge(commit_methods_df, commit_method_updates_df, on="method_name")
            commit_methods_df = commit_methods_df.join(commit_method_updates_df)
            commit_df = pd.merge(commits_df[commits_df.commit_hash == h], commit_methods_df, on=["timestamp", "project_id"])
        commits_methods_df = pd.concat([commits_methods_df, commit_df])
           
    commits_parents_df.commit_hash = commits_parents_df.project_id  + commits_parents_df.commit_hash
    commits_parents_df.parent_hash = commits_parents_df.project_id  + commits_parents_df.parent_hash
    commits_files_df.commit_hash = commits_files_df.project_id  + commits_files_df.commit_hash
    commits_files_df.file_hash = commits_files_df.project_id  + commits_files_df.file_hash
    commits_methods_df.commit_hash = commits_methods_df.project_id  + commits_methods_df.commit_hash
    commits_methods_df.method_hash = commits_methods_df.project_id  + commits_methods_df.method_hash

    return commits_parents_df, commits_files_df, commits_methods_df


def gen_file_graph(miner):
    """ The files miner - get_all """
    print('\nGenerating Files Graph ... ')
    files = miner.file_miner.get_all()
    files_df = pd.DataFrame(files)
    files_df = files_df.add_prefix("file_")
    files_df.rename(columns = {'file_project_id':'project_id'}, inplace = True)
    # print("\nFiles_df\n", files_df.columns)
    # display(files_df)    

    """ The files miner - get_past_methods, get_current_methods """ 
    files_updates_df = pd.DataFrame()
    for h in list(files_df.file_hash):
        file_past_methods = miner.file_miner.get_past_methods(h)
        file_past_methods_df = pd.DataFrame(file_past_methods)
         
        file_cur_methods = miner.file_miner.get_current_methods(h)
        file_cur_methods_df = pd.DataFrame(file_cur_methods)
        if file_cur_methods_df.empty and file_past_methods_df.empty:
            file_updates_df = files_df[files_df.file_hash == h]       
        else:
            file_methods_df = file_cur_methods_df.append(file_past_methods_df)
            file_methods_df = file_methods_df.add_prefix("method_")
            file_methods_df.method_name = file_methods_df.method_file_name + "_" + file_methods_df.method_name
            file_methods_df.rename(columns = {'method_file_name':'file_name', 'method_project_id':'project_id'}, inplace = True)        
            file_updates_df = pd.merge(files_df[files_df.file_hash == h], file_methods_df, on=["file_name","project_id"])
        
        files_updates_df = pd.concat([files_updates_df, file_updates_df])
        # if not file_updates_df.empty:
            # display(file_updates_df.to_markdown())
    # print("\nFiles_updates_df\n", files_updates_df.columns)
    # display(files_updates_df)    
    files_updates_df.file_hash = files_updates_df.project_id  + files_updates_df.file_hash
    files_updates_df.method_hash = files_updates_df.project_id  + files_updates_df.method_hash

    return files_updates_df


def gen_method_graph(miner):
    """ The methods miner - get_all """
    print('\nGenerating Methods Graph ... ')
    methods = miner.method_miner.get_all()
    methods_df = pd.DataFrame(methods)
    methods_df = methods_df.add_prefix("method_")
    methods_df.method_name = methods_df.method_file_name + "_" + methods_df.method_name
    methods_df.rename(columns = {'method_file_name':'file_name', 'method_project_id':'project_id'}, inplace = True)
    # print("\nMethods_df\n", methods_df.columns)
    # display(methods_df)    

    # """ The methods miner - get_change_history """
    # methods_updates_df = pd.DataFrame()
    # if not methods_df.empty:
    #     for h in list(methods_df.method_hash):
    #         method_updates = miner.method_miner.get_change_history(h)
    #         method_updates_df = pd.DataFrame(method_updates)
    #         method_updates_df = method_updates_df.add_prefix("methodupdate_")
    #         method_updates_df.rename(columns = {'method_file_name':'file_name', 'methodupdate_timestamp':'timestamp', 'methodupdate_long_name':'method_name'}, inplace = True)
    #         if method_updates_df.empty:
    #             method_updates_df = methods_df[methods_df.method_hash == h]
    #         else:
    #             method_updates_df["method_name"] = methods_df[methods_df.method_hash == h].method_name.iloc[0]  
    #             method_updates_df = pd.merge(methods_df[methods_df.method_hash == h], method_updates_df, on="method_name")
    #         methods_updates_df = pd.concat([methods_updates_df, method_updates_df])
    # print("\nMethods_updates_df\n", methods_updates_df.columns)
    # display(methods_updates_df)    
    # methods_updates_df.method_hash = methods_updates_df.project_id  + methods_updates_df.method_hash
    methods_df.method_hash = methods_df.project_id  + methods_df.method_hash

    return methods_df

def gen_git_graph(config_path, project_id, url):    
    subprocess.Popen("python3 -m examples.index_all --config="+config_path, shell=True).wait()
    # os.system("python3 -m examples.mine_all --config="+config_path)
    
    """ initialize mine manager """
    miner = MineManager(config_path=config_path)
    
    branches_commits_df, branches_files_df, branches_methods_df = gen_branch_graph(miner)
    devs_commits_df, devs_files_df, devs_methods_df = gen_dev_graph(miner, url)
    commits_parents_df, commits_files_df, commits_methods_df = gen_commit_graph(miner)
    files_graph_df = gen_file_graph(miner)
    methods_graph_df = gen_method_graph(miner)
    
    # if combined:
    #     return branches_graph_df, commits_parents_df, \
    #             commits_files_df, commits_methods_df, \
    #                 devs_graph_df, files_graph_df, \
    #                     methods_graph_df
    # else:
    return branches_commits_df[branches_commits_df.project_id == project_id], \
            branches_files_df, branches_methods_df, \
                devs_commits_df[devs_commits_df.project_id == project_id], \
                devs_files_df, devs_methods_df, \
                    commits_parents_df[commits_parents_df.project_id == project_id], \
                    commits_files_df[commits_files_df.project_id == project_id], \
                    commits_methods_df[commits_methods_df.project_id == project_id], \
                        files_graph_df[files_graph_df.project_id == project_id], \
                            methods_graph_df[methods_graph_df.project_id == project_id]

# config_path = "examples/configs/pydriller.yml"

# branches_graph_df, commits_parents_df, commits_files_df, commits_methods_df, devs_graph_df, files_graph_df, methods_graph_df = gen_git_graph(config_path)
# branches_graph_df.to_pickle("working/graphdata/branches_graph_df.pkl")
# commits_parents_df.to_pickle("working/graphdata/commits_parents_df.pkl")
# commits_files_df.to_pickle("working/graphdata/commits_files_df.pkl")
# commits_methods_df.to_pickle("working/graphdata/commits_methods_df.pkl")
# devs_graph_df.to_pickle("working/graphdata/devs_graph_df.pkl")
# files_graph_df.to_pickle("working/graphdata/files_graph_df.pkl")
# methods_graph_df.to_pickle("working/graphdata/methods_graph_df.pkl")
