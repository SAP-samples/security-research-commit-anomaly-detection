import yaml, pandas as pd
import numpy as np
import subprocess, time, os, sys, datetime
from graphrepo.miners import MineManager
from IPython.display import display
file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(file_dir, '..', ))
from src.utils.gen_git_graph import gen_commit_graph, gen_dev_graph
from pydriller import RepositoryMining
import requests, pickle
from github import Github
from iteration_utilities import flatten
from skopt import gp_minimize
from skopt.space import Real, Categorical
from functools import partial

import warnings
warnings.simplefilter(action='ignore')



def gen_commit_reports(ocprops_df, dcf_df, dev_commit_df, author_trust_factors_df, committer_trust_factors_df, rules, url, out_f, rules_list):
    rules["DMR Threshold"] = rules_list[0]
    rules["OCP Threshold"] = rules_list[1]
    rules["CT Min. Time as Contributor Threshold"] = rules_list[2]
    rules["CTR Threshold"] = rules_list[3]
    rules["CT Few commits Threshold"] = rules_list[4]
    rules["CT Same Day commit Threshold"] = rules_list[5]
    rules["Contrib. Trust Un-merged PR Threshold"] = rules_list[6]
    rules["Ownership Un-touched Files Threshold"] = rules_list[7]
    rules["Ownership Un-owned Files Threshold"] = rules_list[8]

    ocprops_df = ocprops_df.reset_index().drop(columns="index")
    dcf_df = dcf_df.reset_index().drop(columns="index")
    dev_commit_df = dev_commit_df.reset_index().drop(columns="index")
    author_trust_factors_df = author_trust_factors_df.reset_index().drop(columns="index")
    committer_trust_factors_df = committer_trust_factors_df.reset_index().drop(columns="index")

    ocprops_df.commit_date = pd.to_datetime(ocprops_df.commit_date, utc=True)
    dcf_df.commit_date = pd.to_datetime(dcf_df.commit_date, utc=True)
    dev_commit_df.commit_date = pd.to_datetime(dev_commit_df.commit_date, utc=True)
    repo_mean_change_prop = ocprops_df[["LOC_added", "LOC_removed", "NOF_added",\
                                        "NOF_removed", "NOF_modified",\
                                        "NOF_renamed", "NOF_uniquetypes"]].mean().to_dict()
    repo_std_change_prop = ocprops_df[["LOC_added", "LOC_removed", "NOF_added",\
                                        "NOF_removed", "NOF_modified",\
                                        "NOF_renamed", "NOF_uniquetypes"]].std().to_dict()
    
    dcf_df["no_of_commits_to_file"] = 0
    dcf_df["no_of_commits_to_file_by_author"] = 0
    dcf_df["no_of_commits_to_file_by_committer"] = 0
    dcf_df["author_file_owner"] = False
    dcf_df["author_ownership_of_file"] = 0
    dcf_df["author_contribution_of_file"] = 0
    dcf_df["author_file_majority_contributor"] = False
    dcf_df["committer_file_owner"] = False
    dcf_df["committer_ownership_of_file"] = 0
    dcf_df["committer_contribution_of_file"] = 0
    dcf_df["committer_file_majority_contributor"] = False

    file_names = dcf_df.file_name.unique()
    for ex_name in rules["File History Excluded Files"]:
        for f_name in file_names:
            if ex_name in f_name:
                file_names = file_names[file_names != f_name]
                 
    for file in dcf_df.file_name.unique():
        # display(dcf_df[["no_of_commits_to_file", "no_of_commits_to_file_by_author", "file_owner", "author_ownership_of_file", "author_contribution_of_file"]][dcf_df.file_name == file])
        dcf_df["no_of_commits_to_file"][dcf_df.file_name == file] = len(dcf_df[dcf_df.file_name == file].commit)

        for author in dcf_df.author_login.unique():
            dcf_df["no_of_commits_to_file_by_author"][(dcf_df.file_name == file) & (dcf_df.author_login == author)] \
                = len(dcf_df[(dcf_df.file_name == file) & (dcf_df.author_login == author)].commit)            
        
        dcf_df["author_ownership_of_file"][dcf_df.file_name == file] \
                = dcf_df["no_of_commits_to_file_by_author"][dcf_df.file_name == file] \
                / dcf_df["no_of_commits_to_file"][dcf_df.file_name == file]
        
        if (dcf_df["author_ownership_of_file"][dcf_df.file_name == file] >= 0.5).any():
            dcf_df["author_file_owner"][(dcf_df.file_name == file) & (dcf_df["author_ownership_of_file"][dcf_df.file_name == file] >= 0.5)] = True

        dcf_df["author_contribution_of_file"][dcf_df.file_name == file] \
            = dcf_df["author_ownership_of_file"][dcf_df.file_name == file] \
                / dcf_df["author_ownership_of_file"][(dcf_df.file_name == file) & (dcf_df.author_file_owner == True)].max() 
        
        if (dcf_df["author_contribution_of_file"][dcf_df.file_name == file] >= 0.5).any():
            dcf_df["author_file_majority_contributor"][(dcf_df.file_name == file) & (dcf_df["author_contribution_of_file"][dcf_df.file_name == file] >= 0.5) & (dcf_df["author_file_owner"][dcf_df.file_name == file] == False)] = True

        for committer in dcf_df.committer_login.unique():
            dcf_df["no_of_commits_to_file_by_committer"][(dcf_df.file_name == file) & (dcf_df.committer_login == committer)] \
                = len(dcf_df[(dcf_df.file_name == file) & (dcf_df.committer_login == committer)].commit)            
        
        dcf_df["committer_ownership_of_file"][dcf_df.file_name == file] \
                = dcf_df["no_of_commits_to_file_by_committer"][dcf_df.file_name == file] \
                / dcf_df["no_of_commits_to_file"][dcf_df.file_name == file]
        
        if (dcf_df["committer_ownership_of_file"][dcf_df.file_name == file] >= 0.5).any():
            dcf_df["committer_file_owner"][(dcf_df.file_name == file) & (dcf_df["committer_ownership_of_file"][dcf_df.file_name == file] >= 0.5)] = True

        dcf_df["committer_contribution_of_file"][dcf_df.file_name == file] \
            = dcf_df["committer_ownership_of_file"][dcf_df.file_name == file] \
                / dcf_df["committer_ownership_of_file"][(dcf_df.file_name == file) & (dcf_df.committer_file_owner == True)].max() 
        
        if (dcf_df["committer_contribution_of_file"][dcf_df.file_name == file] >= 0.5).any():
            dcf_df["committer_file_majority_contributor"][(dcf_df.file_name == file) & (dcf_df["committer_contribution_of_file"][dcf_df.file_name == file] >= 0.5) & (dcf_df["committer_file_owner"][dcf_df.file_name == file] == False)] = True

        # display(dcf_df[["no_of_commits_to_file", "no_of_commits_to_file_by_committer", "file_owner", "committer_ownership_of_file", "committer_contribution_of_file"]][dcf_df.file_name == file])
        

    commits_list = []
    n_flagged_commits = 0
    dev_commit_df = dev_commit_df.sort_values(by=['commit_date']) 
    for h, c_hash in enumerate(dev_commit_df.commit):

        commit_dict = {}
        commit_dict["commit_hash"] = c_hash
        if not dev_commit_df[dev_commit_df.commit == c_hash].author_login.empty:
            commit_dict["author login"] = dev_commit_df[dev_commit_df.commit == c_hash].author_login.item()
        else:
            commit_dict["author login"] = "" 
        if not dev_commit_df[dev_commit_df.commit == c_hash].committer_login.empty:
            commit_dict["committer login"] = dev_commit_df[dev_commit_df.commit == c_hash].committer_login.item()
        else:
            commit_dict["committer login"] = ""           
        # try:
        #     commit_dict["author name"] = dev_commit_df[dev_commit_df.commit == c_hash].author.item()
        # except:
        #     commit_dict["author name"] = ""
        try:
            commit_dict["authored on/committed on"] = dev_commit_df[dev_commit_df.commit == c_hash].commit_date.item()
        except:
            commit_dict["authored on/committed on"] = ""    
        commit_dict["commit_message"] = dcf_df[dcf_df.commit == c_hash].commit_message.values
        commit_dict["NOF_modified"] = ocprops_df[ocprops_df.commit == c_hash].NOF_modified.values

        
        ''' Generating Outlier Change Properties Rules Values '''
        
        author_mean_change_prop = ocprops_df[ocprops_df.commit == c_hash]\
                                    [["LOC_added", "LOC_removed", "NOF_added",\
                                        "NOF_removed", "NOF_modified",\
                                        "NOF_renamed", "NOF_uniquetypes"]].mean().to_dict()
        author_std_change_prop = ocprops_df[ocprops_df.commit == c_hash]\
                                    [["LOC_added", "LOC_removed", "NOF_added",\
                                        "NOF_removed", "NOF_modified",\
                                        "NOF_renamed", "NOF_uniquetypes"]].std().to_dict()
        # if h == 0:
        commit_mean_change_prop = ocprops_df[ocprops_df.commit == c_hash]\
                                        [["LOC_added", "LOC_removed", "NOF_added",\
                                        "NOF_removed", "NOF_modified",\
                                        "NOF_renamed", "NOF_uniquetypes"]].mean().to_dict()
        # else:
        #     c_hash_parent = dev_commit_df.commit[h-1]
        #     commit_mean_change_prop = (ocprops_df[ocprops_df.commit == c_hash]\
        #                                 [["LOC_added", "LOC_removed", "NOF_added",\
        #                                 "NOF_removed", "NOF_modified",\
        #                                 "NOF_renamed", "NOF_uniquetypes"]] \
        #                                 + ocprops_df[ocprops_df.commit == c_hash_parent]\
        #                                 [["LOC_added", "LOC_removed", "NOF_added",\
        #                                 "NOF_removed", "NOF_modified",\
        #                                 "NOF_renamed", "NOF_uniquetypes"]]).mean().to_dict()
        for (kr_m, vr_m), (kr_std, vr_std), (ka_m, va_m), (ka_std, va_std), (kc, vc) \
            in zip(repo_mean_change_prop.items(), repo_std_change_prop.items(), \
                    author_mean_change_prop.items(), author_std_change_prop.items(), \
                        commit_mean_change_prop.items()):
            commit_dict[kc + "_mean"] = vc            
            if np.abs(vc) >= np.abs(va_m + 2*va_std) or np.abs(vc) >= np.abs(vr_m + 2*vr_std): 
                commit_dict[kc] = True                
            else:
                commit_dict[kc] = False
        OCP_value = 0
        OCP_value += commit_dict["LOC_added"]
        OCP_value += commit_dict["LOC_removed"]
        OCP_value += commit_dict["NOF_added"]
        OCP_value += commit_dict["NOF_removed"]
        OCP_value += commit_dict["NOF_modified"]
        OCP_value += commit_dict["NOF_renamed"]
        OCP_value += commit_dict["NOF_uniquetypes"]
        if (OCP_value / 7) >= rules["OCP Threshold"]:
            commit_dict["R5 OCP Threshold Breach"] = True
        else:
            commit_dict["R5 OCP Threshold Breach"] = False
        
        
        ''' Generating Contrib. Trust Rules Values '''
        
        commit_dict["T1 author does not exist Flag"] = False
        commit_dict["T2 author account age Flag"] = False
        commit_dict["T3 author fewer commits Flag"] = False
        commit_dict["T4 author first commit Flag"] = False
        commit_dict["T5 author same day commits Flag"] = False
        commit_dict["T6 author time since first commit Flag"] = False
        commit_dict["T7 author rejected PRs Flag"] = False

        commit_dict["T1 author does not exist Flag"] = not author_trust_factors_df[author_trust_factors_df.author_login == commit_dict["author login"]].author_exists.item()
        
        commit_dict["T2 author account age"] = author_trust_factors_df[author_trust_factors_df.author_login == commit_dict["author login"]].author_account_age.item()
        if  commit_dict["T2 author account age"] <= rules["CT Min. Time as Contributor Threshold"]:
            commit_dict["T2 author account age Flag"] = True
        
        commit_dict["T3 author commits"] = author_trust_factors_df[author_trust_factors_df.author_login == commit_dict["author login"]].author_commits.item()
        if (commit_dict["T3 author commits"] / len(ocprops_df.commit)) <= rules["CT Few commits Threshold"]:
            commit_dict["T3 author fewer commits Flag"] = True

        commit_dict["T4 author first commit Flag"] = dev_commit_df[dev_commit_df.commit == c_hash].author_first_commit.item()

        commit_dict["T5 author same day commits"] = dev_commit_df[dev_commit_df.commit == c_hash].author_no_of_same_day_commits.item() 
        if (commit_dict["T5 author same day commits"] / commit_dict["T3 author commits"]) >= rules["CT Same Day commit Threshold"]:
            commit_dict["T5 author same day commits Flag"] = True
        
        commit_dict["T6 author time since first commit"] = author_trust_factors_df[author_trust_factors_df.author_login == commit_dict["author login"]].author_time_since_first_commit.item()
        if commit_dict["T6 author time since first commit"] <= rules["CT Min. Time as Contributor Threshold"]:
            commit_dict["T6 author time since first commit Flag"] = True

        commit_dict["T7 author rejected PRs"] = dcf_df[dcf_df.author_login == commit_dict["author login"]].author_n_rejected_PRs.iloc[0] / (dcf_df[dcf_df.author_login == commit_dict["author login"]].author_n_PRs.iloc[0] + 1e-12)
        if commit_dict["T7 author rejected PRs"] >= rules["Contrib. Trust Un-merged PR Threshold"]:
            commit_dict["T7 author rejected PRs Flag"] = True

        
        commit_dict["T1 committer does not exist Flag"] = False
        commit_dict["T2 committer account age Flag"] = False
        commit_dict["T3 committer fewer commits Flag"] = False
        commit_dict["T4 committer first commit Flag"] = False
        commit_dict["T5 committer same day commits Flag"] = False
        commit_dict["T6 committer time since first commit Flag"] = False
        commit_dict["T7 committer rejected PRs Flag"] = False

        commit_dict["T1 committer does not exist Flag"] = not committer_trust_factors_df[committer_trust_factors_df.committer_login == commit_dict["committer login"]].committer_exists.item()
        
        commit_dict["T2 committer account age"] = committer_trust_factors_df[committer_trust_factors_df.committer_login == commit_dict["committer login"]].committer_account_age.item()
        if  commit_dict["T2 committer account age"] <= rules["CT Min. Time as Contributor Threshold"]:
            commit_dict["T2 committer account age Flag"] = True
        
        commit_dict["T3 committer commits"] = committer_trust_factors_df[committer_trust_factors_df.committer_login == commit_dict["committer login"]].committer_commits.item()
        if (commit_dict["T3 committer commits"] / len(ocprops_df.commit)) <= rules["CT Few commits Threshold"]:
            commit_dict["T3 committer fewer commits Flag"] = True

        commit_dict["T4 committer first commit Flag"] = dev_commit_df[dev_commit_df.commit == c_hash].committer_first_commit.item()

        commit_dict["T5 committer same day commits"] = dev_commit_df[dev_commit_df.commit == c_hash].committer_no_of_same_day_commits.item() 
        if (commit_dict["T5 committer same day commits"] / commit_dict["T3 committer commits"]) >= rules["CT Same Day commit Threshold"]:
            commit_dict["T5 committer same day commits Flag"] = True
        
        commit_dict["T6 committer time since first commit"] = committer_trust_factors_df[committer_trust_factors_df.committer_login == commit_dict["committer login"]].committer_time_since_first_commit.item()
        if commit_dict["T6 committer time since first commit"] <= rules["CT Min. Time as Contributor Threshold"]:
            commit_dict["T6 committer time since first commit Flag"] = True

        commit_dict["T7 committer rejected PRs"] = dcf_df[dcf_df.committer_login == commit_dict["committer login"]].committer_n_rejected_PRs.iloc[0] / (dcf_df[dcf_df.committer_login == commit_dict["committer login"]].committer_n_PRs.iloc[0] + 1e-12)
        if commit_dict["T7 committer rejected PRs"] >= rules["Contrib. Trust Un-merged PR Threshold"]:
            commit_dict["T7 committer rejected PRs Flag"] = True
        
        # CT_value = 0
        # CT_value += (commit_dict["T1 author does not exist Flag"] | commit_dict["T1 committer does not exist Flag"])
        # CT_value += (commit_dict["T2 author account age Flag"] | commit_dict["T2 committer account age Flag"])
        # CT_value += (commit_dict["T3 author fewer commits Flag"] | commit_dict["T3 committer fewer commits Flag"])
        # CT_value += (commit_dict["T4 author first commit Flag"] | commit_dict["T4 committer first commit Flag"])
        # CT_value += (commit_dict["T5 author same day commits Flag"] | commit_dict["T5 committer same day commits Flag"])
        # CT_value += (commit_dict["T6 author time since first commit Flag"] | commit_dict["T6 committer time since first commit Flag"])
        # CT_value += (commit_dict["T7 author rejected PRs Flag"] | commit_dict["T7 committer rejected PRs Flag"])

        CT_author_value = 0
        CT_author_value += commit_dict["T1 author does not exist Flag"]
        CT_author_value += commit_dict["T2 author account age Flag"]
        CT_author_value += commit_dict["T3 author fewer commits Flag"]
        CT_author_value += commit_dict["T4 author first commit Flag"]
        CT_author_value += commit_dict["T5 author same day commits Flag"]
        CT_author_value += commit_dict["T6 author time since first commit Flag"]
        CT_author_value += commit_dict["T7 author rejected PRs Flag"]
        CT_committer_value = 0
        CT_committer_value += commit_dict["T1 committer does not exist Flag"]
        CT_committer_value += commit_dict["T2 committer account age Flag"]
        CT_committer_value += commit_dict["T3 committer fewer commits Flag"]
        CT_committer_value += commit_dict["T4 committer first commit Flag"]
        CT_committer_value += commit_dict["T5 committer same day commits Flag"]
        CT_committer_value += commit_dict["T6 committer time since first commit Flag"]
        CT_committer_value += commit_dict["T7 committer rejected PRs Flag"]

        if (CT_author_value / 7) >= rules["CTR Threshold"] or (CT_committer_value / 7) >= rules["CTR Threshold"]:
        # if (CT_value / 7) >= rules["CTR Threshold"]:
            commit_dict["R6 Untrusted author/committer"] = True
        else:
            commit_dict["R6 Untrusted author/committer"] = False

        
        ''' Generating Decision Model Rules Values '''
        sens_files = ["xml", "json", "jar", "ini", "dat", "cnf", "yml", "toml", \
                        "gradle", "bin", "config", "exe", "properties", "cmd", "build"]
        commit_dict["No of sensitive files"] = 0
        for sens_f in sens_files:
            if sens_f in dcf_df[dcf_df.commit == c_hash].file_type:
                commit_dict["No of sensitive files"] += 1
    
        if commit_dict["No of sensitive files"] >= rules["Sensitive Files Threshold"]:
            commit_dict["R1 Sensitive Files Threshold Breach"] = True
        else:
            commit_dict["R1 Sensitive Files Threshold Breach"] = False


        own_ex_files = rules["Ownership Excluded Files"]
        commit_dict["author No of new files modified"] = 0
        commit_dict["R2 author No of new files modified"] = False
        commit_dict["R3 author Owner or Majority Contributor Threshold Breach"] = True

        author_commit_file_df = dcf_df[dcf_df.author_login == commit_dict["author login"]]
        for f_type in own_ex_files:
            if not author_commit_file_df[author_commit_file_df.file_type == f_type].empty:
                author_commit_file_df[author_commit_file_df.file_type == f_type].file_name = ""
                # author_commit_file_df = author_commit_file_df.drop(author_commit_file_df.index[author_commit_file_df.file_type == f_type])
        author_commit_file_df = author_commit_file_df.sort_values(by=['commit']) 
        commits_df = author_commit_file_df.groupby("commit").count().reset_index()
        commits_df = commits_df.sort_values(by=['commit']) 
        if not commit_dict["T4 author first commit Flag"]:
            
            commit_nr = commits_df.index[commits_df.commit == c_hash].item()
            previous_commits = commits_df.commit[:commit_nr]
            for file in author_commit_file_df[author_commit_file_df.commit == c_hash].file_name:
                if file != "":
                    previous_files = []
                    for c in previous_commits:
                        previous_files.append(author_commit_file_df[author_commit_file_df.commit == c].file_name.to_list())
                    previous_files = list(flatten(previous_files))
                    current_files = previous_files
                    if file not in previous_files:
                        commit_dict["author No of new files modified"] += 1
                        current_files.append(file)
            if commit_dict["author No of new files modified"] / len(current_files) \
                >= rules["Ownership Un-touched Files Threshold"]:
                commit_dict["R2 author No of new files modified"] = True

            author_files_df = author_commit_file_df[author_commit_file_df.file_name != ""]
            if ((sum(author_files_df.author_file_majority_contributor.values) + sum(author_files_df.author_file_owner.values)) \
                    / len(author_files_df.file_name.values)) \
                    >= rules["Ownership Un-owned Files Threshold"]:
                commit_dict["R3 author Owner or Majority Contributor Threshold Breach"] = False

            # if ((len(author_files_df.file_name.values) - sum(author_files_df.author_file_owner.values)) \
            #         / sum(author_commit_file_df.author_file_owner.values)) \
            #     >= rules["Ownership Un-owned Files Threshold"] \
            #     or ((len(author_files_df.file_name.values) - sum(author_files_df.author_file_majority_contributor.values)) \
            #         / sum(author_files_df.author_file_majority_contributor.values)) \
            #         >= rules["Ownership Un-owned Files Threshold"]:
            #     commit_dict["R3 author Owner or Majority Contributor Threshold Breach"] = True

        commit_dict["R4 author Files touched Breach"] = False 
        if commit_dict["NOF_added"] == True:
            touched_files = author_commit_file_df[author_commit_file_df.commit == c_hash].file_name.values
            owned_mjc_files = author_commit_file_df[(author_commit_file_df.author_file_owner == True) | (author_commit_file_df.author_file_majority_contributor == True)].file_name.values
            for tf in touched_files:
                if tf not in owned_mjc_files:
                    commit_dict["R4 author Files touched Breach"] = True  

        commit_dict["committer No of new files modified"] = 0
        commit_dict["R2 committer No of new files modified"] = False
        commit_dict["R3 committer Owner or Majority Contributor Threshold Breach"] = True

        committer_commit_file_df = dcf_df[dcf_df.committer_login == commit_dict["committer login"]]
        for f_type in own_ex_files:
            if not committer_commit_file_df[committer_commit_file_df.file_type == f_type].empty:
                committer_commit_file_df[committer_commit_file_df.file_type == f_type].file_name = ""                
                # committer_commit_file_df = committer_commit_file_df.drop(committer_commit_file_df.index[committer_commit_file_df.file_type == f_type])
        committer_commit_file_df = committer_commit_file_df.sort_values(by=['commit']) 
        commits_df = committer_commit_file_df.groupby("commit").count().reset_index()
        commits_df = commits_df.sort_values(by=['commit']) 
        if not commit_dict["T4 committer first commit Flag"]:
            
            commit_nr = commits_df.index[commits_df.commit == c_hash].item()
            previous_commits = commits_df.commit[:commit_nr]
            for file in committer_commit_file_df[committer_commit_file_df.commit == c_hash].file_name:
                if file != "":
                    previous_files = []
                    for c in previous_commits:
                        previous_files.append(committer_commit_file_df[committer_commit_file_df.commit == c].file_name.to_list())
                    previous_files = list(flatten(previous_files))
                    current_files = previous_files
                    if file not in previous_files:
                        commit_dict["committer No of new files modified"] += 1
                        current_files.append(file)
            if commit_dict["committer No of new files modified"] / len(current_files) \
                >= rules["Ownership Un-touched Files Threshold"]:
                commit_dict["R2 committer No of new files modified"] = True

            committer_files_df = committer_commit_file_df[committer_commit_file_df.file_name != ""]
            if ((sum(committer_files_df.committer_file_majority_contributor.values) + sum(committer_files_df.committer_file_owner.values)) \
                / len(committer_files_df.file_name.values)) \
                >= rules["Ownership Un-owned Files Threshold"]:
                commit_dict["R3 committer Owner or Majority Contributor Threshold Breach"] = False

            # if ((len(committer_files_df.file_name.values) - sum(committer_files_df.committer_file_owner.values)) \
            #     / sum(committer_commit_file_df.committer_file_owner.values)) \
            #     >= rules["Ownership Un-owned Files Threshold"] \
            #     or ((len(committer_files_df.file_name.values) - sum(committer_files_df.committer_file_majority_contributor.values)) \
            #         / sum(committer_files_df.committer_file_majority_contributor.values)) \
            #         >= rules["Ownership Un-owned Files Threshold"]:
            #     commit_dict["R3 committer Owner or Majority Contributor Threshold Breach"] = True
        
        commit_dict["R4 committer Files touched Breach"] = False 
        if commit_dict["NOF_added"] == True:
            touched_files = committer_commit_file_df[committer_commit_file_df.commit == c_hash].file_name.values
            owned_mjc_files = committer_commit_file_df[(committer_commit_file_df.committer_file_owner == True) | (committer_commit_file_df.committer_file_majority_contributor == True)].file_name.values
            for tf in touched_files:
                if tf not in owned_mjc_files:
                    commit_dict["R4 committer Files touched Breach"] = True     

        commit_dict["R7 Commit linked to rejected PRs"] = (commit_dict["T7 author rejected PRs Flag"] | commit_dict["T7 committer rejected PRs Flag"])
        
        Decision_Model_Value = 0
        Decision_Model_Value += commit_dict["R1 Sensitive Files Threshold Breach"]
        Decision_Model_Value += (commit_dict["R2 author No of new files modified"] | commit_dict["R2 committer No of new files modified"])
        Decision_Model_Value += (commit_dict["R3 author Owner or Majority Contributor Threshold Breach"] | commit_dict["R3 committer Owner or Majority Contributor Threshold Breach"])
        Decision_Model_Value += (commit_dict["R4 author Files touched Breach"] | commit_dict["R4 committer Files touched Breach"])
        Decision_Model_Value += commit_dict["R5 OCP Threshold Breach"]
        Decision_Model_Value += commit_dict["R6 Untrusted author/committer"]
        Decision_Model_Value += commit_dict["R7 Commit linked to rejected PRs"]
        if (Decision_Model_Value / 7) >= rules["DMR Threshold"]:
        # if ((OCP_value + CT_value + Decision_Model_Value) / 21) >= rules["DMR Threshold"]:
            commit_dict["Flagged"] = True
        else:
            commit_dict["Flagged"] = False
            

        commits_list.append([commit_dict, commit_dict["Flagged"]])
        n_flagged_commits += commit_dict["Flagged"]

    print("Number of Flagged Commits in Repo " + url + ": ", n_flagged_commits)
    print("Number of Flagged Commits in Repo " + url + ": ", n_flagged_commits, file=out_f)

    return url, commits_list, n_flagged_commits                



def gen_dataset(urls_file_path:str = "data/urls_lists/urls_list_all_malicious.txt"):    

    with open(urls_file_path) as f:
        urls_list = f.read().split("\n")    
    
    data_ocprops_df, data_dcf_df, data_dev_commit_df, \
        data_author_trust_factors_df, data_committer_trust_factors_df \
            = [], [], [], [], []

    # to_date = pd.to_datetime('20200430', format='%Y%m%d', errors='coerce')

    for url in urls_list:
        i = 0
        project_id = url.split('/')[-1]
        print('Generating data for project: '+project_id)

        if project_id == 'php-src':
            since = pd.to_datetime('20210315', format='%Y%m%d', errors='coerce')
            to = pd.to_datetime('20210415', format='%Y%m%d', errors='coerce')            
        else:
            since = None
            to = None

        g = Github("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")
        reponame = "/".join(url.split("/")[-2:])
        repo = g.get_repo(reponame)
        
        gitapi_commit_dict = {
                                "author_login": [],
                                "author_created_at": [],
                                "committer_login": [],
                                "committer_created_at": [],
                                "commit_hash": []
                                }
        gitapi_commits_r = repo.get_commits(since=since, until=to)
        np.save("src/baseline/tmp/gitapi_commits.npy", np.array(gitapi_commits_r, dtype=object))
        gitapi_commits = np.load("src/baseline/tmp/gitapi_commits.npy", allow_pickle=True)
        for c in gitapi_commits.item():
            i += 1
            if i % 2400 == 0:
                print("Waiting...")
                time.sleep(3600)

            print("Commit: ", i)
            try:
                gitapi_commit_dict["author_login"].append(c.author.login)
            except:
                gitapi_commit_dict["author_login"].append("")
            try:
                gitapi_commit_dict["author_created_at"].append(c.author.created_at)
            except:
                gitapi_commit_dict["author_created_at"].append("")
            try:
                gitapi_commit_dict["committer_login"].append(c.committer.login)
            except:
                gitapi_commit_dict["committer_login"].append("")
            try:
                gitapi_commit_dict["committer_created_at"].append(c.committer.created_at)
            except:
                gitapi_commit_dict["committer_created_at"].append("")
            gitapi_commit_dict["commit_hash"].append(c.sha)
        gitapi_commit_df = pd.DataFrame(gitapi_commit_dict)

        pulls_r = repo.get_pulls(state='all', sort='created')
        np.save("src/baseline/tmp/pulls.npy", np.array(pulls_r, dtype=object))
        pulls = np.load("src/baseline/tmp/pulls.npy", allow_pickle=True)
        author_pull_requests_dict = {
                                "author_login": [],
                                "n_PRs": [],
                                "n_open_PRs": [],
                                "n_rejected_PRs": []
                                }
        committer_pull_requests_dict = {
                                "committer_login": [],
                                "n_PRs": [],
                                "n_open_PRs": [],
                                "n_rejected_PRs": []
                                }
        for p, pr in enumerate(pulls.item()): 
            i += 1
            if i % 2400 == 0:
                print("Waiting...")
                time.sleep(3600)
            print("Pull Request: ", p)           
            author_pull_requests_dict["author_login"].append(pr.user.login)
            author_pull_requests_dict["n_PRs"].append(1)
            committer_pull_requests_dict["committer_login"].append(pr.user.login)
            committer_pull_requests_dict["n_PRs"].append(1)
            if pr.state == "open":
                author_pull_requests_dict["n_open_PRs"].append(1)
                author_pull_requests_dict["n_rejected_PRs"].append(0)
                committer_pull_requests_dict["n_open_PRs"].append(1)
                committer_pull_requests_dict["n_rejected_PRs"].append(0)
            else:
                author_pull_requests_dict["n_open_PRs"].append(0)
                committer_pull_requests_dict["n_open_PRs"].append(0)
                if not pr.merged_at:
                    author_pull_requests_dict["n_rejected_PRs"].append(1)
                    committer_pull_requests_dict["n_rejected_PRs"].append(1)
                else:
                    author_pull_requests_dict["n_rejected_PRs"].append(0)
                    committer_pull_requests_dict["n_rejected_PRs"].append(0)
        author_pull_requests_df = pd.DataFrame(author_pull_requests_dict)
        author_pull_requests_df = author_pull_requests_df.groupby("author_login").sum().reset_index()
        
        committer_pull_requests_df = pd.DataFrame(committer_pull_requests_dict)
        committer_pull_requests_df = committer_pull_requests_df.groupby("committer_login").sum().reset_index()
        

        repo_ocprops_df, repo_dcf_df \
            = pd.DataFrame(), pd.DataFrame()

        ''' Generating graphs for each node type using Pydriller '''   
        # for commit in RepositoryMining(url, to=to_date).traverse_commits():
        for commit in RepositoryMining(url, since=since, to=to).traverse_commits():
            # print(commit.author.name)  
            if not commit.merge:   

                ''' Outlier Change Properties '''
                ocprops_dict = {
                                "LOC_added": 0, 
                                "LOC_removed": 0, 
                                "NOF_added": 0,
                                "NOF_removed": 0,
                                "NOF_modified": 0,
                                "NOF_renamed": 0,
                                "filetypes": [],
                                "NOF_uniquetypes": 0
                                }   
                ''' File History '''
                dcf_dict = {
                            # "author": [], 
                            "author_login": [], 
                            "committer_login": [], 
                            "commit_date": [], 
                            "commit": [], 
                            "commit_message": [], 
                            "file_name": [],
                            "file_type": [],
                            "file_owner": [],
                            "author_exists": [],
                            "author_account_age": [],
                            "committer_exists": [],
                            "committer_account_age": []
                            }   

                for m, modific in enumerate(commit.modifications):
                    ocprops_dict["LOC_added"] += modific.added
                    ocprops_dict["LOC_removed"] += modific.removed
                    ocprops_dict["NOF_added"] += modific.change_type.name == "ADD"
                    ocprops_dict["NOF_removed"] += modific.change_type.name == "DELETE"
                    ocprops_dict["NOF_modified"] += modific.change_type.name == "MODIFY"
                    ocprops_dict["NOF_renamed"] += modific.change_type.name == "RENAME"
                    if len(modific.filename.split(".")) == 1:
                        ocprops_dict["filetypes"].append("unknown")
                    else:
                        ocprops_dict["filetypes"].append(modific.filename.split(".")[-1])

                    # dcf_dict["author"].append(commit.author.name)
                    dcf_dict["commit_date"].append(commit.committer_date)
                    dcf_dict["commit"].append(commit.hash)
                    dcf_dict["commit_message"].append(commit.msg)
                    dcf_dict["file_name"].append(modific.filename)
                    dcf_dict["file_type"].append(modific.filename.split(".")[-1])
                    if modific.change_type == "ADD":
                        dcf_dict["file_owner"].append(True)
                    else:
                        dcf_dict["file_owner"].append(False)

                    # gitapi_commit = repo.get_commit(commit.hash)
                    # username = "".join(commit.author.name).replace(" ", "")
                    author_username = gitapi_commit_df[gitapi_commit_df.commit_hash == commit.hash].author_login.iloc[0]
                    dcf_dict["author_login"].append(author_username)
                    created_at = gitapi_commit_df[gitapi_commit_df.commit_hash == commit.hash].author_created_at.iloc[0]
                    if author_username != "":
                        dcf_dict["author_exists"].append(True)
                    else:
                        dcf_dict["author_exists"].append(False)  
                    if created_at != "":
                        dcf_dict["author_account_age"].append((datetime.datetime.today() - created_at).days)
                    else:
                        dcf_dict["author_account_age"].append(np.nan)   

                    committer_username = gitapi_commit_df[gitapi_commit_df.commit_hash == commit.hash].committer_login.iloc[0]
                    dcf_dict["committer_login"].append(committer_username)
                    created_at = gitapi_commit_df[gitapi_commit_df.commit_hash == commit.hash].committer_created_at.iloc[0]
                    if committer_username != "":
                        dcf_dict["committer_exists"].append(True)
                    else:
                        dcf_dict["committer_exists"].append(False)  
                    if created_at != "":
                        dcf_dict["committer_account_age"].append((datetime.datetime.today() - created_at).days)
                    else:
                        dcf_dict["committer_account_age"].append(np.nan)                                  

                ocprops_dict["NOF_uniquetypes"] = len(np.unique(ocprops_dict["filetypes"]))
                ocprops_dict.pop("filetypes")
                ocprops_df = pd.DataFrame([ocprops_dict])
                ocprops_df["author_login"] = author_username
                ocprops_df["committer_login"] = committer_username
                ocprops_df["commit_date"] = commit.committer_date
                ocprops_df["commit"] = commit.hash                
                
                repo_ocprops_df = pd.concat([repo_ocprops_df, ocprops_df])

                dcf_df = pd.DataFrame(dcf_dict)
                repo_dcf_df = pd.concat([repo_dcf_df, dcf_df])


        # dev_commit_df = repo_dcf_df.groupby(["author", "author_login", "commit_date", "commit"]).count().reset_index()
        dev_commit_df = repo_dcf_df.groupby(["author_login", "committer_login", "commit_date", "commit"]).count().reset_index()

        # trust_factors_df = dev_commit_df.groupby(["author", "author_login"]).count().reset_index()
        author_trust_factors_df = dev_commit_df.groupby(["author_login"]).count().reset_index()
        author_trust_factors_df.rename(columns = {'commit':'author_commits'}, inplace = True)
        author_trust_factors_df["author_time_since_first_commit"] = 0
        dev_commit_df["author_first_commit"] = 0
        dev_commit_df["author_no_of_same_day_commits"] = 0
        dev_commit_df.commit_date = pd.to_datetime(dev_commit_df.commit_date, utc=True)
        dev_commit_df = dev_commit_df.sort_values(by=['commit_date']) 
        dev_commit_df["author_day"] = [int(str(d.year)+str(d.day_of_year)) if not pd.isnull(d) else pd.NaT for d in dev_commit_df.commit_date]

        for author in author_trust_factors_df.author_login:
            author_df = repo_dcf_df[repo_dcf_df.author_login == author]
            author_df.commit_date = pd.to_datetime(author_df.commit_date, utc=True)
            author_df = author_df.sort_values(by=['commit_date']) 
            author_df["author_day"] =  [int(str(d.year)+str(d.day_of_year)) if not pd.isnull(d) else pd.NaT for d in author_df.commit_date]

            author_trust_factors_df["author_exists"][author_trust_factors_df.author_login == author] = author_df["author_exists"].iloc[0]
            author_trust_factors_df["author_account_age"][author_trust_factors_df.author_login == author] = author_df["author_account_age"].iloc[0]

            n_days = (author_df.commit_date.iloc[-1] - author_df.commit_date.iloc[0]).total_seconds() / (24 * 60 * 60)
            author_trust_factors_df["author_time_since_first_commit"][author_trust_factors_df.author_login == author] = n_days

            for commit in dev_commit_df[dev_commit_df.author_login == author].commit:
                if n_days == 0:
                    dev_commit_df["author_first_commit"][(dev_commit_df.commit == commit) & (dev_commit_df.author_login == author)] = 1

            temp_df = author_df.groupby(["author_login", "commit", "author_day"]).count().reset_index()
            author_day_df = temp_df.groupby(["author_login", "author_day"]).count().reset_index()
            author_day_df.rename(columns = {'commit':'author_no_of_same_day_commits'}, inplace = True)

            for day in dev_commit_df[dev_commit_df.author_login == author].author_day:
                dev_commit_df["author_no_of_same_day_commits"][(dev_commit_df.author_day == day) & (dev_commit_df.author_login == author)] \
                    = author_day_df["author_no_of_same_day_commits"][(author_day_df.author_day == day) & (author_day_df.author_login == author)].iloc[0]
          
            repo_dcf_df["author_n_PRs"] = 0
            repo_dcf_df["author_n_open_PRs"] = 0
            repo_dcf_df["author_n_rejected_PRs"] = 0
            # print(pull_requests_df[pull_requests_df.author_login == author])
            if not author_pull_requests_df[author_pull_requests_df.author_login == author].empty:
                repo_dcf_df["author_n_PRs"] = author_pull_requests_df[author_pull_requests_df.author_login == author].n_PRs.iloc[0]
                repo_dcf_df["author_n_open_PRs"] = author_pull_requests_df[author_pull_requests_df.author_login == author].n_open_PRs.iloc[0]
                repo_dcf_df["author_n_rejected_PRs"] = author_pull_requests_df[author_pull_requests_df.author_login == author].n_rejected_PRs.iloc[0]
            
            # print(repo_dcf_df[repo_dcf_df.author_login == author])   

        committer_trust_factors_df = dev_commit_df.groupby(["committer_login"]).count().reset_index()
        committer_trust_factors_df.rename(columns = {'commit':'committer_commits'}, inplace = True)
        committer_trust_factors_df["committer_time_since_first_commit"] = 0
        dev_commit_df["committer_first_commit"] = 0
        dev_commit_df["committer_no_of_same_day_commits"] = 0
        dev_commit_df.commit_date = pd.to_datetime(dev_commit_df.commit_date, utc=True)
        dev_commit_df = dev_commit_df.sort_values(by=['commit_date']) 
        dev_commit_df["committer_day"] = [int(str(d.year)+str(d.day_of_year)) if not pd.isnull(d) else pd.NaT for d in dev_commit_df.commit_date]
           
        for committer in committer_trust_factors_df.committer_login:
            committer_df = repo_dcf_df[repo_dcf_df.committer_login == committer]
            committer_df.commit_date = pd.to_datetime(committer_df.commit_date, utc=True)
            committer_df = committer_df.sort_values(by=['commit_date']) 
            committer_df["committer_day"] =  [int(str(d.year)+str(d.day_of_year)) if not pd.isnull(d) else pd.NaT for d in committer_df.commit_date]

            committer_trust_factors_df["committer_exists"][committer_trust_factors_df.committer_login == committer] = committer_df["committer_exists"].iloc[0]
            committer_trust_factors_df["committer_account_age"][committer_trust_factors_df.committer_login == committer] = committer_df["committer_account_age"].iloc[0]

            n_days = (committer_df.commit_date.iloc[-1] - committer_df.commit_date.iloc[0]).total_seconds() / (24 * 60 * 60)
            committer_trust_factors_df["committer_time_since_first_commit"][committer_trust_factors_df.committer_login == committer] = n_days

            for commit in dev_commit_df[dev_commit_df.committer_login == committer].commit:
                if n_days == 0:
                    dev_commit_df["committer_first_commit"][(dev_commit_df.commit == commit) & (dev_commit_df.committer_login == committer)] = 1

            temp_df = committer_df.groupby(["committer_login", "commit", "committer_day"]).count().reset_index()
            committer_day_df = temp_df.groupby(["committer_login", "committer_day"]).count().reset_index()
            committer_day_df.rename(columns = {'commit':'committer_no_of_same_day_commits'}, inplace = True)

            for day in dev_commit_df[dev_commit_df.committer_login == committer].committer_day:
                dev_commit_df["committer_no_of_same_day_commits"][(dev_commit_df.committer_day == day) & (dev_commit_df.committer_login == committer)] \
                    = committer_day_df["committer_no_of_same_day_commits"][(committer_day_df.committer_day == day) & (committer_day_df.committer_login == committer)].iloc[0]
          
            repo_dcf_df["committer_n_PRs"] = 0
            repo_dcf_df["committer_n_open_PRs"] = 0
            repo_dcf_df["committer_n_rejected_PRs"] = 0
            # print(pull_requests_df[pull_requests_df.committer_login == committer])
            if not committer_pull_requests_df[committer_pull_requests_df.committer_login == committer].empty:
                repo_dcf_df["committer_n_PRs"] = committer_pull_requests_df[committer_pull_requests_df.committer_login == committer].n_PRs.iloc[0]
                repo_dcf_df["committer_n_open_PRs"] = committer_pull_requests_df[committer_pull_requests_df.committer_login == committer].n_open_PRs.iloc[0]
                repo_dcf_df["committer_n_rejected_PRs"] = committer_pull_requests_df[committer_pull_requests_df.committer_login == committer].n_rejected_PRs.iloc[0]
            
            # print(repo_dcf_df[repo_dcf_df.committer_login == committer])            

        # trust_factors_df = trust_factors_df[["author", "author_login", "author_exists", "account_age", "dev_commits", "time_since_first_commit"]]
        # dev_commit_df = dev_commit_df[["author", "author_login", "commit_date", "commit", "first_commit", "no_of_same_day_commits", "day"]]
        author_trust_factors_df = author_trust_factors_df[["author_login", "author_exists", "author_account_age", \
                                                    "author_commits", "author_time_since_first_commit"]]
        committer_trust_factors_df = committer_trust_factors_df[["committer_login", "committer_exists", "committer_account_age", \
                                                    "committer_commits", "committer_time_since_first_commit"]]
        dev_commit_df = dev_commit_df[["author_login", "committer_login", "commit_date", "commit", \
                                        "author_first_commit", "author_no_of_same_day_commits", "author_day", \
                                            "committer_first_commit", "committer_no_of_same_day_commits", "committer_day"]]

        data_ocprops_df.append(repo_ocprops_df)
        data_dcf_df.append(repo_dcf_df)      
        data_dev_commit_df.append(dev_commit_df)      
        data_author_trust_factors_df.append(author_trust_factors_df)      
        data_committer_trust_factors_df.append(committer_trust_factors_df)      
    
        print("Waiting...")
        time.sleep(3)

        pickle.dump(repo_ocprops_df,  open("src/baseline/tmp/ocprops_df"+str(project_id)+".pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(repo_dcf_df,  open("src/baseline/tmp/dcf_df"+str(project_id)+".pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(dev_commit_df,  open("src/baseline/tmp/dev_commit_df"+str(project_id)+".pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(author_trust_factors_df,  open("src/baseline/tmp/author_trust_factors_df"+str(project_id)+".pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(committer_trust_factors_df,  open("src/baseline/tmp/committer_trust_factors_df"+str(project_id)+".pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)

    # np.save("src/baseline/tmp/data_ocprops_df"+str(len(urls_list))+".npy", np.array(data_ocprops_df, dtype=object))
    # np.save("src/baseline/tmp/data_dcf_df"+str(len(urls_list))+".npy", np.array(data_dcf_df, dtype=object))
    # np.save("src/baseline/tmp/data_dev_commit_df"+str(len(urls_list))+".npy", np.array(data_dev_commit_df, dtype=object))
    # np.save("src/baseline/tmp/data_author_trust_factors_df"+str(len(urls_list))+".npy", np.array(data_author_trust_factors_df, dtype=object))
    # np.save("src/baseline/tmp/data_committer_trust_factors_df"+str(len(urls_list))+".npy", np.array(data_committer_trust_factors_df, dtype=object))


def analyze_repos(rules, urls_list, out_f, b_min, rules_list):
    print("\nRules List: ", rules_list)
    print("\nRules List: ", rules_list, file=out_f)
    analyzed_repos_results, flagged = [], []
    for url in urls_list:
        project_id = url.split('/')[-1]
        print("Analyzing Repo: ", project_id)
        print("Analyzing Repo: ", project_id, file=out_f)
        ocprops_df = pickle.load(open("src/baseline/tmp/ocprops_df"+str(project_id)+".pkl", "rb"))
        dcf_df = pickle.load(open("src/baseline/tmp/dcf_df"+str(project_id)+".pkl", "rb"))
        dev_commit_df = pickle.load(open("src/baseline/tmp/dev_commit_df"+str(project_id)+".pkl", "rb"))
        author_trust_factors_df = pickle.load(open("src/baseline/tmp/author_trust_factors_df"+str(project_id)+".pkl", "rb"))
        committer_trust_factors_df = pickle.load(open("src/baseline/tmp/committer_trust_factors_df"+str(project_id)+".pkl", "rb"))

        url, commits_list, n_flagged_commits = gen_commit_reports(ocprops_df, dcf_df, \
                            dev_commit_df, author_trust_factors_df, committer_trust_factors_df, \
                                rules, url, out_f, rules_list)
        analyzed_repos_result = [url, commits_list, n_flagged_commits, n_flagged_commits/len(commits_list)]
        flagged.append(n_flagged_commits/len(commits_list))
        
        # analyzed_repos_results.append([url, commits_list, n_flagged_commits, n_flagged_commits/len(commits_list)])

        pickle.dump(analyzed_repos_result,  open("src/baseline/tmp/analyzed_repo_result"+str(project_id)+".pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    # np.save("src/baseline/tmp/analyzed_repos_results"+str(len(urls_list))+".npy", np.array(analyzed_repos_results, dtype=object))

    if b_min:
        mean_flagged = np.mean(flagged)
        if mean_flagged >= 0.001:
            return mean_flagged
        else:
            return 1.
    print("\nMean percentage of flagged commits: ", np.mean(flagged))
    print("\nMean percentage of flagged commits: ", np.mean(flagged), file=out_f)


def bayes_opt(analyze_repos, rules_list:list, rules:dict, urls_list:list, out_f:str, b_min:bool, n_calls:int):    
    print("Performing Bayesian Minimization")
    res_gp = gp_minimize(partial(analyze_repos, rules, urls_list, out_f, b_min), rules_list, n_calls=n_calls)
    print("All Results: \n", res_gp)
    print("All Results: \n", res_gp, file=out_f)
    print("Best Hyperparameters: ", res_gp.x)
    print("Best Hyperparameters: ", res_gp.x, file=out_f)
    with open('best_hp.txt', 'wb') as F:
        pickle.dump(res_gp.x, F)
    return res_gp.x



'''__________ Main _________'''


''' Generating Dataset from Git Repos '''
root = os.path.dirname(os.path.realpath(__file__)) 
urls_file_path =  os.path.join(root, "..", "..", "data", "urls_lists", "urls_list_Octopus.txt")
with open(urls_file_path) as f:
        urls_list = f.read().split("\n")

os.system('mkdir ' + root + "/tmp/")
os.system('mkdir ' + root + "/tmp/"+str(datetime.date.today()))


# st = time.time()
# gen_dataset(urls_file_path = urls_file_path)
# print("\nTime taken to extract "+str(len(urls_list))+" repos", np.round((time.time() - st)/60), " minutes\n")

# data_ocprops_df = np.load("src/baseline/tmp/data_ocprops_df"+str(len(urls_list))+".npy", allow_pickle=True)
# data_dcf_df = np.load("src/baseline/tmp/data_dcf_df"+str(len(urls_list))+".npy", allow_pickle=True)
# data_dev_commit_df = np.load("src/baseline/tmp/data_dev_commit_df"+str(len(urls_list))+".npy", allow_pickle=True)
# data_author_trust_factors_df = np.load("src/baseline/tmp/data_author_trust_factors_df"+str(len(urls_list))+".npy", allow_pickle=True)
# data_committer_trust_factors_df = np.load("src/baseline/tmp/data_committer_trust_factors_df"+str(len(urls_list))+".npy", allow_pickle=True)

b_min = False
if b_min:
    ''' Performing Bayesian Minimization ''' 
    rules_list = [
                Categorical(np.arange(2/7, 5/7, 1/7), name="DMR Threshold"),
                Categorical(np.arange(2/7, 5/7, 1/7), name="OCP Threshold"),
                Categorical(np.arange(7, 14, 7), name="CT Min. Time as Contributor Threshold"),
                Categorical(np.arange(2/7, 5/7, 1/7), name="CTR Threshold"),
                Categorical(np.arange(0.025, 0.075, 0.025), name="CT Few commits Threshold"),
                Categorical(np.arange(0.25, 0.75, 0.25), name="CT Same Day commit Threshold"),
                Categorical(np.arange(0.25, 0.75, 0.25), name="Contrib. Trust Un-merged PR Threshold"),
                Categorical(np.arange(0.0, 0.50, 0.25), name="Ownership Un-touched Files Threshold"),
                Categorical(np.arange(0.75, 1.25, 0.25), name="Ownership Un-owned Files Threshold")
            ]   
    rules = {}
    rules["File History Excluded Files"] = ["README", "gitignore"]
    rules["File History Consider First commit to File"] = True
    rules["Exclude History for New Contribs."] = True
    rules["Sensitive Files Threshold"] = 1
    rules["Ownership Excluded Files"] = ["class", "md", "gitignore"]
    rules["Ownership Consider Major Contributors."] = True
    n_calls = 100
    os.system('mkdir ' + root + "/tmp/"+str(datetime.date.today()))
    out_f = open(root + "/tmp/"+str(datetime.date.today())+"/bm_output_"+str(len(urls_list))\
                    +str(datetime.date.today())+".txt", "a")
    hp = bayes_opt(analyze_repos, rules_list, rules, urls_list, out_f, True, n_calls)

    ''' Loading optimum hyperparameters obtained from Bayesian Minimization ''' 
    with open ('best_hp.txt', 'rb') as F:
        hp = pickle.load(F)



# rules_list = [0.5, 0.5, 7, 0.5, 0.025, 0.2, 0.25, 0.0, 0.75]
rules_list = [0.5, 0.5, 7, 0.5, 0.05, 0.5, 0.5, 0.25, 0.75]
# rules_list = hp
out_f = open(root + "/tmp/"+str(datetime.date.today())+"/output_"+str(len(urls_list))+"_"+str(rules_list)+"_"+str(datetime.date.today())+".txt", "a")

rules = {}
rules["File History Excluded Files"] = ["README", "gitignore"]
rules["File History Consider First commit to File"] = True
rules["Exclude History for New Contribs."] = True
rules["Sensitive Files Threshold"] = 1
rules["Ownership Excluded Files"] = ["class", "md", "gitignore"]
rules["Ownership Consider Major Contributors."] = True

st = time.time()
analyze_repos(rules, urls_list, out_f, False, rules_list)
print("\nTime taken to analyze "+str(len(urls_list))+" repos", np.round((time.time() - st)/60), " minutes\n")

# analyzed_repos = np.load("src/baseline/tmp/analyzed_repos_results"+str(len(urls_list))+".npy", allow_pickle=True)
for url in urls_list:    
    project_id = url.split('/')[-1]
    analyzed_repo = pickle.load(open("src/baseline/tmp/analyzed_repo_result"+str(project_id)+".pkl", "rb"))
    print("\n\nNumber of Flagged Commits in Repo " + analyzed_repo[0] + ": ", analyzed_repo[2], file=out_f)
    for analyzed_commit in analyzed_repo[1]:
        if analyzed_commit[1] == True:
            print("\nDictionary of Flagged Commit: ", analyzed_commit[0], file=out_f)
