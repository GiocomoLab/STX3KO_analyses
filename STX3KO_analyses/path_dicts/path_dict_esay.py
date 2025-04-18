'''
This is an example path dictionary file. 
Edit the ROOT paths for your specific system 
     and save as "path_dict_user" or "path_dict_machine"
     for easy future access.

Example usage in a jupyter notebook:
` from InVivoDA_analyses.path_dict_msosa import path_dictionary as path dict `

'''

import os

### REMOTE ###
# RCLONE_DATA_ROOT = "DATA"

###  LOCAL  ###
HOME = os.path.expanduser("C://")

DATA_ROOT = os.path.join("C://","Users","esay","data","Stx3")  # parent path to data
PP_ROOT = DATA_ROOT # path to preprocessed data
PKL_ROOT = os.path.join("C://","Users","esay","data","Stx3","YMazeSessPkls")
SBX_ROOT = os.path.join("Z://","giocomo","InVivoDA","2P_Data") # scanbox data path, if different from preprocessed data path
# SBX_ROOT = os.path.join("C://","Users","esay","data","Stx3")  # parent path to data



GIT_ROOT = os.path.join(HOME,"repos")

FIG_DIR = os.path.join(DATA_ROOT,"fig_scratch")


path_dictionary = {
    "preprocessed_root": PP_ROOT,
    "pkl_root": PKL_ROOT,
    "sbx_root": SBX_ROOT,
    "VR_Data": os.path.join(PP_ROOT,"VR_Data"),
    "git_repo_root": GIT_ROOT,
    "TwoPUtils": os.path.join(GIT_ROOT,"TwoPUtils"),
    "home": HOME,
    "fig_dir": FIG_DIR,
}


# os.makedirs(path_dictionary,['fig_dir'],exist_ok=True)