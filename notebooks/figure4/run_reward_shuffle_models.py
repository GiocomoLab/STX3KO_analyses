import pathlib
import joblib

import itertools
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import pickle
import pingouin as pg

import statsmodels.formula.api as smf



import STX3KO_analyses as stx
from STX3KO_analyses import utilities as u
from STX3KO_analyses.run_sig_fields_spatial_perm import filter_fields


ko_mice = stx.ymaze_sess_deets.ko_mice
ctrl_mice = stx.ymaze_sess_deets.ctrl_mice



def run_model(df_perm, all_mice, ctrl_mice, stat_column):

    genotype_map = {}
    for mouse in all_mice:
        genotype_map[mouse] = "ko"
    for mouse in ctrl_mice:
        genotype_map[mouse] = "ctrl"
    
    
    df_perm["cre"] = df_perm["mouse"].map(genotype_map)
    df_perm["cre"] = df_perm["cre"].astype("category")

    md_perm = smf.mixedlm(f"{stat_column} ~ cre * day * nov", df_perm, groups=df_perm["mouse"])
    try:
        res_perm = md_perm.fit()
        return res_perm.fe_params["cre[T.ko]"]
    except:
        return np.nan  # skip failed fits (rare)


def run_width_dense():


    with open('/home/mplitt/shuffle_pkls/dense_place_field_spatial_shuffle_F_dff.pkl','rb') as file:
        shuff_results = pickle.load(file)

    df = {'cre': [], 'mouse': [], 'day': [], 'nov': [], 'width': []}
    for key, mice in zip(('ctrl', 'ko'),(ctrl_mice, ko_mice)):
        for mouse in mice:
            for day in range(6):
                for nov in ( 'fam', 'nov'):

                    rising_edges = shuff_results[mouse][day][nov]['rising_edges']
                    falling_edges = shuff_results[mouse][day][nov]['falling_edges']
                    widths = shuff_results[mouse][day][nov]['field_widths']
                
                    mask = (widths>1) * (widths<25) * (rising_edges[:,1]>0) * (falling_edges[:,1]<29)
                    widths = widths[mask]
                    

                    for w in widths:
                        df['cre'].append(key)
                        df['mouse'].append(mouse)
                        df['day'].append(day)
                        df['nov'].append(nov)
                
                        df['width'].append(w*10)

    df = pd.DataFrame.from_dict(df)


    df['cre'] = df["cre"].astype("category")
    df["mouse"] = df["mouse"].astype("category")
    df["day"] = df["day"].astype("category")
    df["nov"] = df["nov"].astype("category")
    df["rank_width"] = df["width"].rank()

    result_dict = {}
    for stat_column in ('width', 'rank_width'):
        

        # Fit a mixed model on ranks
        model = smf.mixedlm(f"{stat_column} ~ cre * day * nov", df, groups=df["mouse"])
        model_result = model.fit()

        # run genotype shuffles
        shuff_coefs = np.array(joblib.Parallel(n_jobs=16)(joblib.delayed(run_model)(df.copy(), 
                                                                                    ctrl_mice+ko_mice, 
                                                                                    comb, 
                                                                                    stat_column,
                                ) for i, comb in enumerate(itertools.combinations(ctrl_mice+ko_mice,len(ctrl_mice))) ))



        result_dict[stat_column] = {'true_model': model_result,
                                    'genotype coef.': model_result.fe_params['cre[T.ko]'],
                                    'shuffle genotype coef.': shuff_coefs,
                                    }
        
    return result_dict

if __name__ == "__main__":

    result_dict = run_n_fields_dense()
    with open('/home/mplitt/shuffle_pkls/dense_n_fields_mixedlm_shuffle.pkl', 'wb') as file:
        pickle.dump(result_dict, file)

    result_dict = run_width_dense()
    with open('/home/mplitt/shuffle_pkls/dense_width_mixedlm_shuffle.pkl', 'wb') as file:
        pickle.dump(result_dict, file)