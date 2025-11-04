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
    
    
    df_perm["condition"] = df_perm["mouse"].map(genotype_map)
    # df_perm["condition"] = df_perm["condition"].astype("category")

    md_perm = smf.mixedlm(f"{stat_column} ~ C(ttype) + C(condition) + C(day) + speed + licks", df_perm, groups=df_perm["mouse"])
    try:
        res_perm = md_perm.fit()
        return res_perm.fe_params["C(condition)[T.ko]"]
    except:
        return np.nan  # skip failed fits (rare)





def run_novel_activity_dense():


    block_rate = stx.novel_activity_rate.BlockTransitionActivityRate(ts_key='F_dff')
    df = block_rate.build_dataframe(max_trial=5, norm='population', norm_behavior=False)

    df = df.loc[df['day']<5]
    df = df.loc[df['ttype'].isin(('familiar','novel')), :]

    
    df["rank_rate"] = df["rate"].rank()

    result_dict = {}
    for stat_column in ('rate', 'rank_rate'):
        

        # Fit a mixed model on ranks
        model = smf.mixedlm(f"{stat_column} ~ C(ttype) + C(condition) + C(day) + speed + licks", df, groups=df['mouse'])
        model_result = model.fit()

        # run genotype shuffles
        shuff_coefs = np.array(joblib.Parallel(n_jobs=16)(joblib.delayed(run_model)(df.copy(), 
                                                                                    ctrl_mice+ko_mice, 
                                                                                    comb, 
                                                                                    stat_column,
                                ) for i, comb in enumerate(itertools.combinations(ctrl_mice+ko_mice,len(ctrl_mice))) ))



        result_dict[stat_column] = {'true_model': model_result,
                                    'genotype coef.': model_result.fe_params['C(condition)[T.ko]'],
                                    'shuffle genotype coef.': shuff_coefs,
                                    }
        
    return result_dict




if __name__ == "__main__":

    result_dict = run_novel_activity_dense()
    with open('/home/mplitt/shuffle_pkls/dense_novel_activity_mixedlm_shuffle.pkl', 'wb') as file:
        pickle.dump(result_dict, file)

    