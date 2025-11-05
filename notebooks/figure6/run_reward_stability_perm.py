import pathlib
import joblib
import pickle

import itertools
import numpy as np
import scipy as sp
import pandas as pd


import pingouin as pg

import statsmodels.formula.api as smf



import STX3KO_analyses as stx
from STX3KO_analyses import utilities as u
from STX3KO_analyses.run_sig_fields_spatial_perm import filter_fields


ko_mice = stx.ymaze_sess_deets.ko_mice
ctrl_mice = stx.ymaze_sess_deets.ctrl_mice



def run_pc_model(df_perm, all_mice, ctrl_mice, stat_column):

    genotype_map = {}
    for mouse in all_mice:
        genotype_map[mouse] = "ko"
    for mouse in ctrl_mice:
        genotype_map[mouse] = "ctrl"
    
    
    df_perm["cond"] = df_perm["mouse"].map(genotype_map)


    md_perm = smf.mixedlm(f"{stat_column} ~ C(cond)*C(ttype)*C(day) + speed + licks", df_perm, groups=df_perm["mouse"])
    try:
        res_perm = md_perm.fit()
        if res_perm.converged:
            return res_perm.fe_params["C(cond)[T.ko]"]
        else:
            return np.nan
    except:
        return np.nan  # skip failed fits (rare)


def run_pc_frac_dense():
    
    df = pd.read_pickle('/home/mplitt/shuffle_pkls/reward_stability_df_dff.pkl')
    df['next_day_corr_rank'] = df['next_day_corr'].rank()

    result_dict = {}
    for stat_column in ('next_day_corr', 'next_day_corr_rank'):
        

        # Fit a mixed model on ranks
        model = smf.mixedlm(f"{stat_column} ~ C(day) + C(ttype) + C(reward_cell)*C(cond)", df, groups=df["mouse"])
        model_result = model.fit()

        # run genotype shuffles
        shuff_coefs = np.array(joblib.Parallel(n_jobs=16)(joblib.delayed(run_pc_model)(df.copy(), 
                                                                                    ctrl_mice+ko_mice, 
                                                                                    comb, 
                                                                                    stat_column,
                                ) for i, comb in enumerate(itertools.combinations(ctrl_mice+ko_mice,len(ctrl_mice))) ))



        result_dict[stat_column] = {'true_model': model_result,
                                    'genotype coef.': model_result.fe_params['C(cond)[T.ko]'],
                                    'shuffle genotype coef.': shuff_coefs,
                                    }
        
    return result_dict


 


if __name__ == "__main__":

    result_dict = run_pc_frac_dense()
    with open('/home/mplitt/shuffle_pkls/dense_pc_frac_mixedlm_shuffle.pkl', 'wb') as file:
        pickle.dump(result_dict, file)



   