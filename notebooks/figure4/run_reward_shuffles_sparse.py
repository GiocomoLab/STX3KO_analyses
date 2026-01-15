import pathlib
import joblib
import numpy as np
import scipy as sp
import pandas as pd
import pickle

import pingouin as pg
import statsmodels.formula.api as smf



import STX3KO_analyses as stx
from STX3KO_analyses import utilities as u
from STX3KO_analyses.run_sig_fields_spatial_perm import filter_fields

sparse_mice = stx.ymaze_sess_deets.sparse_mice

rng = np.random.default_rng()

def run_model(df_perm, stat_column):
    # Permute genotype within each mouse if within-mouse design
    df_perm["chan"] = df_perm.groupby("mouse", observed=True)["chan"].transform(np.random.permutation)
    red_df = df_perm.groupby(['mouse', 'chan', 'day', 'ttype']).mean().reset_index()
    red_df['rank_frac'] = red_df['rzone_bool'].rank()

    md_perm = smf.mixedlm(f"{stat_column} ~ C(chan) * C(day) * C(ttype)", red_df, groups=red_df["mouse"])
    try:
        res_perm = md_perm.fit()
        if res_perm.converged:
            return res_perm.fe_params['C(chan)[T.channel_1]']
        else:
            return np.nan
    except:
        return np.nan
    

def run_pc_frac_sparse(n_perms=10000):



    frac_cls = stx.reward_overrep.PeriRewardCellFrac_Sparse(sparse_mice, 
                                                        np.arange(6), 
                                                        place_cell_only=True,
                                                        ts_key='F_dff',
                                                        for_shuffs=True)
    df = frac_cls.df_shuff
    
    
    red_df = df.groupby(['mouse', 'chan', 'day', 'ttype']).mean().reset_index()
    red_df['rank_frac'] = red_df['rzone_bool'].rank()

    result_dict = {}
    for stat_column in ('rzone_bool', 'rank_frac'):

        # Fit a mixed model on ranks
        model = smf.mixedlm(f"{stat_column} ~ C(chan) * C(day) * C(ttype)", red_df, groups=red_df["mouse"])
        model_result = model.fit()

        # run genotype shuffles
        shuff_coefs = np.array(joblib.Parallel(n_jobs=16)(joblib.delayed(run_model)(df.copy(),  
                                                                                    stat_column,
                                ) for p in range(n_perms)))



        result_dict[stat_column] = {'true_model': model_result,
                                    'genotype coef.': model_result.fe_params['C(chan)[T.channel_1]'],
                                    'shuffle genotype coef.': shuff_coefs,
                                    }
        
    return result_dict

def run_rr_model(df_perm, stat_column):
    # Permute genotype within each mouse if within-mouse design
    df_perm["chan"] = df_perm.groupby("mouse", observed=True)["chan"].transform(np.random.permutation)
    red_df = df_perm.groupby(['mouse', 'chan', 'day']).mean().reset_index()
    red_df['rank_frac'] = red_df['rzone_bool'].rank()

    aov = pg.rm_anova(data = red_df, dv = stat_column, within=['day', 'chan'], subject='mouse')
    return aov.loc[aov['Source']=='chan', 'F'].values
        
    # md_perm = smf.mixedlm(f"{stat_column} ~ C(chan) * C(day) ", red_df, groups=red_df["mouse"])
    # try:
    #     res_perm = md_perm.fit()
    #     if res_perm.converged:
    #         return res_perm.fe_params['C(chan)[T.channel_1]']
    #     else:
    #         return np.nan
    # except:
    #     return np.nan
    

def run_rr_frac_sparse(n_perms=10000):
    reward_cells = stx.reward_overrep.RewardCells_Sparse()
    df = reward_cells.build_summary_df_for_shuffles()
    
    
    red_df = df.groupby(['mouse', 'chan', 'day']).mean().reset_index()
    red_df['rank_frac'] = red_df['rzone_bool'].rank()

    result_dict = {}
    for stat_column in ('rzone_bool', 'rank_frac'):

        # Fit a mixed model on ranks
        aov = pg.rm_anova(data = red_df, dv = stat_column, within=['day', 'chan'], subject='mouse')
        # model = smf.mixedlm(f"{stat_column} ~ C(chan) * C(day) ", red_df, groups=red_df["mouse"])
        # model_result = model.fit()

        # run genotype shuffles
        shuff_coefs = np.array(joblib.Parallel(n_jobs=16)(joblib.delayed(run_rr_model)(df.copy(),  
                                                                                    stat_column,
                                ) for p in range(n_perms)))



        result_dict[stat_column] = {'true_model': aov, #model_result,
                                    'genotype coef.': aov.loc[aov['Source']=='chan', 'F'].values, #model_result.fe_params['C(chan)[T.channel_1]'],
                                    'shuffle genotype coef.': shuff_coefs,
                                    }
        
    return result_dict


if __name__ == "__main__":

    result_dict = run_pc_frac_sparse(n_perms=1000)
    with open('/home/mplitt/shuffle_pkls/sparse_pc_frac_mixedlm_shuffle.pkl', 'wb') as file:
        pickle.dump(result_dict, file)

    

    result_dict = run_rr_frac_sparse(n_perms=1000)
    with open('/home/mplitt/shuffle_pkls/sparse_rewardcell_mixedlm_shuffle.pkl', 'wb') as file:
        pickle.dump(result_dict, file)



