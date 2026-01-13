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

sparse_mice = stx.ymaze_sess_deets.sparse_mice



def run_model(df_perm, stat_column):
    # Permute genotype within each mouse if within-mouse design
    df_perm["channel"] = df_perm.groupby("mouse", observed=True)["channel"].transform(np.random.permutation)
    red_df_perm = reduce_shuff_df(df_perm)
    red_df_perm["rank_norm_rate"] = red_df_perm["norm_rate"].rank()

    # md_perm = smf.mixedlm(f"{stat_column} ~ C(channel) + C(ttype) + C(day) + norm_speed + norm_licks", red_df_perm, groups=red_df_perm["mouse"])
    md_perm = smf.mixedlm(f"{stat_column} ~ C(ttype) + C(channel) *  C(day)", red_df_perm, groups=red_df_perm["mouse"])
    try:
        res_perm = md_perm.fit()
        if res_perm.converged:
            return {'C(channel)[T.channel_1]': res_perm.fe_params['C(channel)[T.channel_1]'],
                    'C(channel)[T.channel_1]:C(day)[T.1]': res_perm.fe_params['C(channel)[T.channel_1]:C(day)[T.1]'],
                    'C(channel)[T.channel_1]:C(day)[T.2]': res_perm.fe_params['C(channel)[T.channel_1]:C(day)[T.2]'],
                    'C(channel)[T.channel_1]:C(day)[T.3]': res_perm.fe_params['C(channel)[T.channel_1]:C(day)[T.3]'],
                    'C(channel)[T.channel_1]:C(day)[T.4]': res_perm.fe_params['C(channel)[T.channel_1]:C(day)[T.4]'],
            }
        else:
            return {'C(channel)[T.channel_1]': np.nan,
                'C(channel)[T.channel_1]:C(day)[T.1]': np.nan,
                'C(channel)[T.channel_1]:C(day)[T.2]': np.nan,
                'C(channel)[T.channel_1]:C(day)[T.3]': np.nan,
                'C(channel)[T.channel_1]:C(day)[T.4]': np.nan,
            }

    except:
        return {'C(channel)[T.channel_1]': np.nan,
                'C(channel)[T.channel_1]:C(day)[T.1]': np.nan,
                'C(channel)[T.channel_1]:C(day)[T.2]': np.nan,
                'C(channel)[T.channel_1]:C(day)[T.3]': np.nan,
                'C(channel)[T.channel_1]:C(day)[T.4]': np.nan,
        }


def reduce_shuff_df(df):
    red_df = df.groupby(['channel', 'mouse', 'day', 'ttype']).mean().reset_index()
    red_df['norm_rate'] = 0.
    red_df['norm_speed'] = 0.
    red_df['norm_licks'] = 0.
    for mouse in sparse_mice:
        for day in range(5):
            if (mouse == 'SparseKO_09') and (day==2):
                continue
            for chan in ('channel_0', 'channel_1'):
                mask = (red_df['mouse']==mouse) * (red_df['day']==day) * (red_df['channel']==chan)
                sub_df = red_df.loc[mask]
                base_act = sub_df.loc[sub_df['ttype']=='baseline']['rate'].values
                base_speed = sub_df.loc[sub_df['ttype']=='baseline']['speed'].values
                base_licks = sub_df.loc[sub_df['ttype']=='baseline']['licks'].values
                # print(base_act)
                red_df.loc[mask,'norm_rate'] = red_df.loc[mask,'rate'].values/base_act
                red_df.loc[mask,'norm_speed'] = red_df.loc[mask,'speed'].values/base_speed
                red_df.loc[mask,'norm_licks'] = red_df.loc[mask,'licks'].values/base_licks

    mask = (red_df['day']<5) * red_df['ttype'].isin(('familiar','novel')) 
    return red_df.loc[mask,:]


def run_novel_activity_sparse(n_perms=1000):


    block_rate = stx.novel_activity_rate.BlockTransitionActivityRate_Sparse(ts_key='spks')
    df = block_rate.build_dataframe_for_shuffles(max_trial=5)

    red_df = reduce_shuff_df(df)
    red_df["rank_norm_rate"] = red_df["norm_rate"].rank()

    result_dict = {}
    for stat_column in ('norm_rate', 'rank_norm_rate'):
        

        # Fit a mixed model on ranks
        # model = smf.mixedlm(f"{stat_column} ~ C(ttype) + C(channel) + C(day) + norm_speed + norm_licks", red_df, groups=red_df['mouse'])
        model = smf.mixedlm(f"{stat_column} ~ C(ttype) + C(channel) * C(day)", red_df, groups=red_df['mouse'])
        model_result = model.fit()

        # run genotype shuffles
        shuff_coefs_list = np.array(joblib.Parallel(n_jobs=16)(joblib.delayed(run_model)(df.copy(), 
                                                                                    stat_column,
                                ) for p in range(n_perms)))

        genotype_coefs = {'C(channel)[T.channel_1]': model_result.fe_params['C(channel)[T.channel_1]'],
                'C(channel)[T.channel_1]:C(day)[T.1]': model_result.fe_params['C(channel)[T.channel_1]:C(day)[T.1]'],
                'C(channel)[T.channel_1]:C(day)[T.2]': model_result.fe_params['C(channel)[T.channel_1]:C(day)[T.2]'],
                'C(channel)[T.channel_1]:C(day)[T.3]': model_result.fe_params['C(channel)[T.channel_1]:C(day)[T.3]'],
                'C(channel)[T.channel_1]:C(day)[T.4]': model_result.fe_params['C(channel)[T.channel_1]:C(day)[T.4]'],
        }

        shuff_coefs = {'C(channel)[T.channel_1]': [],
                'C(channel)[T.channel_1]:C(day)[T.1]': [],
                'C(channel)[T.channel_1]:C(day)[T.2]': [],
                'C(channel)[T.channel_1]:C(day)[T.3]': [],
                'C(channel)[T.channel_1]:C(day)[T.4]': [],
                }
 
        for shuff in shuff_coefs_list:
            for k,v  in shuff.items():
                shuff_coefs[k].append(v)

        for k,v in shuff_coefs.items():
            shuff_coefs[k] = np.array(v)

        result_dict[stat_column] = {'true_model': model_result,
                                    'genotype coef.': genotype_coefs,
                                    'shuffle genotype coef.': shuff_coefs,
                                    }
        
    return result_dict




if __name__ == "__main__":

    result_dict = run_novel_activity_sparse()
    with open('/home/mplitt/shuffle_pkls/sparse_novel_activity_mixedlm_shuffle_spks.pkl', 'wb') as file:
        pickle.dump(result_dict, file)

    