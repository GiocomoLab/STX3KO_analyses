import pathlib
import numpy as np
import scipy as sp
import joblib
import pandas as pd
import pingouin as pg
import seaborn as sns
import pickle

import statsmodels.formula.api as smf

import TwoPUtils as tpu
import STX3KO_analyses as stx
from STX3KO_analyses import utilities as u

sparse_mice = stx.ymaze_sess_deets.sparse_mice


rng = np.random.default_rng()

def run_model(df_perm, stat_column):
    # Permute genotype within each mouse if within-mouse design
    df_perm["channel"] = df_perm.groupby("mouse", observed=True)["channel"].transform(np.random.permutation)

    md_perm = smf.mixedlm(f"{stat_column} ~ C(day) * C(channel) * C(ttype)", df_perm, groups=df_perm["mouse"])
    try:
        res_perm = md_perm.fit()
        return res_perm.fe_params['C(channel)[T.channel_1]']
    except:
        return np.nan
    

def run_pc_fracs(n_perms=10000):

    rows = []
    for mouse in sparse_mice:
        for day in range(6):
            if mouse == 'SparseKO_09' and day == 2:
                continue
            sess = u.load_single_day(mouse,day)


            for chan in ('channel_0', 'channel_1'):
                for ttype in ('fam', 'nov'):
                    if ttype == 'fam':
                        mask = sess.fam_place_cell_mask(chan=chan, mux=True)
                    else:
                        mask = sess.nov_place_cell_mask(chan=chan, mux=True)

                    pc_frac = float(mask.sum())/mask.shape[0]
                    
                    rows.append({
                        'mouse': mouse,
                        'day': day,
                        'channel': chan,
                        'ttype': ttype,
                        'frac': pc_frac,
                    })

    df = pd.DataFrame(rows)

    df['rank_frac'] = df['frac'].rank()

    result_dict = {}

    mdl = smf.mixedlm('rank_frac ~ C(day) * C(channel) * C(ttype)', df, groups=df['mouse'])
    model_result = mdl.fit()


    # run genotype shuffles
    shuff_coefs = np.array(joblib.Parallel(n_jobs=16)(joblib.delayed(run_model)(df.copy(),  
                                                                                'rank_frac',
                            ) for p in range(n_perms)))



    result_dict['rank_frac'] = {'true_model': model_result,
                                'genotype coef.': model_result.fe_params['C(channel)[T.channel_1]'],
                                'shuffle genotype coef.': shuff_coefs,
                                }
    
    return result_dict




if __name__ == "__main__":

    result_dict = run_pc_fracs()
    with open('/home/mplitt/shuffle_pkls/sparse_pc_fracs_mixedlm_shuffle.pkl', 'wb') as file:
        pickle.dump(result_dict, file)