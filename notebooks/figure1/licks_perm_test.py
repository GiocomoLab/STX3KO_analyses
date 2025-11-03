import pathlib
import joblib
import itertools
import numpy as np
import scipy as sp
import pandas as pd
import pickle

import statsmodels.formula.api as smf



import STX3KO_analyses as stx



ko_mice = stx.ymaze_sess_deets.ko_mice
ctrl_mice = stx.ymaze_sess_deets.ctrl_mice



def run_model(df_perm, all_mice, ctrl_mice, stat_column):

    genotype_map = {}
    for mouse in all_mice:
        genotype_map[mouse] = "ko"
    for mouse in ctrl_mice:
        genotype_map[mouse] = "ctrl"
    
    
    df_perm["cond"] = df_perm["mouse"].map(genotype_map)


    md_perm = smf.mixedlm(f"{stat_column} ~ C(cond)*C(ttype) + C(day)", df_perm, groups=df_perm["mouse"])

    try:
        res_perm = md_perm.fit()
        return res_perm.fe_params['C(cond)[T.ko]']
    except:
        return np.nan  # skip failed fits (rare)





def run_licks():
    rows = []

    for (cond, mice) in zip(('ctrl', 'ko'), (ctrl_mice, ko_mice)):
        for day in range(6):
            print(day)
            for mouse in mice:
                print(mouse)
                sess = stx.utilities.load_vr_day(mouse, day, trial_mat_keys = ('licks_sum',),verbose = False, pkldir = '/mnt/BigDisk/YMaze_VR_Pkls/')

                bin_edges = sess.trial_matrices['bin_edges']
                right_rzone_front = np.argwhere((sess.rzone_late['tfront']<=bin_edges[1:])*(sess.rzone_late['tfront']>=bin_edges[:-1]))[0][0]
                left_rzone_front = np.argwhere((sess.rzone_early['tfront']<=bin_edges[1:])*(sess.rzone_early['tfront']>=bin_edges[:-1]))[0][0]

                if sess.novel_arm==-1:
                    rzone_nov = left_rzone_front
                    rzone_fam = right_rzone_front
                else:
                    rzone_nov = right_rzone_front
                    rzone_fam = left_rzone_front

                for ttype in ('fam', 'nov'):
                    
                    if ttype == 'nov':
                        trial_mask = sess.trial_info["LR"] == sess.novel_arm
                        rzone = rzone_nov
                    else:
                        trial_mask = sess.trial_info["LR"] != sess.novel_arm
                        rzone = rzone_fam

                    
                    licks = sess.trial_matrices['licks_sum'][trial_mask,:]
                    rzone_licks = np.nanmean(licks[:,rzone-5:rzone+1], axis=0).sum()
                    

                    rows.append({'cond': cond,
                                'mouse': mouse,
                                'day': day,
                                'ttype': ttype,
                                'rzone_licks': rzone_licks,
                                })
    df = pd.DataFrame(rows)
    df['rzone_licks_rank'] = df['rzone_licks'].rank()
                
    result_dict = {}
    for stat_column in ('rzone_licks', 'rzone_licks_rank'):
        

        # Fit a mixed model on ranks
        model = smf.mixedlm(f"{stat_column} ~ C(cond)*C(ttype) + C(day)", df, groups=df["mouse"])
        model_result = model.fit()

        # run genotype shuffles
        shuff_coefs = np.array(joblib.Parallel(n_jobs=16)(joblib.delayed(run_model)(df.copy(), 
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

    result_dict = run_licks()
    with open('/home/mplitt/shuffle_pkls/licks_sum_mixedlm_shuffle.pkl', 'wb') as file:
        pickle.dump(result_dict, file)