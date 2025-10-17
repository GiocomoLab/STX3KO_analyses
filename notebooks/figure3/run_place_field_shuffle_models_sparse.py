import pathlib
import joblib
import numpy as np
import scipy as sp
import pandas as pd
import pickle


import statsmodels.formula.api as smf



import STX3KO_analyses as stx
from STX3KO_analyses import utilities as u
from STX3KO_analyses.run_sig_fields_spatial_perm import filter_fields

sparse_mice = stx.ymaze_sess_deets.sparse_mice

rng = np.random.default_rng()

def run_model(df_perm, stat_column):
    # Permute genotype within each mouse if within-mouse design
    df_perm["chan"] = df_perm.groupby("mouse", observed=True)["chan"].transform(np.random.permutation)

    md_perm = smf.mixedlm(f"{stat_column} ~ chan * day * nov", df_perm, groups=df_perm["mouse"])
    try:
        res_perm = md_perm.fit()
        return res_perm.fe_params['chan[T.channel_1]']
    except:
        return np.nan
    

def run_n_fields_sparse(n_perms=1000):



    with open('/home/mplitt/shuffle_pkls/sparse_place_field_spatial_shuffle_F_dff.pkl','rb') as file:
        shuff_results = pickle.load(file)


    df = {'chan': [], 'mouse': [], 'day': [], 'nov': [], 'n_fields': []}
    for mouse in sparse_mice:
        for day in range(6):
            if (mouse == 'SparseKO_09') and (day==2):
                continue

            for nov in ('fam','nov'):
                for chan in ('channel_0', 'channel_1'):
                
                    rising_edges = shuff_results[mouse][chan][day][nov]['rising_edges']
                    falling_edges = shuff_results[mouse][chan][day][nov]['falling_edges']
                    widths = shuff_results[mouse][chan][day][nov]['field_widths']
                    
                    mask = (widths>1) * (widths<25)  #* (rising_edges[:,1]>0) * (falling_edges[:,1]<29)
                

                    n_fields = np.bincount(rising_edges[mask,0])
                    
            
                    n_fields = n_fields[n_fields>0]

                    for n in n_fields:
                        df['mouse'].append(mouse)
                        df['day'].append(day)
                        df['nov'].append(nov)
                        df['chan'].append(chan)
                        df['n_fields'].append(n)

    df = pd.DataFrame.from_dict(df)


    df['chan'] = df["chan"].astype("category")
    df["mouse"] = df["mouse"].astype("category")
    df["day"] = df["day"].astype("category")
    df["nov"] = df["nov"].astype("category")
    df["rank_n_fields"] = df["n_fields"].rank()


    result_dict = {}
    for stat_column in ('n_fields', 'rank_n_fields'):

        # Fit a mixed model on ranks
        model = smf.mixedlm(f"{stat_column} ~ chan * day * nov", df, groups=df["mouse"])
        model_result = model.fit()

        # run genotype shuffles
        shuff_coefs = np.array(joblib.Parallel(n_jobs=16)(joblib.delayed(run_model)(df.copy(),  
                                                                                    stat_column,
                                ) for p in range(n_perms)))



        result_dict[stat_column] = {'true_model': model_result,
                                    'genotype coef.': model_result.fe_params['chan[T.channel_1]'],
                                    'shuffle genotype coef.': shuff_coefs,
                                    }
        
    return result_dict


def run_width_sparse(n_perms=1000):



    with open('/home/mplitt/shuffle_pkls/sparse_place_field_spatial_shuffle_F_dff.pkl','rb') as file:
        shuff_results = pickle.load(file)


    df = {'chan': [], 'mouse': [], 'day': [], 'nov': [], 'width': []}
    for mouse in sparse_mice:
        for day in range(6):
            
            if (mouse == 'SparseKO_09') and (day==2):
                continue

            for nov in ('fam','nov'):
                for chan in ('channel_0', 'channel_1'):
                    rising_edges = shuff_results[mouse][chan][day][nov]['rising_edges']
                    falling_edges = shuff_results[mouse][chan][day][nov]['falling_edges']
                    widths = shuff_results[mouse][chan][day][nov]['field_widths']
                    
                    mask = (widths>1) * (widths<25) * (rising_edges[:,1]>0) * (falling_edges[:,1]<29)
                    widths = widths[mask]

                    for w in widths:
                    
                        df['mouse'].append(mouse)
                        df['day'].append(day)
                        df['nov'].append(nov)
                        df['chan'].append(chan)
                        df['width'].append(10*w)

    df = pd.DataFrame.from_dict(df)

    df['chan'] = df["chan"].astype("category")
    df["mouse"] = df["mouse"].astype("category")
    df["day"] = df["day"].astype("category")
    df["nov"] = df["nov"].astype("category")
    df["rank_width"] = df["width"].rank()


    result_dict = {}
    for stat_column in ('width', 'rank_width'):

        # Fit a mixed model on ranks
        model = smf.mixedlm(f"{stat_column} ~ chan * day * nov", df, groups=df["mouse"])
        model_result = model.fit()

        # run genotype shuffles
        shuff_coefs = np.array(joblib.Parallel(n_jobs=16)(joblib.delayed(run_model)(df.copy(),  
                                                                                    stat_column,
                                ) for p in range(n_perms)))



        result_dict[stat_column] = {'true_model': model_result,
                                    'genotype coef.': model_result.fe_params['chan[T.channel_1]'],
                                    'shuffle genotype coef.': shuff_coefs,
                                    }
        
    return result_dict

if __name__ == "__main__":

    result_dict = run_n_fields_sparse(n_perms=1000)
    with open('/home/mplitt/shuffle_pkls/sparse_n_fields_mixedlm_shuffle.pkl', 'wb') as file:
        pickle.dump(result_dict, file)

    result_dict = run_width_sparse(n_perms=1000)
    with open('/home/mplitt/shuffle_pkls/sparse_width_mixedlm_shuffle.pkl', 'wb') as file:
        pickle.dump(result_dict, file)





