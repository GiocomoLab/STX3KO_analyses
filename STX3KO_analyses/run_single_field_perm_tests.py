import os
import joblib
import pickle

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from matplotlib import gridspec

import pandas as pd
from pingouin import mixed_anova, anova, pairwise_tukey, pairwise_tests
import pingouin as pg
from statsmodels.regression.mixed_linear_model import MixedLM
import statsmodels.formula.api as smf

import TwoPUtils as tpu
import STX3KO_analyses as stx
from STX3KO_analyses import utilities as u
from STX3KO_analyses import utilities_ES as u_es
from STX3KO_analyses import run_sig_fields_spatial_perm as spatial_perms


plt.rcParams['pdf.fonttype']=42
ko_mice = stx.ymaze_sess_deets.ko_mice
ctrl_mice = stx.ymaze_sess_deets.ctrl_mice
sparse_mice = stx.ymaze_sess_deets.sparse_mice


# def run_shuffle(sess, rng, last_block_only=False):
def run_shuffle(spks, trial_starts, teleports, t, lr, novel_arm,rng):
    # spks = sess.timeseries['F_dff']
    spks_shuff = np.zeros(spks.shape)
    
    
    for trial, (start, stop) in enumerate(zip(trial_starts, teleports)):
        _spks = 1*spks[:,start:stop]    
        _spks = np.roll(_spks,rng.integers(0,stop-start),axis=-1)
        spks_shuff[:,start:stop] = 1*_spks
        
    
    tmat = tpu.spatial_analyses.trial_matrix(spks_shuff.T,t, trial_starts, teleports, min_pos=13, max_pos=43, bin_size=1, mat_only=True)
    
    nov_mean = np.nanmean(tmat[lr==novel_arm, :, :], axis=0)
    fam_mean = np.nanmean(tmat[lr==-1*novel_arm, :, :], axis=0)
    return nov_mean, fam_mean


def run_dense_mice(tskey):
    rng = np.random.default_rng()
    shuff_results = {}
    for mice in (ctrl_mice, ko_mice):
        for mouse in mice:
            shuff_results[mouse]={}
            for day in range(6):
                shuff_results[mouse][day] = {}

                sess = u.load_single_day(mouse, day,pkl_basedir='/home/mplitt/YMazeSessPkls')
                spks = sess.timeseries[tskey]
                trial_mat = sess.trial_matrices[tskey]
                ##
                # spks = sess.timeseries['F_dff']-.2
                # spks[spks<0]=0
                ##
                trial_starts, teleports = sess.trial_start_inds, sess.teleport_inds
                t = sess.timeseries['t'].ravel()
                lr = sess.trial_info['LR']
                novel_arm = sess.novel_arm
                
                shuff_trial_mat = np.array(joblib.Parallel(n_jobs=-1)(joblib.delayed(run_shuffle)(spks, trial_starts, teleports, t, lr, novel_arm, rng) for i in range(1000)))
                print(shuff_trial_mat.shape)
                fam_shuff_thresh = np.nanpercentile(shuff_trial_mat[:, 1, :, :], 99, axis=0)
                fam_field_mask = np.nanmean(trial_mat[lr!=novel_arm,:,:])>fam_shuff_thresh
                rising_edges, falling_edges, field_widths = spatial_perms.get_field_stats(fam_field_mask)
                shuff_results[mouse][day]['fam'] ={'field_mask': fam_field_mask, 
                                                'rising_edges': rising_edges, 
                                                'falling_edges': falling_edges, 
                                                'field_widths': field_widths,
                                                }
                
                nov_shuff_thresh = np.nanpercentile(shuff_trial_mat[:, 0, :, :], 99, axis=0)
                nov_field_mask = np.nanmean(trial_mat[lr==novel_arm,:,:])>nov_shuff_thresh
                rising_edges, falling_edges, field_widths = spatial_perms.get_field_stats(nov_field_mask)
                shuff_results[mouse][day]['nov'] ={'field_mask': nov_field_mask, 
                                                'rising_edges': rising_edges, 
                                                'falling_edges': falling_edges, 
                                                'field_widths': field_widths,
                                                }
                
    with open(f'/home/mplitt/shuffle_pkls/place_field_shuff_results_{tskey}.pkl','wb') as file:
            pickle.dump(shuff_results,file)
            
def run_sparse_mice(tskey):
    rng = np.random.default_rng()
    shuff_results = {}
    for mouse in sparse_mice:
        shuff_results[mouse]={}
        
        for day in range(6):
            if mouse == 'SparseKO_09' and day ==2:
             continue

            shuff_results[mouse][day] = {}
            
            sess = u.load_single_day(mouse, day, pkl_basedir='/home/mplitt/YMazeSessPkls')
            
            for chan in ('channel_0', 'channel_1'):
                if chan=='channel_0':
                    t = sess.vr_data_chan0['t']
                else:
                    t = sess.vr_data_chan1['t']
                    
                shuff_results[mouse][day][chan] = {}
            
                spks = sess.timeseries[f'{chan}_{tskey}']
                trial_mat = sess.trial_matrices[f'{chan}_{tskey}']
                ##
                ##
                trial_starts, teleports = sess.trial_starts[chan], sess.trial_ends[chan]
                
                lr = sess.trial_info['LR']
                novel_arm = sess.novel_arm
            
                shuff_trial_mat = np.array(joblib.Parallel(n_jobs=-1)(joblib.delayed(run_shuffle)(spks, trial_starts, teleports, t, lr, novel_arm, rng) for i in range(1000)))
                print(shuff_trial_mat.shape)
                fam_shuff_thresh = np.nanpercentile(shuff_trial_mat[:, 1, :, :], 99, axis=0)
                fam_field_mask = np.nanmean(trial_mat[lr!=novel_arm,:,:])>fam_shuff_thresh
                rising_edges, falling_edges, field_widths = spatial_perms.get_field_stats(fam_field_mask)
                shuff_results[mouse][day][chan]['fam'] ={'field_mask': fam_field_mask, 
                                                'rising_edges': rising_edges, 
                                                'falling_edges': falling_edges, 
                                                'field_widths': field_widths,
                                                }
                
                nov_shuff_thresh = np.nanpercentile(shuff_trial_mat[:, 0, :, :], 99, axis=0)
                nov_field_mask = np.nanmean(trial_mat[lr==novel_arm,:,:])>nov_shuff_thresh
                rising_edges, falling_edges, field_widths = spatial_perms.get_field_stats(nov_field_mask)
                shuff_results[mouse][day][chan]['nov'] ={'field_mask': nov_field_mask, 
                                                'rising_edges': rising_edges, 
                                                'falling_edges': falling_edges, 
                                                'field_widths': field_widths,
                                                }
                

                # fam_shuff_thresh = np.nanpercentile(shuff_trial_mat[:, 1, :, :], 99, axis=0)
                # nov_shuff_thresh = np.nanpercentile(shuff_trial_mat[:, 0, :, :], 99, axis=0)
                # shuff_results[mouse][day][chan]['fam']=fam_shuff_thresh
                # shuff_results[mouse][day][chan]['nov']=nov_shuff_thresh
            

    with open(f'/home/mplitt/shuffle_pkls/sparse_place_field_shuff_results_{tskey}.pkl','wb') as file:
            pickle.dump(shuff_results,file)

if __name__ == "__main__":
    tskey = 'F_dff'
    run_dense_mice(tskey)
    run_sparse_mice(tskey)