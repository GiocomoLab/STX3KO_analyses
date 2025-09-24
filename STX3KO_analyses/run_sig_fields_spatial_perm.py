import numpy as np
import joblib
import pickle


import pandas as pd

import TwoPUtils as tpu
import STX3KO_analyses as stx
from STX3KO_analyses import utilities as u
from STX3KO_analyses import utilities_ES as u_es



ko_mice = stx.ymaze_sess_deets.ko_mice
ctrl_mice = stx.ymaze_sess_deets.ctrl_mice
sparse_mice = stx.ymaze_sess_deets.sparse_mice


rng = np.random.default_rng()




def get_field_stats(field_mask):

    _field_mask = np.zeros([field_mask.shape[0]+2, field_mask.shape[1]])
    _field_mask[1:-1,:]=field_mask

    
    rising_edges, falling_edges = np.argwhere((_field_mask[1:,:]>_field_mask[:-1,:]).T), np.argwhere((_field_mask[:-1,:]>_field_mask[1:,:]).T)
    field_widths = falling_edges[:,1]-rising_edges[:,1]
    

    
    num_fields = np.bincount(rising_edges[:,0])
    

    
    return rising_edges, falling_edges, field_widths, num_fields
    
    

def field_masks(mouse,day,tskey, n_perms = 1000, pcnt = 95 ):
    
    sess = u_es.load_single_day(mouse,day, pkl_basedir='/home/mplitt/YMazeSessPkls')
    
    def _run_fields(nov):
        if nov:
            trial_mask = sess.trial_info['LR']==sess.novel_arm
        else:
            trial_mask = sess.trial_info['LR']== -1*sess.novel_arm
        shuff_mat = np.zeros([1000, *sess.trial_matrices[tskey].shape[1:]])
        trial_mat = sess.trial_matrices[tskey][trial_mask,:,:]
        n_trials = trial_mat.shape[0]
        
        shuffs = rng.integers(trial_mat.shape[1], size = [n_perms, trial_mat.shape[0]])
        
        # calculate shuffles
        _tmat = 0*trial_mat
        for perm in range(n_perms):
            for trial in range(n_trials):
                _tmat[trial,:,:] = np.roll(trial_mat[trial,:,:], shuffs[perm, trial])
            shuff_mat[perm,:,:] = np.nanmean(_tmat, axis=0)
            
        thresh = np.nanpercentile(shuff_mat,pcnt, axis=0)
        field_mask = 1*(np.nanmean(trial_mat,axis=0)>thresh)
        rising_edges, falling_edges, field_widths, num_fields = get_field_stats(field_mask)
        return {'field_mask': field_mask, 
                'rising_edges': rising_edges, 
                'falling_edges': falling_edges, 
                'field_widths': field_widths,
                'num_fields': num_fields}
    
    return {'fam': _run_fields(False), 'nov': _run_fields(True)}


def run_dense_mice():
    shuffle_results = {}
    for mouse in ctrl_mice+ko_mice:
        print(mouse)
        
        days = np.arange(6)
        results_list = joblib.Parallel(n_jobs=int(days.shape[0]))(joblib.delayed(field_masks)(mouse, day, 'F_dff') for day in days)
        shuffle_results[mouse] = dict(zip(days,results_list))

    with open('/home/mplitt/shuffle_pkls/dense_place_field_spatial_shuffle_F_dff.pkl','wb') as file:
        pickle.dump(shuffle_results, file)

def run_sparse_mice():

    shuffle_results = {}
    for mouse in sparse_mice:
        print(mouse)
        if (mouse == 'SparseKO_09'):
            days = np.array([0,1,3,4,5])
        else: 
            days = np.arange(6)

        shuffle_results[mouse] = {}

        for chan in ('channel_0', 'channel_1'):
            tskey = f'{chan}_F_dff'
            results_list = joblib.Parallel(n_jobs=int(days.shape[0]))(joblib.delayed(field_masks)(mouse, day, tskey) for day in days)
            shuffle_results[mouse][chan] = dict(zip([int(d) for d in days],results_list))

    with open('/home/mplitt/shuffle_pkls/sparse_place_field_spatial_shuffle_F_dff.pkl','wb') as file:
        pickle.dump(shuffle_results, file)

if __name__ == "__main__":

    # run_dense_mice()
    run_sparse_mice()
        