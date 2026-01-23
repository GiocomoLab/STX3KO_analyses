import numpy as np
import joblib
import pickle
import scipy as sp


import pandas as pd

import TwoPUtils as tpu
import STX3KO_analyses as stx
from STX3KO_analyses import utilities as u




ko_mice = stx.ymaze_sess_deets.ko_mice
ctrl_mice = stx.ymaze_sess_deets.ctrl_mice
sparse_mice = stx.ymaze_sess_deets.sparse_mice


rng = np.random.default_rng()




def get_field_stats(field_mask):

    _field_mask = np.zeros([field_mask.shape[0]+2, field_mask.shape[1]])
    _field_mask[1:-1,:]=field_mask

    
    rising_edges, falling_edges = np.argwhere((_field_mask[1:,:]>_field_mask[:-1,:]).T), np.argwhere((_field_mask[:-1,:]>_field_mask[1:,:]).T)
    field_widths = falling_edges[:,1]-rising_edges[:,1]
    

    
    # num_fields = np.bincount(rising_edges[:,0])
    

    
    return rising_edges, falling_edges, field_widths#, num_fields
    
def filter_fields(fields_dict, trial_mat):
        
      
        field_dict_filt = {'field_mask': fields_dict['field_mask'],
                           'rising_edges': [],
                           'falling_edges': [],
                           'field_widths': [],
                           'formation_laps': []}
        for i, (rising_edge, falling_edge, width) in enumerate(zip(fields_dict['rising_edges'],
                                                    fields_dict['falling_edges'],
                                                     fields_dict['field_widths'])):
            if width>1 and width<20:

                cell = rising_edge[0]
                print(cell)
                tmat = trial_mat[:,:, cell]

                fieldmat = tmat[:,rising_edge[1]:falling_edge[1]]
                fieldmat_th = 1.*((fieldmat>=.1*np.nanmax(fieldmat)).sum(axis=1)>0)
           
                # cross threshold and active for 3 of 5 laps
                formation_lapvec = fieldmat_th[:-4] * (sp.signal.convolve(fieldmat_th,np.ones([5,]), mode= 'valid')>=3)
                formation_lap_inds = np.nonzero(formation_lapvec)[0]

                num_nonzero = formation_lap_inds.shape[0]
            
                if num_nonzero>0:
                    flap = formation_lap_inds[0]
                    if flap<(fieldmat.shape[0]-4): # and formation_lap>0:

                        # active on 50% of trials after formation lap
                        activity_bool = fieldmat_th[flap:].mean()>.2

                        # if activity_bool
                        if activity_bool:
                            field_dict_filt['rising_edges'].append(rising_edge)
                            field_dict_filt['falling_edges'].append(falling_edge)
                            field_dict_filt['field_widths'].append(width)
                            field_dict_filt['formation_laps'].append(flap)
                          

            return field_dict_filt


def field_masks(mouse,day,tskey, n_perms = 1000, pcnt = 95 ):
    
    sess = u.load_single_day(mouse,day, pkl_basedir='/home/mplitt/YMazeSessPkls')
    
    def _run_fields(nov):
        if nov:
            trial_mask = sess.trial_info['LR']==sess.novel_arm
        else:
            trial_mask = sess.trial_info['LR']== -1*sess.novel_arm
        shuff_mat = np.zeros([n_perms, *sess.trial_matrices[tskey].shape[1:]])
        trial_mat = sess.trial_matrices[tskey][trial_mask,:,:]
        trial_mat[np.isnan(trial_mat)]=0
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
        rising_edges, falling_edges, field_widths = get_field_stats(field_mask)
        return {'field_mask': field_mask, 
                'rising_edges': rising_edges, 
                'falling_edges': falling_edges, 
                'field_widths': field_widths,
                }
    
    

    
    return {'fam': _run_fields(False), 'nov': _run_fields(True)}



def run_sparse_mice(tau=0.7):

    shuffle_results = {}
    for mouse in sparse_mice:
        print(mouse)
        if (mouse == 'SparseKO_09'):
            days = np.array([0,1,3,4,5])
        else: 
            days = np.arange(6)

        shuffle_results[mouse] = {}

        for chan in ('channel_0', 'channel_1'):
            if chan == 'channel_0':
                tskey = f'{chan}_spks'
            else:
                if tau==0.7:
                    tskey = f'{chan}_spks'
                else:

                    tskey = f'{chan}_spks_tau{tau:.1f}'
            results_list = joblib.Parallel(n_jobs=int(days.shape[0]))(joblib.delayed(field_masks)(mouse, day, tskey) for day in days)
            shuffle_results[mouse][chan] = dict(zip([int(d) for d in days],results_list))

    with open(f'/home/mplitt/shuffle_pkls/sparse_place_field_spatial_shuffle_spks_tau{tau:.1f}.pkl','wb') as file:
        pickle.dump(shuffle_results, file)

if __name__ == "__main__":

    run_sparse_mice(tau=1.5)
    run_sparse_mice(tau=1.2)
    run_sparse_mice(tau=1.0)
    run_sparse_mice(tau=0.7)
        