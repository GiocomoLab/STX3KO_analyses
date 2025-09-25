import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
import pandas as pd

import STX3KO_analyses as stx
from STX3KO_analyses import utilities as u

ko_mice = stx.ymaze_sess_deets.ko_mice
ctrl_mice = stx.ymaze_sess_deets.ctrl_mice
sparse_mice = stx.ymaze_sess_deets.sparse_mice


def shifts_to_matrix(shifts, max_trial):

    argmax_mat = np.zeros([len(shifts), max_trial])
    for i, shift in enumerate(shifts):
        argmax_mat[i,:] = shift[:max_trial]
    return argmax_mat

def get_field_shifts(trial_mat, fields_dict, ttype):

    shift, ind = [], []
    for i, (rising_edge, falling_edge, width) in enumerate(zip(fields_dict['rising_edges'],
                                                        fields_dict['falling_edges'],
                                                     fields_dict['field_widths'])):
        
        # if ttype =='novel' and falling_edge[1]<10:
        #     continue
           

        if (width>5) and (width<25):

            cell = rising_edge[0]
            tmat = trial_mat[:,:,cell]
            
            tmat[np.isnan(tmat)]=0
        
            fieldmat = tmat[:,rising_edge[1]:falling_edge[1]]
            
            # greater than 20% of max
            fieldmat_th = 1.*((fieldmat>=.2*np.nanmax(fieldmat)).sum(axis=1)>0)

            # cross threshold and active for 3 of 5 laps
            formation_lapvec = fieldmat_th[:-4] * (sp.signal.convolve(fieldmat_th,np.ones([5,]), mode= 'valid')>=3)
            formation_lap_inds = np.nonzero(formation_lapvec)[0]

            num_nonzero = formation_lap_inds.shape[0]
            
            if num_nonzero>0:
                formation_lap = formation_lap_inds[0]
                if formation_lap<(fieldmat.shape[0]-4):

                    # active on 20% of trials after formation lap
                    activity_bool = fieldmat_th[formation_lap:].mean()>.2

                    # if activity_bool
                    if activity_bool:

                        # argmax = np.argmax(fieldmat[formation_lap:,:],axis=1)
                        argmax = np.array([sp.ndimage.center_of_mass(fieldmat[l,:]) for l in range(formation_lap,fieldmat.shape[0])]).ravel()
                        # mu_max = np.argmax(fieldmat[formation_lap+1:,:].mean(axis=0))
                        shift.append(argmax-argmax.mean())
                        ind.append(i)
    return shift, ind

