import os
import joblib
import pickle
import dill

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from matplotlib import gridspec

import pandas as pd
from pingouin import mixed_anova, anova, pairwise_tukey, pairwise_tests
import pingouin as pg
from statsmodels.regression.mixed_linear_model import MixedLM

import TwoPUtils as tpu
import STX3KO_analyses as stx
from STX3KO_analyses import utilities as u




ko_mice = stx.ymaze_sess_deets.ko_mice

ctrl_mice = stx.ymaze_sess_deets.ctrl_mice



class HSE:
    
    def __init__(self, sess, tskey='F_dff_th'):
        
        if isinstance(sess.scan_info,list):
            self.frame_rate = sess.scan_info[0]['frame_rate']
        else:
            self.frame_rate = sess.scan_info['frame_rate']
        
        
        
        
        self.spks = sess.timeseries[tskey]
        if '_th' in tskey:
            self.trial_mat = sess.trial_matrices[tskey[:-3]]
        else:
            self.trial_mat = sess.trial_matrices[tskey]
        self.speed = sess.timeseries['speed']
        self.lr = sess.timeseries['LR'].ravel()
        self.t = sess.timeseries['t'].ravel()
        self.reward = sess.timeseries['reward'].ravel()
        self.block_number = sess.timeseries['block_number'].ravel()
        
        self.novel_arm = sess.novel_arm
        self.trial_info = sess.trial_info
        self.trial_start_inds = sess.trial_start_inds
        self.teleport_inds = sess.teleport_inds
        
        
        # mask for post reward epochs
        self.post_reward_bool = None
        self.post_reward_epochs = None
        self._post_reward_mask(sess)
        
        
        # threshold spks timeseries to get "significant" transients
        if '_th' in tskey:
            self.thresh_spks = 0
        else:
            self.thresh_spks = .2#*np.nanmax(sess.timeseries[tskey], axis=-1, keepdims=True)
        self.spks_th = 1.*np.copy(self.spks)
        self.spks_th[(self.spks<self.thresh_spks)]=0
        # self.spks_th[(self.spks<.2)]=0
        
        
        
        # all activations during stopping, including periods when not near reward zone
        self.activation_mask = None
        # activations during reward consumption
        self.post_reward_activation_mask = None
        # all activations during running
        self.trial_activation_mask = None
        self.get_activations()
        
        boxcar = np.ones([1,4]) # 500 ms window
        self.post_reward_activation_boxcar = sp.signal.convolve(self.post_reward_activation_mask,boxcar,mode='same')
        self.pop_coactivity_boxcar = self.post_reward_activation_boxcar.sum(axis=0)
        
        self._rng = np.random.default_rng()
        
        self.coactivity_thresh = None
        self.shuff_results = None
        self.post_reward_coactive_mask = 0*self.post_reward_activation_boxcar
        self.pop_coactivity_th = None
        if self.post_reward_epochs.shape[0]>0:
            self.get_popact_thresh()
        
            self.post_reward_coactive_mask = np.copy(self.post_reward_activation_boxcar)
            self.post_reward_coactive_mask[:,self.pop_coactivity_boxcar<self.coactivity_thresh]=0

            self.pop_coactivity_th = np.zeros(self.pop_coactivity_boxcar.shape)
            self.pop_coactivity_th[self.pop_coactivity_boxcar>=self.coactivity_thresh]=1
        
        
        
        # active during trial and during reward consumption
#         self.post_reward_reactivation_mask = None
#         # active during trial but not during reward consumption
#         self.non_reactivation_mask = None
#         self.get_post_reward_reactivations()
        
        
    def _post_reward_mask(self, sess, max_post_reward_time = 10, max_post_reward_dist = 3):
        post_reward_bool = 0.*self.t.ravel()
        for start,stop in zip(self.trial_start_inds, self.teleport_inds):

            cum_reward = np.cumsum(self.reward.ravel()[start:stop])>0
            post_reward_inds = np.argwhere(cum_reward).ravel()
            if post_reward_inds.shape[0]>0:
                post_reward_bool[post_reward_inds[0]+start:start+np.minimum(post_reward_inds[0]+int(max_post_reward_time*self.frame_rate),stop-start)] = 1


        nov_post_rzone_mask = (sess.timeseries['LR']==sess.novel_arm)*(sess.timeseries['t']>sess.rzone_nov['tfront']+max_post_reward_dist)
        fam_post_rzone_mask = (sess.timeseries['LR']==-1*sess.novel_arm)*(sess.timeseries['t']>sess.rzone_fam['tfront']+max_post_reward_dist)
        speed_mask = self.speed.ravel()>=2
        
        post_reward_bool[nov_post_rzone_mask.ravel()]=0
        post_reward_bool[fam_post_rzone_mask.ravel()]=0
        post_reward_bool[speed_mask] = 0
        
        
        
        _post_reward_bool = np.zeros([post_reward_bool.shape[0]+1,])
        _post_reward_bool[1:] = post_reward_bool
        post_reward_starts = np.argwhere(_post_reward_bool[1:]>_post_reward_bool[:-1]).ravel()+1
        _post_reward_bool = np.zeros([post_reward_bool.shape[0]+1,])
        _post_reward_bool[:-1] = post_reward_bool
        post_reward_ends = np.argwhere(_post_reward_bool[:-1]>_post_reward_bool[1:]).ravel()-1
        assert post_reward_starts.shape[0]==post_reward_ends.shape[0], "post reward starts and ends screwed up"
        
        
        
        # filter for post reward periods greater than 750 sec
        epoch_mask = (post_reward_ends-post_reward_starts)/self.frame_rate>.75
        post_reward_starts, post_reward_ends = post_reward_starts[epoch_mask], post_reward_ends[epoch_mask]
        
        post_reward_epochs = np.zeros([post_reward_starts.shape[0],2])
        post_reward_epochs[:, 0] = post_reward_starts
        post_reward_epochs[:, 1] = post_reward_ends
        self.post_reward_epochs = post_reward_epochs
        # print(self.post_reward_epochs.shape)
        
        
        
        self.post_reward_bool = np.zeros(post_reward_bool.shape)
        for start, stop in zip(post_reward_starts, post_reward_ends):
            self.post_reward_bool[start:stop] = 1
            
        self.post_reward_starts = [int(ind) for ind in post_reward_starts]
        self.post_reward_ends = [int(ind) for ind in post_reward_ends]
        
        
    def get_activations(self):
    
        spks_ledge = 1*(np.diff(1.*(self.spks_th>0),prepend=0,axis=-1)>0)

        self.activation_mask = 1*(spks_ledge>0)*(self.speed<2)
        self.post_reward_activation_mask = 1*(spks_ledge>0)*(self.speed<2)*(self.post_reward_bool[np.newaxis,:])
        self.trial_activation_mask = 1*(spks_ledge>0)*(1-self.post_reward_bool[np.newaxis,:])
        
    @staticmethod
    def shuff_act(post_reward_activation_mask, post_reward_epochs, rng, post_reward_bool, bin_edges=np.arange(51)):
        
        act = 1*np.copy(post_reward_activation_mask)
        for ep_ind in range(post_reward_epochs.shape[0]):
            start, stop = [int(s) for s in post_reward_epochs[ep_ind,:]]

            _act = act[:,start:stop]
            shifts = rng.integers(0,stop-start, (_act.shape[0],))

            for cell in range(_act.shape[0]):
                _act[cell,:] = np.roll(_act[cell,:], shifts[cell])
            act[:,start:stop]=_act

        boxcar = np.ones([4,]) # 250 ms window - results in 500 ms coincidence
        pop_act_boxcar = sp.signal.convolve(act.sum(axis=0), boxcar, mode='same')
        return np.histogram(pop_act_boxcar[post_reward_bool>0], bin_edges)[0]

    def get_popact_thresh(self, nperms = 1000, p_thresh=.95):
        self.bin_edges = np.arange(100)
        
        
        results = np.array(joblib.Parallel(n_jobs=-1)(joblib.delayed(self.shuff_act)(self.post_reward_activation_mask, 
                                                                                     self.post_reward_epochs, 
                                                                                     self._rng,
                                                                                     self.post_reward_bool,
                                                                                     bin_edges=self.bin_edges) for i in range(nperms)))
        # with Pool() as p
        #     results = np.array(p.starmap(self.shuff_act, range(nperms), ))
            
        shuff_hist = results.sum(axis=0)
        shuff_hist = shuff_hist/shuff_hist.sum()
        self.coactivity_thresh = self.bin_edges[:-1][np.argwhere(np.cumsum(shuff_hist)>p_thresh).ravel()[0]]
        self.shuff_results = results
        
        
        
    

if __name__ == '__main__':
    # hse_dict = {}
    tskey='F_dff'
    basedir = f'/home/mplitt/YMazeReplayPkls_{tskey}/'
    os.makedirs(basedir, exist_ok=True)
    for cond_key, mice in zip(('ctrl', 'cre'),(ctrl_mice, ko_mice)):
        # hse_dict[cond_key] = {}
        for day in range(6):
            # hse_dict[cond_key][day] = {}
            for mouse in mice:
                print(mouse,day)
                filename = basedir + f'{mouse}_day{day}.pkl'
                if True: #not os.path.exists(filename):
                    hse = HSE(u.load_single_day(mouse,day,verbose=False),tskey=tskey)

                    with open(filename, 'wb') as file:
                        pickle.dump(hse,file)
                    
                # hse_dict[cond_key][day][mouse]=filename
                                                