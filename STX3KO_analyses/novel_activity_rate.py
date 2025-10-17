from . import session
from . import utilities as u 


from . import ymaze_sess_deets

from matplotlib import pyplot as plt
from matplotlib import gridspec

import scipy as sp
import numpy as np
import pandas as pd


from pingouin import mixed_anova, pairwise_ttests

ctrl_mice =ymaze_sess_deets.ctrl_mice
ko_mice = ymaze_sess_deets.ko_mice
sparse_mice = ymaze_sess_deets.sparse_mice


class CellCOV:

    def __init__(self, 
                 mouse_dict={'ctrl': ctrl_mice, 'ko':ko_mice}, 
                 days=np.arange(6), ts_key='F_dff', block = 5):
        '''
        
        '''

        self.mouse_dict = mouse_dict
        self.days = days
        self.ts_key = ts_key
        self.block = block

        self.covs = {}
        self.get_cov()


    def get_activity(self):

        for cond, mice, in self.mouse_dict.items():
            self.activity_rates[cond] = {}
            for mouse in mice:
                print(mouse)
                self.activity_rates[cond][mouse] = {}
                for day in self.days:

                    d = {}
                 
                    sess = u.load_single_day(mouse, day,verbose=False)


    



class BlockTransitionActivityRate:

    def __init__(self, 
                 mouse_dict={'ctrl': ctrl_mice, 'ko':ko_mice}, 
                 days=np.arange(6), ts_key='F_dff', block = 5):
        '''
        
        '''

        self.mouse_dict = mouse_dict
        self.days = days
        self.ts_key = ts_key
        self.block = block

        self.activity_rates = {}
        self.get_activity()


    def get_activity(self):



        for cond, mice, in self.mouse_dict.items():
            self.activity_rates[cond] = {}
            for mouse in mice:
                print(mouse)
                self.activity_rates[cond][mouse] = {}
                for day in self.days:

                    d = {}
                 
                    sess = u.load_single_day(mouse, day,verbose=False)


                    block_start_trial_num = np.argwhere(sess.trial_info['block_number']==self.block)[0][0]


                    baseline_trials = slice(block_start_trial_num-10, block_start_trial_num)

                    block_mask = sess.trial_info['block_number']==self.block
                    block_fam_mask = block_mask * (sess.trial_info['LR']==-1*sess.novel_arm)
                    block_nov_mask = block_mask * (sess.trial_info['LR']==sess.novel_arm)

                    # average over trials and positions
                    d['baseline_avg_act'] = np.nanmean(np.nanmean(sess.trial_matrices[self.ts_key][baseline_trials,:,:],
                                                      axis=0, keepdims=True), axis=1, keepdims=True)
                    
                    d['baseline_act'] = sess.trial_matrices[self.ts_key][baseline_trials,:, :]
                    d['block_act'] = sess.trial_matrices[self.ts_key][block_mask, :, :]
                    d['block_fam_act'] = sess.trial_matrices[self.ts_key][block_fam_mask, :, :]
                    d['block_nov_act'] = sess.trial_matrices[self.ts_key][block_nov_mask, :, :]



                    self.activity_rates[cond][mouse][day] = d


    def plot_single_session(self, cond, mouse, day, 
                            baseline_corr = True,
                            vmin=None, vmax=None, cmap='magma'):
        data = self.activity_rates[cond][mouse][day]
        
        fig = plt.figure()
        gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, .1], height_ratios=[1, 1], wspace=.5)
        ax1 = fig.add_subplot(gs[:, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 1])
        cbar_ax = fig.add_subplot(gs[0,2])



        if baseline_corr:

            b = data['baseline_avg_act']

            
            comb = np.nanmean(np.concatenate((data['baseline_act'], data['block_act']))/b,axis=-1)
            fam = np.nanmean(data['block_fam_act']/b, axis=-1)
            nov = np.nanmean(data['block_nov_act']/b, axis=-1)

        else:
            comb = np.nanmean(np.concatenate((data['baseline_act'], data['block_act'])),axis=-1)
            fam = np.nanmean(data['block_fam_act'], axis=-1)
            nov = np.nanmean(data['block_nov_act'], axis=-1)

        if vmin is None:
            vmin = np.nanpercentile(comb,1)

        if vmax is None:
            vmax = np.nanpercentile(comb, 99)

        kwargs = {'vmin': vmin,
                'vmax': vmax,
                'cmap': cmap}

        h = ax1.imshow(comb, **kwargs)

        
        h = ax2.imshow(fam, **kwargs)
        ax2.set_title('familiar')

        
        h = ax3.imshow(nov, **kwargs)
        ax3.set_title('novel')

        fig.suptitle(f"mouse: {mouse}, day: {day}")


class BlockTransitionActivityRate_Sparse:

    def __init__(self, 
                 mice = sparse_mice, 
                 days=np.arange(6), ts_key='F_dff', block = 5):
        '''
        
        '''

        self.mice = sparse_mice
        self.days = days
        self.ts_key = ts_key
        self.block = block

        self.activity_rates = {}
        self.get_activity()


    def get_activity(self):



        
        for mouse in self.mice:
            print(mouse)
            self.activity_rates[mouse] = {}
            for day in self.days:

                self.activity_rates[mouse]
                
                sess = u.load_single_day(mouse, day,verbose=False)


                block_start_trial_num = np.argwhere(sess.trial_info['block_number']==self.block)[0][0]


                baseline_trials = slice(block_start_trial_num-10, block_start_trial_num)

                block_mask = sess.trial_info['block_number']==self.block
                block_fam_mask = block_mask * (sess.trial_info['LR']==-1*sess.novel_arm)
                block_nov_mask = block_mask * (sess.trial_info['LR']==sess.novel_arm)

                # average over trials and positions
                d['baseline_avg_act'] = np.nanmean(np.nanmean(sess.trial_matrices[self.ts_key][baseline_trials,:,:],
                                                    axis=0, keepdims=True), axis=1, keepdims=True)
                
                d['baseline_act'] = sess.trial_matrices[self.ts_key][baseline_trials,:, :]
                d['block_act'] = sess.trial_matrices[self.ts_key][block_mask, :, :]
                d['block_fam_act'] = sess.trial_matrices[self.ts_key][block_fam_mask, :, :]
                d['block_nov_act'] = sess.trial_matrices[self.ts_key][block_nov_mask, :, :]



                self.activity_rates[cond][mouse][day] = d


    def plot_single_session(self, cond, mouse, day, 
                            baseline_corr = True,
                            vmin=None, vmax=None, cmap='magma'):
        data = self.activity_rates[cond][mouse][day]
        
        fig = plt.figure()
        gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, .1], height_ratios=[1, 1], wspace=.5)
        ax1 = fig.add_subplot(gs[:, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 1])
        cbar_ax = fig.add_subplot(gs[0,2])



        if baseline_corr:

            b = data['baseline_avg_act']

            
            comb = np.nanmean(np.concatenate((data['baseline_act'], data['block_act']))/b,axis=-1)
            fam = np.nanmean(data['block_fam_act']/b, axis=-1)
            nov = np.nanmean(data['block_nov_act']/b, axis=-1)

        else:
            comb = np.nanmean(np.concatenate((data['baseline_act'], data['block_act'])),axis=-1)
            fam = np.nanmean(data['block_fam_act'], axis=-1)
            nov = np.nanmean(data['block_nov_act'], axis=-1)

        if vmin is None:
            vmin = np.nanpercentile(comb,1)

        if vmax is None:
            vmax = np.nanpercentile(comb, 99)

        kwargs = {'vmin': vmin,
                'vmax': vmax,
                'cmap': cmap}

        h = ax1.imshow(comb, **kwargs)

        
        h = ax2.imshow(fam, **kwargs)
        ax2.set_title('familiar')

        
        h = ax3.imshow(nov, **kwargs)
        ax3.set_title('novel')

        fig.suptitle(f"mouse: {mouse}, day: {day}")