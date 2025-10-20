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
        self.df = None


    def get_activity(self):



        for cond, mice, in self.mouse_dict.items():
            self.activity_rates[cond] = {}
            for mouse in mice:
                print(mouse)
                self.activity_rates[cond][mouse] = {}
                for day in self.days:

                    d = {}
                 
                    sess = u.load_single_day(mouse, day,verbose=False)
                    trial_mat = sess.trial_matrices[self.ts_key]
                    speed_mat = sess.trial_matrices['speed']
                    lick_mat = sess.trial_matrices['licks']
                    if trial_mat.ndim ==2:
                        trial_mat = trial_mat[:,:, np.newaxis]


                    block_start_trial_num = np.argwhere(sess.trial_info['block_number']==self.block)[0][0]


                    baseline_trials = slice(block_start_trial_num-10, block_start_trial_num)

                    block_mask = sess.trial_info['block_number']==self.block
                    block_fam_mask = block_mask * (sess.trial_info['LR']==-1*sess.novel_arm)
                    block_nov_mask = block_mask * (sess.trial_info['LR']==sess.novel_arm)

                    # average over trials and positions
                    d['baseline_avg_act'] = np.nanmean(np.nanmean(trial_mat[baseline_trials,:,:],
                                                      axis=0, keepdims=True), axis=1, keepdims=True)
                    
                    d['baseline_act'] = trial_mat[baseline_trials,:, :]
                    d['block_act'] = trial_mat[block_mask, :, :]
                    d['block_fam_act'] = trial_mat[block_fam_mask, :, :]
                    d['block_nov_act'] = trial_mat[block_nov_mask, :, :]

                    d['baseline_speed'] = speed_mat[baseline_trials, :]
                    d['block_speed'] = speed_mat[block_mask, :]
                    d['block_fam_speed'] = speed_mat[block_fam_mask, :]
                    d['block_nov_speed'] = speed_mat[block_nov_mask, :]

                    d['baseline_licks'] = lick_mat[baseline_trials, :]
                    d['block_licks'] = lick_mat[block_mask, :]
                    d['block_fam_licks'] = lick_mat[block_fam_mask, :]
                    d['block_nov_licks'] = lick_mat[block_nov_mask, :]



                    self.activity_rates[cond][mouse][day] = d

    def build_dataframe(self, norm = 'population', norm_behavior=True, max_trial = 5):

        rows = []
        for cond, mice, in self.mouse_dict.items():
            for mouse in mice:
                for day in self.days:
                    data = self.activity_rates[cond][mouse][day]

                    

                    if norm == 'population':
                        base = np.nanmean(np.nanmean(data['baseline_act'],axis=1),axis=-1)
                        fam = np.nanmean(np.nanmean(data['block_fam_act'],axis=1),axis=-1)[:max_trial]/base.mean()
                        nov = np.nanmean(np.nanmean(data['block_nov_act'],axis=1), axis=-1)[:max_trial]/base.mean()


                        base_speed = np.nanmean(data['baseline_speed'],axis=1)
                        fam_speed = np.nanmean(data['block_fam_speed'],axis=1)[:max_trial]
                        nov_speed = np.nanmean(data['block_nov_speed'], axis=-1)[:max_trial]

                        base_licks = np.nanmean(data['baseline_licks'],axis=1)
                        fam_licks = np.nanmean(data['block_fam_licks'],axis=1)[:max_trial]
                        nov_licks = np.nanmean(data['block_nov_licks'], axis=-1)[:max_trial]

                        if norm_behavior:
                            fam_speed = fam_speed/base_speed.mean()
                            nov_speed = nov_speed/base_speed.mean()

                            fam_licks = fam_licks/base_licks.mean()
                            nov_licks = nov_licks/base_licks.mean()


                    elif norm == 'cell':
                        cell_denom = data['baseline_avg_act']
                        base = np.nanmean(np.nanmean(data['baseline_act']/cell_denom,axis=1),axis=-1)
                        fam = np.nanmean(np.nanmean(data['block_fam_act']/cell_denom,axis=1),axis=-1)[:max_trial]
                        nov = np.nanmean(np.nanmean(data['block_nov_act']/cell_denom,axis=1), axis=-1)[:max_trial]

                        base_speed = np.nanmean(data['baseline_speed'],axis=1)
                        fam_speed = np.nanmean(data['block_fam_speed'],axis=1)[:max_trial]
                        nov_speed = np.nanmean(data['block_nov_speed'], axis=-1)[:max_trial]

                        base_licks = np.nanmean(data['baseline_licks'],axis=1)
                        fam_licks = np.nanmean(data['block_fam_licks'],axis=1)[:max_trial]
                        nov_licks = np.nanmean(data['block_nov_licks'], axis=-1)[:max_trial]

                        if norm_behavior:
                            fam_speed = fam_speed/base_speed.mean()
                            nov_speed = nov_speed/base_speed.mean()

                            fam_licks = fam_licks/base_licks.mean()
                            nov_licks = nov_licks/base_licks.mean()

                    elif norm is None:
                        base = np.nanmean(np.nanmean(data['baseline_act'],axis=1),axis=-1)
                        fam = np.nanmean(np.nanmean(data['block_fam_act'],axis=1),axis=-1)[:max_trial]
                        nov = np.nanmean(np.nanmean(data['block_nov_act'],axis=1), axis=-1)[:max_trial]

                        base_speed = np.nanmean(data['baseline_speed'],axis=1)
                        fam_speed = np.nanmean(data['block_fam_speed'],axis=1)[:max_trial]
                        nov_speed = np.nanmean(data['block_nov_speed'], axis=-1)[:max_trial]

                        base_licks = np.nanmean(data['baseline_licks'],axis=1)
                        fam_licks = np.nanmean(data['block_fam_licks'],axis=1)[:max_trial]
                        nov_licks = np.nanmean(data['block_nov_licks'], axis=-1)[:max_trial]
                        
                    else:
                        raise ValueError("norm must be one of 'population', 'cell', or None")
                    
                    rows.append({'condition': cond,
                                'mouse': mouse,
                                'day': day,
                                'baseline_rate': base.mean(),
                                'familiar_rate': fam.mean(),
                                'novel_rate': nov.mean(),
                                'baseline_speed': base_speed.mean(),
                                'familiar_speed': fam_speed.mean(),
                                'novel_speed': nov_speed.mean(),
                                'baseline_licks': base_licks.mean(),
                                'familiar_licks': fam_licks.mean(),
                                'novel_licks': nov_licks.mean()})
    
        self.df = pd.DataFrame(rows)
        return self.df


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
        self.df = None


    def get_activity(self):



        
        for mouse in self.mice:
            print(mouse)
            self.activity_rates[mouse] = {}
            for day in self.days:
                if mouse == 'SparseKO_09' and day ==2:
                    continue


                self.activity_rates[mouse][day] = {}
                
                sess = u.load_single_day(mouse, day,verbose=False)


                block_start_trial_num = np.argwhere(sess.trial_info['block_number']==self.block)[0][0]


                baseline_trials = slice(block_start_trial_num-10, block_start_trial_num)

                block_mask = sess.trial_info['block_number']==self.block
                block_fam_mask = block_mask * (sess.trial_info['LR']==-1*sess.novel_arm)
                block_nov_mask = block_mask * (sess.trial_info['LR']==sess.novel_arm)


                for chan in ('channel_0', 'channel_1'):
                    d = {}

                    _ts_key = f'{chan}_{self.ts_key}'



                
                    # average over trials and positions
                    d['baseline_avg_act'] = np.nanmean(np.nanmean(sess.trial_matrices[_ts_key][baseline_trials,:,:],
                                                        axis=0, keepdims=True), axis=1, keepdims=True)
                    
                    d['baseline_act'] = sess.trial_matrices[_ts_key][baseline_trials,:, :]
                    d['block_act'] = sess.trial_matrices[_ts_key][block_mask, :, :]
                    d['block_fam_act'] = sess.trial_matrices[_ts_key][block_fam_mask, :, :]
                    d['block_nov_act'] = sess.trial_matrices[_ts_key][block_nov_mask, :, :]



                    self.activity_rates[mouse][day][chan] = d

    def build_dataframe(self, norm = 'population', max_trial = 5):

        rows = []
        
        for mouse in self.mice:
            for day in self.days:
                for chan in ('channel_0', 'channel_1'):
                    if mouse == 'SparseKO_09' and day ==2:
                        continue
                    data = self.activity_rates[mouse][day][chan]

                    if norm == 'population':
                        base = np.nanmean(np.nanmean(data['baseline_act'],axis=1),axis=-1)
                        # base /= base.mean()
                        fam = np.nanmean(np.nanmean(data['block_fam_act'],axis=1),axis=-1)[:max_trial]/base.mean()
                        nov = np.nanmean(np.nanmean(data['block_nov_act'],axis=1), axis=-1)[:max_trial]/base.mean()

                    elif norm == 'cell':
                        cell_denom = data['baseline_avg_act']
                        base = np.nanmean(np.nanmean(data['baseline_act']/cell_denom,axis=1),axis=-1)
                        fam = np.nanmean(np.nanmean(data['block_fam_act']/cell_denom,axis=1),axis=-1)[:max_trial]
                        nov = np.nanmean(np.nanmean(data['block_nov_act']/cell_denom,axis=1), axis=-1)[:max_trial]
                    elif norm is None:
                        base = np.nanmean(np.nanmean(data['baseline_act'],axis=1),axis=-1)
                        fam = np.nanmean(np.nanmean(data['block_fam_act'],axis=1),axis=-1)[:max_trial]
                        nov = np.nanmean(np.nanmean(data['block_nov_act'],axis=1), axis=-1)[:max_trial]
                    else:
                        raise ValueError("norm must be one of 'population', 'cell', or None")
                    
                    rows.append({'channel': chan,
                                'mouse': mouse,
                                'day': day,
                                'baseline_rate': base.mean(),
                                'familiar_rate': fam.mean(),
                                'novel_rate': nov.mean()})
    
        self.df = pd.DataFrame(rows)
        return self.df



    def plot_single_session(self, cond, mouse, day, chan,
                            baseline_corr = True,
                            vmin=None, vmax=None, cmap='magma'):
        data = self.activity_rates[mouse][day][chan]
        
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