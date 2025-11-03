from . import session
from . import utilities as u 

from . import ymaze_sess_deets

from matplotlib import pyplot as plt

import scipy as sp
import numpy as np
import pandas as pd


from pingouin import mixed_anova, pairwise_ttests

ctrl_mice =ymaze_sess_deets.ctrl_mice
ko_mice = ymaze_sess_deets.ko_mice
sparse_mice = ymaze_sess_deets.sparse_mice




def pos_to_rad(pos, n_spatial_bins=30):
    '''assuming position is in integer bins form 0 to n_spatial_bins-1'''
    rad = np.linspace(-np.pi, np.pi, num=n_spatial_bins+1)

    return pos/(n_spatial_bins+1)*2*np.pi


class PeriRewardCellFrac_Dense:

    def __init__(self, mouse_dict, days, ts_key='F_dff', place_cell_only=False):
        '''
        mouse_dict : e.g. {'ctrl': ctrl_mice, 'ko':ko_mice}
        '''

        self.mouse_dict = mouse_dict
        self.days = days

        self.df = None
        self.ttypes = ('fam', 'nov')
        self.place_cell_only = place_cell_only
        self.fill_df(ts_key)
        


    def fill_df(self, ts_key):
        df = {
            'cond': [],
            'mouse':[],
            'day': [],
            'ttype': [],
            'lr': [],
            'frac': [],
            'licks': [],
            'speed': [],
        }

        for cond, mice in self.mouse_dict.items():
            for mouse in mice:
                for day in self.days:
                    sess = u.load_single_day(mouse, day, trial_mat_keys=[ts_key, 'licks', 'speed'])
                    for ttype in self.ttypes:
                        
                        if ttype == 'fam':
                            trial_mask = sess.trial_info['LR']!=sess.novel_arm
                            lr = -1*sess.novel_arm
                            cell_mask = sess.fam_place_cell_mask()
                        else:
                            trial_mask = sess.trial_info['LR']==sess.novel_arm
                            lr = sess.novel_arm
                            cell_mask = sess.nov_place_cell_mask()


                        first_bin = sess.trial_matrices['bin_centers'][0]
                        if lr == -1:
                            antic_zone = (sess.rzone_early['tfront']-5, sess.rzone_early['tfront']-1)
                        else:
                            antic_zone = (sess.rzone_late['tfront']-5, sess.rzone_late['tfront']-1)
                        antic_zone = [a-first_bin for a in antic_zone]

                        avg_mat = np.nanmean(sess.trial_matrices[ts_key][trial_mask,:,:],axis=0)
                        lick_mat = np.nanmean(sess.trial_matrices['licks'][trial_mask,:],axis=0).ravel()
                        speed_mat = np.nanmean(sess.trial_matrices['speed'][trial_mask,:],axis=0).ravel()

                        if self.place_cell_only:
                            avg_mat = avg_mat[:, cell_mask]
                        argmax = np.nanargmax(avg_mat,axis=0)

                        reward_cells = (argmax>=antic_zone[0]) * (argmax<=antic_zone[1])
                        reward_frac = reward_cells.sum().astype(float)/float(reward_cells.shape[0])

                        antic_zone_inds = [np.floor(antic_zone[0]).astype(int), np.ceil(antic_zone[1]).astype(int)]
                        reward_licks = lick_mat[antic_zone_inds[0]:antic_zone_inds[1]+1].mean()
                        reward_speed = speed_mat[antic_zone_inds[0]:antic_zone_inds[1]+1].mean()


                        df['cond'].append(cond)
                        df['mouse'].append(mouse)
                        df['day'].append(day)
                        df['ttype'].append(ttype)
                        df['lr'].append(lr)
                        df['frac'].append(reward_frac)
                        df['licks'].append(reward_licks)
                        df['speed'].append(reward_speed)
        self.df = pd.DataFrame.from_dict(df)

 

class PeriRewardCellFrac_Sparse:

    def __init__(self, mice, days, ts_key='F_dff', place_cell_only=True):
        '''
        mouse_dict : e.g. {'ctrl': ctrl_mice, 'ko':ko_mice}
        '''

        self.mice = mice
        self.days = days

        self.df = None
        self.ttypes = ('fam', 'nov')
        self.place_cell_only = place_cell_only
        self.fill_df(ts_key)
        


    def fill_df(self, ts_key):
        df = {
            'mouse':[],
            'chan': [],
            'day': [],
            'ttype': [],
            'lr': [],
            'frac': [],
            'licks': [],
            'speed': [],
        }

        
        for mouse in self.mice:
            for day in self.days:

                if (mouse == 'SparseKO_09') and (day==2):
                    continue
                sess = u.load_single_day(mouse, day,pkl_basedir='/home/mplitt/YMazeSessPkls')
                for ttype in self.ttypes:
          
                    for chan in ('channel_0','channel_1'):
                        if ttype == 'fam':
                            trial_mask = sess.trial_info['LR']!=sess.novel_arm
                            lr = -1*sess.novel_arm
                            cell_mask = sess.fam_place_cell_mask(mux=True, chan=chan)
                        else:
                            trial_mask = sess.trial_info['LR']==sess.novel_arm
                            lr = sess.novel_arm
                            cell_mask = sess.nov_place_cell_mask(mux=True, chan=chan)


                        first_bin = sess.trial_matrices['bin_centers'][0]
                        if lr == -1:
                            antic_zone = (sess.rzone_early['tfront']-5, sess.rzone_early['tfront']-1)
                        else:
                            antic_zone = (sess.rzone_late['tfront']-5, sess.rzone_late['tfront']-1)
                        antic_zone = [a-first_bin for a in antic_zone]

                        avg_mat = np.nanmean(sess.trial_matrices[f'{chan}_{ts_key}'][trial_mask,:,:],axis=0)
                        lick_mat = np.nanmean(sess.trial_matrices['licks'][trial_mask,:],axis=0).ravel()
                        speed_mat = np.nanmean(sess.trial_matrices['speed'][trial_mask,:],axis=0).ravel()

                        if self.place_cell_only:
                            avg_mat = avg_mat[:, cell_mask]
                        argmax = np.nanargmax(avg_mat,axis=0)

                        reward_cells = (argmax>=antic_zone[0]) * (argmax<=antic_zone[1])
                        reward_frac = reward_cells.sum().astype(float)/float(reward_cells.shape[0])

                        antic_zone_inds = [np.floor(antic_zone[0]).astype(int), np.ceil(antic_zone[1]).astype(int)]
                        reward_licks = np.nanmean(lick_mat[antic_zone_inds[0]:antic_zone_inds[1]+1])
                        reward_speed = np.nanmean(speed_mat[antic_zone_inds[0]:antic_zone_inds[1]+1])

                        
                        df['mouse'].append(mouse)
                        df['day'].append(day)
                        df['ttype'].append(ttype)
                        df['lr'].append(lr)
                        df['frac'].append(reward_frac)
                        df['chan'].append(chan)
                        df['licks'].append(reward_licks)
                        df['speed'].append(reward_speed)
        self.df = pd.DataFrame.from_dict(df)








#

class PeriRewardPlaceCellActivity:

    def __init__(self, days=np.arange(6), ts_key='spks', fam=True):
        '''


        '''
        self.ko_mice = ymaze_sess_deets.ko_mice
        self.ctrl_mice = ymaze_sess_deets.ctrl_mice
        self.__dict__.update({'days': days, 'ts_key': ts_key, 'fam': fam})
        self.n_days = days.shape[0]

        get_pc_max = u.loop_func_over_days(self.ratemap_perireward, days, ts_key=ts_key, fam=fam)

        self.ko_ratemap = {mouse: get_pc_max(mouse) for mouse in self.ko_mice}
        self.ctrl_ratemap = {mouse: get_pc_max(mouse) for mouse in self.ctrl_mice}

        self.ko_sums = None
        self.ctrl_sums = None

        self.ko_plot_mu = None
        self.ctrl_plot_mu = None

        self.ko_plot_sem = None
        self.ctrl_plot_sem = None



    @staticmethod
    def ratemap_perireward(sess: session.YMazeSession, ts_key: str = 'spks', fam: bool = True, xbounds=(-10, 3)):

        bin_edges = sess.trial_matrices['bin_edges']
        if fam:
            cell_mask = sess.fam_place_cell_mask()
            trial_mask = sess.trial_info['LR'] == -1 * sess.novel_arm
            rzone_front = np.argwhere((sess.rzone_fam['tfront'] <= bin_edges[1:]) * \
                                      (sess.rzone_fam['tfront'] >= bin_edges[:-1]))[0][0]
        else:
            cell_mask = sess.nov_place_cell_mask()
            trial_mask = sess.trial_info['LR'] == sess.novel_arm
            rzone_front = np.argwhere((sess.rzone_nov['tfront'] <= bin_edges[1:]) * \
                                      (sess.rzone_nov['tfront'] >= bin_edges[:-1]))[0][0]

        ratemap_z = sp.stats.zscore(np.nanmean(sess.trial_matrices[ts_key][trial_mask, :, :], axis=0)[:, cell_mask], axis=0)

        return ratemap_z[rzone_front + xbounds[0]:rzone_front + xbounds[1], :]


    def perireward_activity(self):

        fig, ax = plt.subplots(2, self.n_days, figsize=[self.n_days * 5, 10], sharey=True)

        x = np.arange(-10, 3)
        anova_mask = (x > -5) * (x <= -1)
        plot_mask = (x >= -10) * (x <= 1)

        def get_ratemap_sum(ratemap):
            '''

            :param frac:
            :return:
            '''
            plot_mu = np.zeros([len(ratemap.keys()), self.n_days, x.shape[0]])
            plot_sem = np.zeros([len(ratemap.keys()), self.n_days, x.shape[0]])
            sums = np.zeros([len(ratemap.keys()), self.n_days])
            for m, (mouse, data_list) in enumerate(ratemap.items()):
                for col, data in enumerate(data_list):
                    mu, sem = data.mean(axis=-1), sp.stats.sem(data, axis=-1)

                    sums[m, col] = mu[anova_mask].mean()
                    plot_mu[m, col, :] = mu
                    plot_sem[m, col, :] = sem
            return sums, plot_mu, plot_sem

        self.ko_sums, self.ko_plot_mu, self.ko_plot_sem = get_ratemap_sum(self.ko_ratemap)
        self.ctrl_sums, self.ctrl_plot_mu, self.ctrl_plot_sem = get_ratemap_sum(self.ctrl_ratemap)

        for day in range(self.n_days):
            # for m in range(len(self.ko_mice)):
            #     # ax[0, day].fill_between(x, self.ko_plot_mu[m, day, :] - self.ko_plot_sem[m, day, :],
            #                             self.ko_plot_mu[m, day, :] + self.ko_plot_sem[m, day, :], color='red', alpha=.3)
            # for m in range(len(self.ctrl_mice)):
            #     ax[0, day].fill_between(x, self.ctrl_plot_mu[m, day, :] - self.ctrl_plot_sem[m, day, :],
            #                             self.ctrl_plot_mu[m, day, :] + self.ctrl_plot_sem[m, day, :], color='black',
            #                             alpha=.3)

            ax[1, day].plot(x, 1/30.*(0*x + 1.), 'k--')
            ko_mu, ko_sem = self.ko_plot_mu[:, day, :].mean(axis=0), sp.stats.sem(self.ko_plot_mu[:, day, :])
            ax[1, day].fill_between(x, ko_mu - ko_sem, ko_mu + ko_sem, color='red', alpha=.3)

            ctrl_mu, ctrl_sem = self.ctrl_plot_mu[:, day, :].mean(axis=0), sp.stats.sem(
                self.ctrl_plot_mu[:, day, :])
            ax[1, day].fill_between(x, ctrl_mu - ctrl_sem, ctrl_mu + ctrl_sem, color='black', alpha=.3)

            for row in range(2):
                ax[row, day].spines['top'].set_visible(False)
                ax[row, day].spines['right'].set_visible(False)

                ax[row, day].set_title("Day %d" % (day + 1))
                ax[row, day].set_xlabel("Distance from reward")
        ax[0, 0].set_ylabel('Norm. Activity Rate')
        ax[1, 0].set_ylabel('Norm. Activity Rate')

        fig.subplots_adjust(hspace=.5)

        return fig, ax

    def mixed_anova(self, verbose=True, group_tukey=True, day_tukey=True):
        '''

        :param verbose:
        :param group_tukey:
        :param day_tukey:
        :return:
        '''

        df = {'ko_ctrl': [],
              'day': [],
              'sum': [],
              'mouse': []}

        for mouse in range(len(self.ko_mice)):
            for day in self.days:
                df['ko_ctrl'].append(0)
                df['day'].append(day)
                df['sum'].append(self.ko_sums[mouse, day])
                df['mouse'].append(mouse)

        for mouse in range(len(self.ctrl_mice)):
            for day in self.days:
                df['ko_ctrl'].append(1)
                df['day'].append(day)
                df['sum'].append(self.ctrl_sums[mouse, day])
                df['mouse'].append(mouse + len(self.ko_mice))

        df = pd.DataFrame(df)
        results = {}
        aov = mixed_anova(data=df, dv='sum', between='ko_ctrl', within='day', subject='mouse')
        results['anova'] = aov
        posthoc = pairwise_ttests(data=df, dv='sum', between='ko_ctrl', within='day', subject='mouse', padjust ='holm')
        results['posthoc']=posthoc
        if verbose:
            print('Mixed design ANOVA results')
            print(aov)
            print(posthoc)

        # if group_tukey:
        #     ko_ctrl_tukey = pairwise_tukey(data=df, dv='sum', between='ko_ctrl')
        #     results['ko_ctrl_tukey'] = ko_ctrl_tukey
        #     if verbose:
        #         print('PostHoc Tukey: KO vs Ctrl')
        #         print(ko_ctrl_tukey)
        #
        # if day_tukey:
        #     day_stats = []
        #     print('PostHov Tukey on each day')
        #     for day in self.days:
        #         print('Day %d' % day)
        #         stats = pairwise_tukey(data=df[df['day'] == day], dv='sum', between='ko_ctrl')
        #         day_stats.append(stats)
        #         if verbose:
        #             print(stats)
        #     results['day_tukey'] = day_stats

        return results




def plot_leftright_crossval_placecells_withinday(day, ts_key = 'F_dff', vmin = 0, vmax = 4):
    '''

    :param day:
    :param ts_key:
    :return:
    '''


    def lr_ratemaps(mice):
        '''

        :param mice:
        :return:
        '''
        l_rm_train, l_rm_test, r_rm_train, r_rm_test = [], [], [], []
        for mouse in mice:
            sess = u.load_single_day(mouse, day,pkl_basedir='/home/mplitt/YMazeSessPkls', trial_mat_keys=[ts_key,])
            if 'left' in sess.place_cell_info.keys():
                l_cellmask = sess.place_cell_info['left']['masks']
                r_cellmask= sess.place_cell_info['right']['masks']
            else:
                l_cellmask = sess.place_cell_info[-1]['masks'].sum(axis=0)>0
                r_cellmask = sess.place_cell_info[1]['masks'].sum(axis=0) > 0

            trial_mat = sess.trial_matrices[ts_key]

            l_trialmask = sess.trial_info['LR'] == -1
            r_trialmask = sess.trial_info['LR'] == 1

            l_trialmat = trial_mat[l_trialmask, :, :]
            l_trialmat = l_trialmat[:, :, l_cellmask]

            r_trialmat = trial_mat[r_trialmask, :, :]
            r_trialmat = r_trialmat[:, :, r_cellmask]

            l_rm_train.append(np.nanmean(l_trialmat[::2, :, :], axis=0))
            l_rm_test.append(np.nanmean(l_trialmat[1::2, :, :], axis=0))

            r_rm_train.append(np.nanmean(r_trialmat[::2, :, :], axis=0))
            r_rm_test.append(np.nanmean(r_trialmat[1::2, :, :], axis=0))

        return np.concatenate(l_rm_train, axis=-1), np.concatenate(l_rm_test, axis=-1), \
               np.concatenate(r_rm_train, axis=-1), np.concatenate(r_rm_test, axis=-1)

    def sort_norm(rm_train, rm_test):
        mu, std = np.nanmean(rm_train, axis=0, keepdims=True), np.nanstd(rm_train, axis=0, keepdims=True)
        sortvec = np.argsort(np.argmax(rm_train, axis=0))

        rm_test = (rm_test-mu)/std

        return rm_test[:, sortvec]

    ko_l_train, ko_l_test, ko_r_train, ko_r_test = lr_ratemaps(ymaze_sess_deets.ko_mice)
    ctrl_l_train, ctrl_l_test, ctrl_r_train, ctrl_r_test = lr_ratemaps(ymaze_sess_deets.ctrl_mice)

    

    fig, ax = plt.subplots(2,2, figsize= [10,10])
    ax[0,0].imshow(sort_norm(ctrl_l_train, ctrl_l_test).T, cmap='pink', aspect='auto', vmin=vmin, vmax=vmax)
    ax[0,1].imshow(sort_norm(ctrl_r_train, ctrl_r_test).T, cmap='pink', aspect='auto', vmin=vmin, vmax=vmax)

    ax[0,0].plot([-.5, ctrl_l_train.shape[0]- .5], [-.5, ctrl_l_train.shape[1]-.5], color='blue')
    ax[0,1].plot([-.5, ctrl_r_train.shape[0] - .5], [-.5, ctrl_r_train.shape[1] - .5], color='blue')

    ax[0, 0].set_title(f"mCherry: Left, N cells {ctrl_l_test.shape[1]}")
    ax[0, 1].set_title(f"mCherry: Right, N cells {ctrl_r_test.shape[1]}")

   
    ax[1, 0].imshow(sort_norm(ko_l_train, ko_l_test).T, cmap='pink', aspect='auto', vmin=vmin, vmax=vmax)
    ax[1, 0].plot([-.5, ko_l_train.shape[0]- .5], [-.5, ko_l_train.shape[1]-.5], color='blue')
    ax[1, 0].set_title(f"Cre: Left, N cells {ko_l_test.shape[1]}")

    ax[1, 1].imshow(sort_norm(ko_r_train, ko_r_test).T, cmap='pink', aspect='auto', vmin=vmin, vmax=vmax)
    ax[1, 1].plot([-.5, ko_r_train.shape[0]- .5], [-.5, ko_r_train.shape[1]-.5], color='blue')
    ax[1, 1].set_title(f"Cre: Right, N cells {ko_r_test.shape[1]}")

    for row in [0,1]:
        for col in [0,1]:
            ax[row,col].set_yticks([])
            ax[row,col].set_ylabel('Cells')
            ax[row, col].set_xlabel('Pos')

    fig.subplots_adjust(hspace=.25, wspace=.5)
    fig.suptitle('Day %d' % day)
    return fig, ax


def plot_leftright_crossval_placecells_withinday_sparse(day, ts_key = 'F_dff', vmin = -.25, vmax = 5):
    '''

    :param day:
    :param ts_key:
    :return:
    '''


    def lr_ratemaps(mice):
        '''

        :param mice:
        :return:
        '''
        train_mats = {'channel_0': {'left': [], 'right': []},
                      'channel_1': {'left': [], 'right': []}}
        test_mats = {'channel_0': {'left': [], 'right': []},
                      'channel_1': {'left': [], 'right': []}}
     
        for mouse in mice:
            if mouse == 'SparseKO_09' and day==2:
                continue
            sess = u.load_single_day(mouse, day, pkl_basedir='/home/mplitt/YMazeSessPkls')

            for chan in ('channel_0', 'channel_1'):
                key = chan + '_' + ts_key
            
                l_cellmask = sess.place_cell_info[key]['left']['masks']
                r_cellmask= sess.place_cell_info[key]['right']['masks']
           
                trial_mat = sess.trial_matrices[key]
                

                l_trialmask = sess.trial_info['LR'] == -1
                r_trialmask = sess.trial_info['LR'] == 1

                l_trialmat = trial_mat[l_trialmask, :, :]
                l_trialmat = l_trialmat[:, :, l_cellmask]
            

                r_trialmat = trial_mat[r_trialmask, :, :]
                r_trialmat = r_trialmat[:, :, r_cellmask]

                if l_cellmask.sum()>0:
                    train_mats[chan]['left'].append(np.nanmean(l_trialmat[::2, :, :], axis=0))
                    test_mats[chan]['left'].append(np.nanmean(l_trialmat[1::2, :, :], axis=0))

                if r_cellmask.sum()>0:
                    train_mats[chan]['right'].append(np.nanmean(r_trialmat[::2, :, :], axis=0))
                    test_mats[chan]['right'].append(np.nanmean(r_trialmat[1::2, :, :], axis=0))

        
        for chan in ('channel_0', 'channel_1'):
            for lr in ('left', 'right'):
                train_mats[chan][lr] = np.concatenate(train_mats[chan][lr], axis=-1)
                test_mats[chan][lr] = np.concatenate(test_mats[chan][lr], axis=-1)
            


        return train_mats, test_mats

    def sort_norm(rm_train, rm_test):
        mu, std = np.nanmean(rm_train, axis=0, keepdims=True), np.nanstd(rm_train, axis=0, keepdims=True)
        sortvec = np.argsort(np.argmax(rm_train, axis=0))

        rm_test = (rm_test-mu)/std

        return rm_test[:, sortvec]

    train_mats, test_mats = lr_ratemaps(ymaze_sess_deets.sparse_mice)
    

    fig, ax = plt.subplots(2,2, figsize= [10,10])
    plot_mat = sort_norm(train_mats['channel_1']['left'], test_mats['channel_1']['left'])
    im0 = ax[0,0].imshow(plot_mat.T, cmap='pink', aspect='auto', vmin=vmin, vmax=vmax)
    ax[0,0].plot([-.5, plot_mat.shape[0]- .5], [-.5, plot_mat.shape[1]-.5], color='blue')
    ax[0, 0].set_title("RGECO: Left, N cells %d" % plot_mat.shape[1])

    plot_mat = sort_norm(train_mats['channel_1']['right'], test_mats['channel_1']['right'])
    ax[0,1].imshow(plot_mat.T, cmap='pink', aspect='auto', vmin=vmin, vmax=vmax)    
    ax[0,1].plot([-.5, plot_mat.shape[0] - .5], [-.5, plot_mat.shape[1] - .5], color='blue')
    ax[0, 1].set_title("RGECO: Right, N cells %d" % plot_mat.shape[1])


    plot_mat = sort_norm(train_mats['channel_0']['left'], test_mats['channel_0']['left'])
    ax[1, 0].imshow(plot_mat.T, cmap='pink', aspect='auto', vmin=vmin, vmax=vmax)    
    ax[1, 0].plot([-.5, plot_mat.shape[0] - .5], [-.5, plot_mat.shape[1] - .5], color='blue')
    ax[1, 0].set_title("GCaMP: Left, N cells %d" % plot_mat.shape[1])

    plot_mat = sort_norm(train_mats['channel_0']['right'], test_mats['channel_0']['right'])
    ax[1, 1].imshow(plot_mat.T, cmap='pink', aspect='auto', vmin=vmin, vmax=vmax)    
    ax[1, 1].plot([-.5, plot_mat.shape[0] - .5], [-.5, plot_mat.shape[1] - .5], color='blue')
    ax[1, 1].set_title("GCaMP: Right, N cells %d" % plot_mat.shape[1])
   
    

    for a in ax.flatten():
        a.set_yticks([])
        a.set_ylabel('Cells')
        a.set_xlabel('Pos')
    # cbar = fig.colorbar(im0, ax = ax.ravel().tolist())

    fig.subplots_adjust(hspace=.25, wspace=.5)
    fig.suptitle('Day %d' % day)
    return fig, ax


class RewardCells:

    def __init__(self, days = np.arange(6), ts_key = 'F_dff_nostop'):

        self.days = days
        self.ts_key = ts_key

        sess = u.load_single_day(ctrl_mice[0],0, verbose=False, trial_mat_keys=[ts_key,])
        self.rz_early = (np.argwhere(sess.trial_matrices['bin_edges'][:-1]>=sess.rzone_early['tfront'])[0], np.argwhere(sess.rzone_early['tback']<=sess.trial_matrices['bin_edges'][1:])[0] )
        self.rz_late = (np.argwhere(sess.trial_matrices['bin_edges'][:-1]>=sess.rzone_late['tfront'])[0], 29 )

        self.left_mats = {}
        self.right_mats = {}
        self.build_dicts()
        self.summary_df = None
        self.build_summary_df()


    def build_dicts(self):

        for ko, mice in zip(('ctrl', 'ko'),(ctrl_mice, ko_mice)):
            self.left_mats[ko]={}
            self.right_mats[ko]={}

            for mouse in mice:
                print(mouse)
                self.left_mats[ko][mouse] = {}
                self.right_mats[ko][mouse] = {}

                for day in self.days:
                    l_mat, r_mat = self.get_lr_map(u.load_single_day(mouse,day,verbose=False, trial_mat_keys=[self.ts_key,]))
                    self.left_mats[ko][mouse][day] = l_mat
                    self.right_mats[ko][mouse][day] = r_mat


    def get_lr_map(self, sess):
        
        left_mask = sess.trial_info['LR']==-1
        pc_mask = np.zeros([sess.trial_matrices[self.ts_key].shape[-1],])
        for key in sess.place_cell_info.keys():
            if len(sess.place_cell_info[key]['masks'].shape)>1:
                pc_mask += 1*sess.place_cell_info[key]['masks'].sum(axis=0)
            else:
                pc_mask += 1*sess.place_cell_info[key]['masks']
        pc_mask = pc_mask>0
        
        return np.nanmean(sess.trial_matrices[self.ts_key][left_mask,1:-1,:],axis=0)[:,pc_mask], np.nanmean(sess.trial_matrices[self.ts_key][~left_mask,1:-1,:],axis=0)[:,pc_mask]
    
    @staticmethod
    def get_smooth_hist(max1, max2,bins = np.arange(0,30)):
        hist, xedges, yedges = np.histogram2d(max1,max2, bins = [bins, bins], density = True)
        hist_sm = sp.ndimage.gaussian_filter(hist, (1,1))
        hist_sm /= hist_sm.ravel().sum()
        return hist_sm
    
    
    def plot_heatmaps(self, day):


        ctrl_hist_sm = 0
        for mouse in ctrl_mice:
            ctrl_hist_sm += self.get_smooth_hist(np.argmax(self.left_mats['ctrl'][mouse][day],axis=0), 
                                                 np.argmax(self.right_mats['ctrl'][mouse][day],axis=0))
        ctrl_hist_sm /= len(ctrl_mice)
        ctrl_hist_sm /= ctrl_hist_sm.ravel().sum()

        fig,ax = plt.subplots(1,3,figsize=[9,3])
        fig.subplots_adjust(wspace=.5)
        h = ax[0].imshow(ctrl_hist_sm,vmin=0,vmax=.005, cmap = 'PuRd')
        ax[0].set_title("Control")
        
        plt.colorbar(h,ax = ax[0],shrink=.5)


        ko_hist_sm = 0
        for mouse in ko_mice:
            ko_hist_sm += self.get_smooth_hist(np.argmax(self.left_mats['ko'][mouse][day],axis=0), 
                                                 np.argmax(self.right_mats['ko'][mouse][day],axis=0))
        ko_hist_sm /= len(ko_mice)
        ko_hist_sm /= ko_hist_sm.ravel().sum()

       
        h = ax[1].imshow(ko_hist_sm,vmin=0,vmax=.005, cmap = 'PuRd')
        ax[1].set_title("KO")
        
        plt.colorbar(h,ax = ax[1], shrink=.5)

        
        h = ax[2].imshow(ctrl_hist_sm-ko_hist_sm,cmap='RdGy', vmin= -.002, vmax = .002)
        plt.colorbar(h,ax = ax[2], shrink=.5)

        for a in ax.flatten():
            a.set_ylabel("Left Position (cm)")
            a.set_xlabel("Right Position (cm)")
            a.fill_between(np.linspace(-.5,28.5), self.rz_early[0], self.rz_early[1]-.5,  alpha=.3, color='blue')
            a.fill_betweenx(np.linspace(-.5,28.5), self.rz_late[0], self.rz_late[1]-.5,  alpha=.3, color='green')

            a.set_yticks([0,10,20], labels = ['0', '100', '200'])
            a.set_xticks([0,10,20], labels = ['0', '100', '200'])
            
            a.plot([-.5,28.5], [-.5,28.5], 
                   'k--')
            a.set_ylim([28.5,-.5])
            a.set_xlim([-.5, 28.5])

        return fig, ax
    

    def build_summary_df(self):


        df = {'mouse':[],
                'ko':[],
                'day':[], 
                'frac': [], 
                }

        for ko, mice in zip(('ctrl','ko'), (ctrl_mice,ko_mice)):
            for mouse in mice:
                for day in range(6):
                    _hist = self.get_smooth_hist(np.argmax(self.left_mats[ko][mouse][day],axis=0), 
                                                 np.argmax(self.right_mats[ko][mouse][day],axis=0))
                    frac = _hist[self.rz_early[0][0]-5:self.rz_early[0][0],
                                 self.rz_late[0][0]-5:self.rz_late[0][0]].sum(axis=-1).sum(axis=-1)
                    
                    df['mouse'].append(mouse)
                    df['ko'].append(ko)
                    df['day'].append(day)
                    df['frac'].append(frac)
       
        self.summary_df = pd.DataFrame.from_dict(df)
       


class RewardCells_Sparse:

    def __init__(self, days = np.arange(6), ts_key = 'F_dff'):

        self.days = days
        self.ts_key = ts_key

        sess = u.load_single_day(sparse_mice[0],0, verbose=False)
        self.rz_early = (np.argwhere(sess.trial_matrices['bin_edges'][:-1]>=sess.rzone_early['tfront'])[0], np.argwhere(sess.rzone_early['tback']<=sess.trial_matrices['bin_edges'][1:])[0] )
        self.rz_late = (np.argwhere(sess.trial_matrices['bin_edges'][:-1]>=sess.rzone_late['tfront'])[0], 29 )

        self.left_mats = {}
        self.right_mats = {}
        self.build_dicts()
        self.summary_df = None
        self.build_summary_df()


    def build_dicts(self):



        for mouse in sparse_mice:
            print(mouse)
            self.left_mats[mouse] = {}
            self.right_mats[mouse] = {}

            for day in self.days:
                if (mouse == 'SparseKO_09') and (day==2):
                    continue
                sess = u.load_single_day(mouse, day, verbose=False)
                self.left_mats[mouse][day] = {}
                self.right_mats[mouse][day] = {}
                for chan in ('channel_0', 'channel_1'):
                    l_mat, r_mat = self.get_lr_map(sess,chan)
                    self.left_mats[mouse][day][chan] = l_mat
                    self.right_mats[mouse][day][chan] = r_mat


    def get_lr_map(self, sess, chan):
        ts_key = chan + '_' + self.ts_key
        
        left_mask = sess.trial_info['LR']==-1

        l_cellmask = 1.*sess.place_cell_info[ts_key]['left']['masks']
        r_cellmask= 1.*sess.place_cell_info[ts_key]['right']['masks']
        pc_mask = l_cellmask+r_cellmask
            
        pc_mask = pc_mask>0
        
        return np.nanmean(sess.trial_matrices[ts_key][left_mask,1:-1,:],axis=0)[:,pc_mask], np.nanmean(sess.trial_matrices[ts_key][~left_mask,1:-1,:],axis=0)[:,pc_mask]
    
    @staticmethod
    def get_smooth_hist(max1, max2,bins = np.arange(0,30)):
        hist, xedges, yedges = np.histogram2d(max1,max2, bins = [bins, bins], density = True)
        hist_sm = sp.ndimage.gaussian_filter(hist, (1,1))
        hist_sm /= hist_sm.ravel().sum()
        return hist_sm
    
    
    def plot_heatmaps(self, day):

        
        ctrl_hist_sm = 0
        ko_hist_sm = 0
        for mouse in sparse_mice:
            ctrl_hist_sm += self.get_smooth_hist(np.argmax(self.left_mats[mouse][day]['channel_1'],axis=0), 
                                                 np.argmax(self.right_mats[mouse][day]['channel_1'],axis=0))
            
            ko_hist_sm += self.get_smooth_hist(np.argmax(self.left_mats[mouse][day]['channel_0'],axis=0), 
                                                 np.argmax(self.right_mats[mouse][day]['channel_0'],axis=0))
        ctrl_hist_sm /= len(sparse_mice)
        ctrl_hist_sm /= ctrl_hist_sm.ravel().sum()

        ko_hist_sm /= len(sparse_mice)
        ko_hist_sm /= ko_hist_sm.ravel().sum()

        fig,ax = plt.subplots(1,3,figsize=[9,3])
        fig.subplots_adjust(wspace=.5)
        h = ax[0].imshow(ctrl_hist_sm,vmin=0,vmax=.005, cmap = 'PuRd')
        ax[0].set_title("Control")
        
        plt.colorbar(h,ax = ax[0],shrink=.5)
       
        h = ax[1].imshow(ko_hist_sm,vmin=0,vmax=.005, cmap = 'PuRd')
        ax[1].set_title("KO")
        
        plt.colorbar(h,ax = ax[1], shrink=.5)

        
        h = ax[2].imshow(ctrl_hist_sm-ko_hist_sm,cmap='RdGy', vmin= -.002, vmax = .002)
        plt.colorbar(h,ax = ax[2], shrink=.5)

        for a in ax.flatten():
            a.set_ylabel("Left Position (cm)")
            a.set_xlabel("Right Position (cm)")
            a.fill_between(np.linspace(-.5,28.5), self.rz_early[0], self.rz_early[1]-.5,  alpha=.3, color='blue')
            a.fill_betweenx(np.linspace(-.5,28.5), self.rz_late[0], self.rz_late[1]-.5,  alpha=.3, color='green')

            a.set_yticks([0,10,20], labels = ['0', '100', '200'])
            a.set_xticks([0,10,20], labels = ['0', '100', '200'])

            a.plot([-.5,28.5], [-.5,28.5], 
                   'k--')
            a.set_ylim([28.5,-.5])
            a.set_xlim([-.5, 28.5])

        return fig, ax
    

    def build_summary_df(self):


        df = {'mouse':[],
                'chan':[],
                'day':[], 
                'frac': [], 
                }

        
        for mouse in sparse_mice:
            for day in range(6):
                if mouse == 'SparseKO_09' and day ==2:
                    continue
                for chan in ('channel_0', 'channel_1'):
                    _hist = self.get_smooth_hist(np.argmax(self.left_mats[mouse][day][chan],axis=0), 
                                                 np.argmax(self.right_mats[mouse][day][chan],axis=0))
                    frac = _hist[self.rz_early[0][0]-5:self.rz_early[0][0],
                                 self.rz_late[0][0]-5:self.rz_late[0][0]].sum(axis=-1).sum(axis=-1)
                    
                    df['mouse'].append(mouse)
                    df['chan'].append(chan)
                    df['day'].append(day)
                    df['frac'].append(frac)
       
        self.summary_df = pd.DataFrame.from_dict(df)