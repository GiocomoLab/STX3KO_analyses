import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
import pandas as pd

from pingouin import mixed_anova, pairwise_tukey

import TwoPUtils as tpu

from . import spatial_analyses
from . import ymaze_sess_deets
from . import utilities as u

class CellStats:

    def __init__(self, ts_key = 'spks', fam=True, days = np.arange(0, 6)):
        '''

        :param ts_key:
        :param fam:
        :param days:
        '''

        self.ko_mice = ymaze_sess_deets.ko_mice
        self.ctrl_mice = ymaze_sess_deets.ctrl_mice
        self.__dict__.update({'days': days, 'ts_key': ts_key, 'fam': fam})
        self.n_days = days.shape[0]

        get_stats = u.loop_func_over_days(self.single_sess_lap_stats, days, ts_key=ts_key, fam=fam)

        self.ko_stats = {mouse: get_stats(mouse) for mouse in self.ko_mice}
        self.ctrl_stats = {mouse: get_stats(mouse) for mouse in self.ctrl_mice}




    @staticmethod
    def single_sess_lap_stats(sess, fam=True, ts_key='spks'):
        '''

        :param sess:
        :param fam:
        :param ts_key:
        :return:
        '''

        if fam:
            cell_mask = sess.fam_place_cell_mask()
            trial_mask = sess.trial_info['LR'] == -1 * sess.novel_arm

            time_mask = 0*sess.timeseries['spks'][0,:]
            for (start,stop, lr) in zip(sess.trial_start_inds, sess.teleport_inds, sess.trial_info['LR']):
                if lr == -1*sess.novel_arm:
                    time_mask[start:stop]=1

        else:
            cell_mask = sess.nov_place_cell_mask()
            trial_mask = sess.trial_info['LR'] == sess.novel_arm

            time_mask = 0 * sess.timeseries['spks'][0,:]
            for (start, stop, lr) in zip(sess.trial_start_inds, sess.teleport_inds, sess.trial_info['LR']):
                if lr == sess.novel_arm:
                    time_mask[start:stop] = 1
        time_mask = time_mask>0
        trial_mat = sess.trial_matrices[ts_key][:, :, cell_mask]
        trial_mat = trial_mat[trial_mask, :, :]
        trial_mat[np.isnan(trial_mat)] = 1E-5

        avg_trial_mat = trial_mat.mean(axis=0, keepdims = True)


        inds = np.arange(0, trial_mat.shape[1])[np.newaxis, :, np.newaxis]

        avg_trial_mat_norm = avg_trial_mat / (np.nansum(avg_trial_mat, axis=1, keepdims=True) + 1E-5)
        avg_com = (avg_trial_mat_norm * inds).sum(axis=1, keepdims=True)
        avg_std = np.power((np.power(inds - avg_com, 2) * avg_trial_mat_norm).sum(axis=1, keepdims=True), .5)
        avg_skewness = (np.power((inds - avg_com) / (avg_std + 1E-5), 3) * avg_trial_mat_norm).sum(axis=1)
        avg_kurtosis = (np.power((inds - avg_com) / (avg_std + 1E-5), 4) * avg_trial_mat_norm).sum(axis=1)

        trial_mat_sm = sp.ndimage.filters.gaussian_filter1d(trial_mat, 3, axis=0)
        trial_mat_sm_norm = trial_mat_sm / (np.nansum(trial_mat_sm, axis=1, keepdims=True) + 1E-5)

        com = (trial_mat_sm_norm * inds).sum(axis=1, keepdims=True)
        std = np.power((np.power(inds - com, 2) * trial_mat_sm_norm).sum(axis=1, keepdims=True), .5)

        skewness = (np.power((inds - com) / (std + 1E-5), 3) * trial_mat_sm_norm).sum(axis=1)
        kurtosis = (np.power((inds - com) / (std + 1E-5), 4) * trial_mat_sm_norm).sum(axis=1)



        _rm = 0 * avg_trial_mat[0,:,:]
        cellts_mu = np.nanmean(sess.timeseries['spks'][:,time_mask],axis=1)[cell_mask]
        _rm[avg_trial_mat[0,:,:]> cellts_mu[np.newaxis,:]*np.ones([avg_trial_mat.shape[1],1])] = 1
        rm = np.zeros((avg_trial_mat.shape[1]+1, avg_trial_mat.shape[2]))
        rm[1:, :] = _rm
        numfields = np.count_nonzero(rm[1:,:] > rm[:-1,:], axis = 0)

        return {
                'std': std,
                'skewness': skewness,
                'trial_avg_skewness': skewness.mean(axis=0),
                'kurtosis': kurtosis,
                'avg_std': avg_std.ravel(),
                'avg_skewness': avg_skewness.ravel(),
                'avg_kurtosis': avg_kurtosis.ravel(),
                'max_counts': spatial_analyses.max_counts(avg_trial_mat[0, :, :]),
                'num_fields': numfields,
                'field_width': spatial_analyses.field_width(avg_trial_mat[0, :, :]),
                }
        pass

    def summary_stat_matrices(self, key):
        '''
        return mouse x day array for cre and mcherry data
        :param key:
        :return:
        '''
        def make_mean_mat(data_dict):
            means = np.zeros([len(data_dict.keys()), self.n_days])
            for k, (mouse, data_list) in enumerate(data_dict.items()):
                for day, stat_dict in enumerate(data_list):
                    means[k, day] = np.nanmean(stat_dict[key])
            return means

        return make_mean_mat(self.ko_stats), make_mean_mat(self.ctrl_stats)


    def violin_plots(self, stat_key, scatter_only=False):
        '''

        :param stat_key:
        :param scatter_only:
        :return:
        '''

        fig, ax = plt.subplots(figsize=[3*self.n_days, 5])

        for k, (mouse, data_list) in enumerate(self.ko_stats.items()):
            for day, data_dict in enumerate(data_list):
                data = data_dict[stat_key]
                data = data[~np.isnan(data)]
                if not scatter_only:
                    parts = ax.violinplot(data, positions=[2 * day + .6 + .1 * k], showextrema=False, showmeans=False,
                                          widths=.1, points=10)
                    for pc in parts['bodies']:
                        pc.set_facecolor('red')
                        pc.set_edgecolor('black')
                        pc.set_alpha(.5)
                _ = ax.scatter(2 * day + .6 +.1 * k, data.mean(), color='red')

        for k, (mouse, data_list) in enumerate(self.ctrl_stats.items()):
            for day, data_dict in enumerate(data_list):
                data = data_dict[stat_key]
                data = data[~np.isnan(data)]
                if not scatter_only:
                    parts = ax.violinplot(data, positions=[2 * day + .1 * k], showextrema=False, showmeans=False,
                                          widths=.1, points=10)
                    for pc in parts['bodies']:
                        pc.set_facecolor('black')
                        pc.set_edgecolor('black')
                        pc.set_alpha(.5)
                _ = ax.scatter(2 * day + .1 * k, data.mean(), color='black')
        fig.suptitle(stat_key)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        return fig, ax



    def combined_hist(self, stat_key, smooth=True, cumulative=False, bins= None, sigma = .1, fill = True):
        '''

        :param stat_key:
        :param smooth:
        :param cumulative:
        :param bins:
        :param sigma:
        :return:
        '''

        fig, ax = plt.subplots(1, 6, figsize=[30, 5], sharey=True)

        def concat_stats(stat_dict):
            stat = {day: [] for day in self.days}
            mice = list(stat_dict.keys())
            for mouse in stat_dict.keys():
                for day in self.days:
                    _data = stat_dict[mouse][day][stat_key]
                    stat[day].append(_data[~np.isnan(_data)])

            for k in stat.keys():
                stat[k] = np.concatenate(stat[k]).ravel()

            return stat

        ko_ravel_stat = concat_stats(self.ko_stats)
        ctrl_ravel_stat = concat_stats(self.ctrl_stats)

        if bins is None:
            _min = np.minimum(np.amin(ko_ravel_stat[self.days[0]]), np.amin(ctrl_ravel_stat[self.days[0]]))
            _max = np.maximum(np.amax(ko_ravel_stat[self.days[0]]), np.amax(ctrl_ravel_stat[self.days[0]]))
            bins = np.linspace(_min, _max)

        for d, day in enumerate(self.days):
            if smooth:
                ctrl_hist = tpu.utilities.gaussian(ctrl_ravel_stat[day][:, np.newaxis], sigma,
                                                 bins[np.newaxis, :]).mean(axis=0)
                ctrl_hist /= ctrl_hist.sum()

                ko_hist = tpu.utilities.gaussian(ko_ravel_stat[day][:,np.newaxis], sigma,
                                                 bins[np.newaxis,:] ).mean(axis=0)
                ko_hist /= ko_hist.sum()
                if cumulative:
                    if fill:
                        ax[d].fill_between(bins, np.cumsum(ctrl_hist), color='black', alpha =.3)
                        ax[d].fill_between(bins, np.cumsum(ko_hist), color='red', alpha = .3)
                    else:
                        ax[d].plot(bins, np.cumsum(ctrl_hist), color='black')
                        ax[d].plot(bins, np.cumsum(ko_hist), color='red')
                else:
                    if fill:
                        ax[d].fill_between(bins, ctrl_hist, color='black', alpha=.3)
                        ax[d].fill_between(bins,ko_hist, color = 'red', alpha = .3)
                    else:
                        ax[d].plot(bins, ctrl_hist, color='black')
                        ax[d].plot(bins, ko_hist, color='red')

            else:
                ax[d].hist(ctrl_ravel_stat[day], bins=bins, color='black', alpha=.3,
                           cumulative=cumulative, density=True, fill = fill)
                ax[d].hist(ko_ravel_stat[day], bins=bins, color='red', alpha=.3,
                           cumulative=cumulative, density=True, fill = fill)

            ax[d].spines['top'].set_visible(False)
            ax[d].spines['right'].set_visible(False)
            fig.suptitle(stat_key)

        return fig, ax


    def mixed_anova(self, stat_key, verbose = True, group_tukey = True, day_tukey = True):
        '''

        :param stat_key:
        :return:
        '''

        ko_sum_stat, ctrl_sum_stat = self.summary_stat_matrices(stat_key)

        df = {'ko_ctrl': [],
              'day': [],
              stat_key: [],
              'mouse': []}

        for mouse in range(len(self.ko_mice)):
            for day in self.days:
                df['ko_ctrl'].append(0)
                df['day'].append(day)
                df[stat_key].append(ko_sum_stat[mouse, day])
                df['mouse'].append(mouse)

        for mouse in range(len(self.ctrl_mice)):
            for day in self.days:
                df['ko_ctrl'].append(1)
                df['day'].append(day)
                df[stat_key].append(ctrl_sum_stat[mouse, day])
                df['mouse'].append(mouse + 5)

        df = pd.DataFrame(df)
        results = {}
        aov = mixed_anova(data=df, dv=stat_key, between='ko_ctrl', within='day', subject='mouse')
        results['anova'] = aov
        if verbose:
            print('Mixed design ANOVA results')
            print(aov)

        if group_tukey:
            ko_ctrl_tukey = pairwise_tukey(data=df, dv=stat_key, between='ko_ctrl')
            results['ko_ctrl_tukey'] = ko_ctrl_tukey
            if verbose:
                print('PostHoc Tukey: KO vs Ctrl')
                print(ko_ctrl_tukey)

        if day_tukey:
            day_stats = []
            print('PostHov Tukey on each day')
            for day in self.days:
                print('Day %d' % day)
                stats = pairwise_tukey(data=df[df['day'] == day], dv=stat_key, between='ko_ctrl')
                day_stats.append(stats)
                if verbose:
                    print(stats)
            results['day_tukey'] = day_stats

        return results

    def across_trial_plot(self, stat_key, max_trial = 140):
        '''

        :return:
        '''

        x = np.arange(0, max_trial)

        def make_plot_array(stat_dict):
            '''

            :param stat_dict:
            :return:
            '''
            mice = stat_dict.keys()
            mu_arr, sem_arr = np.zeros([len(mice), self.n_days, x.shape[0]])*np.nan, np.zeros([len(mice), self.n_days, x.shape[0]])*np.nan
            for m, mouse in enumerate(mice):
                for d, day in enumerate(self.days):
                    stat = np.squeeze(stat_dict[mouse][day][stat_key])
                    mu_arr[m, d, :stat.shape[0]], sem_arr[m, d, :stat.shape[0]] = stat.mean(axis=-1), sp.stats.sem(stat, axis=-1)
            return mu_arr, sem_arr

        fig, ax = plt.subplots(2, 6, figsize=[5*self.n_days, 10], sharey=True)
        ko_mu_arr, ko_sem_arr = make_plot_array(self.ko_stats)

        ctrl_mu_arr, ctrl_sem_arr = make_plot_array(self.ctrl_stats)

        for m in range(ctrl_mu_arr.shape[0]):
            for d in range(ctrl_mu_arr.shape[1]):
                ax[0,d].fill_between(x, ctrl_mu_arr[m, d,:]-ctrl_sem_arr[m, d, :], ctrl_mu_arr[m, d, :] + ctrl_sem_arr[m, d, :],
                                     color = 'black', alpha = .3)

        for m in range(ko_mu_arr.shape[0]):
            for d in range(ko_mu_arr.shape[1]):
                ax[0,d].fill_between(x, ko_mu_arr[m, d,:]-ko_sem_arr[m, d, :], ko_mu_arr[m, d, :] + ko_sem_arr[m, d, :],
                                     color = 'red', alpha = .3)

        ko_mu, ko_sem = np.nanmean(ko_mu_arr, axis=0), sp.stats.sem(ko_mu_arr, axis=0, nan_policy='omit')
        ctrl_mu, ctrl_sem = np.nanmean(ctrl_mu_arr, axis=0), sp.stats.sem(ctrl_mu_arr, axis=0, nan_policy='omit')
        for d in range(self.n_days):
            ax[1, d].fill_between(x, ctrl_mu[d, :] - ctrl_sem[d, :], ctrl_mu[d, :] + ctrl_sem[d, :], color='black',
                                  alpha=.3)
            ax[1,d].fill_between(x, ko_mu[d,:] - ko_sem[d,:], ko_mu[d,:] + ko_sem[d,:], color = 'red', alpha = .3)

        for row in range(2):
            for col in range(self.n_days):
                ax[row,col].spines['top'].set_visible(False)
                ax[row,col].spines['right'].set_visible(False)

                ax[row, col].set_xlabel('Trial #')
                ax[row, col].set_ylabel(stat_key)
        fig.subplots_adjust(wspace = .2, hspace = .2)
        return fig, ax