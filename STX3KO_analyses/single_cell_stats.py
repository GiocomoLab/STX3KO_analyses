import numpy as np
import scipy as sp
from matplotlib import pyplot as plt

from . import spatial_analyses
from . import ymaze_sess_deets
from . import utilities as u

class CellStats:
    # TODO: continue modularizing InfieldVsOutOfField.ipynb here
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
        else:
            cell_mask = sess.mpv_place_cell_mask()
            trial_mask = sess.trial_info['LR'] == sess.novel_arm

        trial_mat = sess.trial_matrices[ts_key][:, :, cell_mask]
        trial_mat = trial_mat[trial_mask, :, :]

        avg_trial_mat = np.nanmean(trial_mat, axis=0)
        avg_trial_mat[np.isnan(avg_trial_mat)] = 1E-5

        inds = np.arange(0, trial_mat.shape[1])[np.newaxis, :, np.newaxis]

        avg_trial_mat_norm = avg_trial_mat / (np.nansum(avg_trial_mat, axis=0, keepdims=True) + 1E-5)
        # center of mass / expected value
        avg_com = (avg_trial_mat_norm * inds).sum(axis=0, keepdims=True)
        # spatial standard deviation
        avg_std = np.power((np.power(inds - com, 2) * avg_trial_mat_norm).sum(axis=0, keepdims=True), .5)
        avg_skewness = (np.power((inds - avg_com) / (avg_std + 1E-5), 3) * avg_trial_mat_norm).sum(axis=1)
        avg_kurtosis = (np.power((inds - avg_com) / (avg_std + 1E-5), 4) * avg_trial_mat_norm).sum(axis=1)

        trial_mat[np.isnan(trial_mat)] = 1E-5

        trial_mat_sm = sp.ndimage.filters.gaussian_filter1d(trial_mat, 3, axis=0)
        trial_mat_sm_norm = trial_mat_sm / (np.nansum(trial_mat_sm, axis=1, keepdims=True) + 1E-5)

        com = (trial_mat_sm_norm * inds).sum(axis=1, keepdims=True)
        std = np.power((np.power(inds - com, 2) * trial_mat_sm_norm).sum(axis=1, keepdims=True), .5)

        skewness = (np.power((inds - com) / (std + 1E-5), 3) * trial_mat_sm_norm).sum(axis=1)
        kurtosis = (np.power((inds - com) / (std + 1E-5), 4) * trial_mat_sm_norm).sum(axis=1)



        return {
                'std': std,
                'skewness': skewness,
                'kurtosis': kurtosis,
                'avg_std': avg_std.ravel(),
                'avg_skewness': avg_skewness.ravel(),
                'avg_kurtosis': avg_kurtosis.ravel(),
                'max_counts': spatial_analyses.max_counts(avg_trial_mat),
                'field_width': spatial_analyses.field_width(avg_trial_mat),
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
                    means[day, k] = np.nanmean(stat_dict[key])
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
                if not scatter_only:
                    parts = ax.violinplot(data, positions=[2 * day + .6 * k], showextrema=False, showmeans=False,
                                          widths=.1, points=10)
                    for pc in parts['bodies']:
                        pc.set_facecolor('red')
                        pc.set_edgecolor('black')
                        pc.set_alpha(.5)
                _ = ax.scatter(2 * day + .6 * k, data.mean(), color='red')

        for k, (mouse, data_list) in enumerate(self.ctrl_stats.items()):
            for day, data_dict in enumerate(data_list):
                data = data_dict[stat_key]
                if not scatter_only:
                    parts = ax.violinplot(data, positions=[2 * day + .1 * k], showextrema=False, showmeans=False,
                                          widths=.1, points=10)
                    for pc in parts['bodies']:
                        pc.set_facecolor('red')
                        pc.set_edgecolor('black')
                        pc.set_alpha(.5)
                _ = ax.scatter(2 * day + .1 * k, data.mean(), color='black')
        fig.suptitle(stat_key)
        return fig, ax



    def combined_hist(self, smooth=True, cumulative=False, bins= None):
        pass


    def mixed_anova(self, key):
        pass


    def across_trial_plot(self):
        pass