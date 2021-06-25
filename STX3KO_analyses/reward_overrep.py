from . import session
from . import utilities as u
from . import ymaze_sess_deets

from matplotlib import pyplot as plt

import scipy as sp
import numpy as np
import pandas as pd

from pingouin import mixed_anova, pairwise_tukey


class PeriRewardPlaceCellFrac:

    def __init__(self, days=np.arange(6), ts_key='spks', fam=True):
        '''

        :param days:
        :param ts_key:
        :param fam:
        '''
        self.ko_mice = ymaze_sess_deets.ko_mice
        self.ctrl_mice = ymaze_sess_deets.ctrl_mice
        self.__dict__.update({'days': days, 'ts_key': ts_key, 'fam': fam})
        self.n_days = days.shape[0]

        get_pc_max = u.loop_func_over_days(self.argmax_perireward, days, ts_keys=ts_key, fam=fam)

        self.ko_frac = {mouse: get_pc_max(mouse) for mouse in self.ko_mice}
        self.ctrl_frac = {mouse: get_pc_max(mouse) for mouse in self.ctrl_mice}

        self.ko_sums = None
        self.ctrl_sums = None

        self.ko_plot_array = None
        self.ctrl_plot_array = None

    @staticmethod
    def argmax_perireward(sess: session.YMazeSession, ts_key: str = 'spks', fam: bool = True):
        '''

        :param sess:
        :param ts_key:
        :param fam:
        :return:
        '''

        trials_mat = sess.trial_matrices[ts_key]
        bin_edges = sess.trial_matrices['bin_edges']
        if fam:
            trial_mask = sess.trial_info['LR'] == -1 * sess.novel_arm
            cell_mask = sess.fam_place_cell_mask()
            rzone_front = np.argwhere((sess.rzone_fam['tfront'] <= bin_edges[1:]) * \
                                      (sess.rzone_fam['tfront'] >= bin_edges[:-1]))[0][0]

        else:
            trial_mask = sess.trial_info['LR'] == sess.novel_arm
            cell_mask = sess.nov_place_cell_mask()
            rzone_front = np.argwhere((sess.rzone_nov['tfront'] <= bin_edges[1:]) * \
                                      (sess.rzone_nov['tfront'] >= bin_edges[:-1]))[0][0]

        # smooth ratemap by 1 bin
        ratemap = sp.ndimage.filters.gaussian_filter1d(np.nanmean(trials_mat[trial_mask, :, :], axis=0), 1, axis=0)

        return np.argmax(ratemap[:, cell_mask], axis=0) - rzone_front

    def perireward_hist(self):
        '''

        :param ko_frac:
        :param ctrl_frac:
        :param sigma:
        :return:
        '''

        fig, ax = plt.subplots(2, self.n_days, figsize=[self.n_days * 5, 10], sharey=True)

        x = np.arange(-30, 15)
        anova_mask = (x > -5) * (x <= -1)
        plot_mask = (x >= -10) * (x <= 1)

        def get_hist(frac):
            '''

            :param frac:
            :return:
            '''
            plot_array = np.zeros([len(frac.keys()), self.n_days, int(plot_mask.sum())])
            sums = np.zeros([len(frac.keys()), self.n_days])
            for m, (mouse, data_list) in enumerate(frac.items()):
                for col, data in enumerate(data_list):
                    hist = np.array([np.count_nonzero(data.ravel() == _bin) for _bin in x.tolist()])
                    hist_sm = sp.ndimage.filters.gaussian_filter1d(hist, 1)
                    hist = hist / hist.sum()
                    hist_sm = hist_sm / hist_sm.sum()

                    sums[m, col] = hist[anova_mask].sum() / hist[~anova_mask].sum()
                    plot_array[m, col, :] = hist_sm[plot_mask]
            return sums, plot_array

        self.ko_sums, self.ko_plot_array = get_hist(self.ko_frac)
        self.ctrl_sums, self.ctrl_plot_array = get_hist(self.ctrl_frac)

        for day in range(self.n_days):
            ax[0, day].plot(x[plot_mask], self.ko_plot_array[:, day, :].T, color='red')
            ko_mu, ko_sem = self.ko_plot_array[:, day, :].mean(axis=0), sp.stats.sem(self.ko_plot_array[:, day, :])
            ax[1, day].fill_between(x[plot_mask], ko_mu - ko_sem, ko_mu + ko_sem, color='red', alpha=.3)

            ax[0, day].plot(x[plot_mask], self.ctrl_plot_array[:, day, :].T, color='black')
            ctrl_mu, ctrl_sem = self.ctrl_plot_array[:, day, :].mean(axis=0), sp.stats.sem(
                self.ctrl_plot_array[:, day, :])
            ax[1, day].fill_between(x[plot_mask], ctrl_mu - ctrl_sem, ctrl_mu + ctrl_sem, color='black', alpha=.3)

            for row in range(2):
                ax[row, day].set_ylim([.0, .075])
                ax[row, day].set_xlim([-10, 1])

                ax[row, day].spines['top'].set_visible(False)
                ax[row, day].spines['right'].set_visible(False)

                ax[row, day].set_title("Day %d" % (day + 1))
                ax[row, day].set_xlabel("Distance from reward")
        ax[0, 0].set_ylabel('% of cells')
        ax[1, 0].set_ylabel('% of cells')

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
              'frac': [],
              'mouse': []}

        for mouse in range(len(self.ko_mice)):
            for day in self.days:
                df['ko_ctrl'].append(0)
                df['day'].append(day)
                df['frac'].append(self.ko_sums[mouse, day])
                df['mouse'].append(mouse)

        for mouse in range(len(self.ctrl_mice)):
            for day in self.days:
                df['ko_ctrl'].append(1)
                df['day'].append(day)
                df['frac'].append(self.ctrl_sums[mouse, day])
                df['mouse'].append(mouse + 5)

        df = pd.DataFrame(df)
        results = {}
        aov = mixed_anova(data=df, dv='frac', between='ko_ctrl', within='day', subject='mouse')
        results['anova'] = aov
        if verbose:
            print('Mixed design ANOVA results')
            print(aov)

        if group_tukey:
            ko_ctrl_tukey = pairwise_tukey(data=df, dv='frac', between='ko_ctrl')
            results['ko_ctrl_tukey'] = ko_ctrl_tukey
            if verbose:
                print('PostHoc Tukey: KO vs Ctrl')
                print(ko_ctrl_tukey)

        if day_tukey:
            day_stats = []
            print('PostHov Tukey on each day')
            for day in self.days:
                print('Day %d' % day)
                stats = pairwise_tukey(data=df[df['day'] == day], dv='frac', between='ko_ctrl')
                day_stats.append(stats)
                if verbose:
                    print(stats)
            results['day_tukey'] = day_stats

        return results


def plot_leftright_crossval_placecells_withinday(sess_list):
    pass


def plot_leftright_crossval_placecells_acrossdays(mice):
    pass
