from . import session
from . import utilities as u
from . import ymaze_sess_deets

from matplotlib import pyplot as plt

import scipy as sp
import numpy as np
import pandas as pd


from pingouin import mixed_anova, pairwise_ttests

class LMM_PeriRewardPlaceCellFrac:

    def __init__(self, days = np.arange(5), ts_key = 'spks'):

        '''


        :param days:
        :param ts_key:
        '''

        self.ko_mice = ymaze_sess_deets.ko_mice
        self.ctrl_mice = ymaze_sess_deets.ctrl_mice
        self.__dict__.update({'days': days, 'days_z': sp.stats.zscore(days), 'ts_key': ts_key, 'fam': fam})
        self.n_days = days.shape[0]

        self.df = pd.DataFrame({'mouse':[],'ko':[],'day':[],'lr':[], 'novfam':[], 'frac': []})
        self.fill_df()


    def fill_df(self):
        for mouse in self.ko_mice:
            for day, dz in zip(self.days,self.days_z):
                self.argmax_perireward(u.load_single_day(mouse, day), 1, dz)




    def argmax_perireward(self, sess: session.YMazeSession, ko, dz, ts_key: str = 'spks'):
        '''

        :param sess:
        :param ts_key:
        :param fam:
        :return:
        '''


        trials_mat = sess.trial_matrices[ts_key]
        bin_edges = sess.trial_matrices['bin_edges']


        for arm in [-1, 1]:
            trial_mask = sess.trial_info['lr']==arm
            if sess.novel_arm == arm:
                cell_mask = sess.nov_place_cell_mask()
                rzone_front = np.argwhere((sess.rzone_nov['tfront'] <= bin_edges[1:]) * \
                                          (sess.rzone_nov['tfront'] >= bin_edges[:-1]))[0][0]
                nov = 1
            else:
                cell_mask = sess.fam_place_cell_mask()
                rzone_front = np.argwhere((sess.rzone_fam['tfront'] <= bin_edges[1:]) * \
                                          (sess.rzone_fam['tfront'] >= bin_edges[:-1]))[0][0]
                nov = 0



            # smooth ratemap by 1 bin
            ratemap = sp.ndimage.filters.gaussian_filter1d(np.nanmean(trials_mat[trial_mask, :, :], axis=0), 1, axis=0)
            max_inds = np.argmax(ratemap[:,cell_mask], axis = 0) - rzone_front
            reward_frac = self.get_frac(max_inds)

            self.df.append({'mouse': sess.mouse,
                            'ko': ko,
                            'day': dz,
                            'lr': arm,
                            'novfam': nov,
                            'frac': reward_frac}, ignore_index=True)



    @staticmethod
    def get_frac(data):
        '''

        :param frac:
        :return:
        '''
        x = np.arange(-30, 15)
        anova_mask = (x > -5) * (x <= 0)

        hist = np.array([np.count_nonzero(data.ravel() == _bin) for _bin in x.tolist()])
        hist = hist / hist.sum()
        return hist[anova_mask].sum() / hist[~anova_mask].sum()



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

        get_pc_max = u.loop_func_over_days(self.argmax_perireward, days, ts_key=ts_key, fam=fam)

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
        ratemap = sp.ndimage.gaussian_filter1d(np.nanmean(trials_mat[trial_mask, :, :], axis=0), 1, axis=0)

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
        # x = np.arange(-60, 30)
        # anova_mask = (x > -10) * (x <= -1)
        # plot_mask = (x >= -20) * (x <= 1)

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
            # ax[0, day].plot(x[plot_mask], self.ko_plot_array[:, day, :].T, color='red')
            ko_mu, ko_sem = self.ko_plot_array[:, day, :].mean(axis=0), sp.stats.sem(self.ko_plot_array[:, day, :])
            ax[1, day].fill_between(x[plot_mask], ko_mu - ko_sem, ko_mu + ko_sem, color='red', alpha=.3)

            # ax[0, day].plot(x[plot_mask], self.ctrl_plot_array[:, day, :].T, color='black')
            ctrl_mu, ctrl_sem = self.ctrl_plot_array[:, day, :].mean(axis=0), sp.stats.sem(
                self.ctrl_plot_array[:, day, :])
            ax[1, day].fill_between(x[plot_mask], ctrl_mu - ctrl_sem, ctrl_mu + ctrl_sem, color='black', alpha=.3)

            for row in range(2):
                # ax[row, day].set_ylim([.0, .075])
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

        # if group_tukey:
        #     ko_ctrl_tukey = pairwise_tukey(data=df, dv='frac', between='ko_ctrl')
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
        #         stats = pairwise_tukey(data=df[df['day'] == day], dv='frac', between='ko_ctrl')
        #         day_stats.append(stats)
        #         if verbose:
        #             print(stats)
        #     results['day_tukey'] = day_stats

        return results


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


class PeriRewardSpeed:

    def __init__(self, days=np.arange(6), ts_key='speed', fam=True):
        '''


        '''
        self.ko_mice = ymaze_sess_deets.ko_mice
        self.ctrl_mice = ymaze_sess_deets.ctrl_mice
        self.__dict__.update({'days': days, 'ts_key': ts_key, 'fam': fam})
        self.n_days = days.shape[0]

        get_speed = u.loop_func_over_days(self.speed_perireward, days, ts_key='speed', fam=fam)

        self.ko_speed = {mouse: get_speed(mouse) for mouse in self.ko_mice}
        self.ctrl_speed = {mouse: get_speed(mouse) for mouse in self.ctrl_mice}

        self.ko_sums = None
        self.ctrl_sums = None

        self.ko_plot_mu = None
        self.ctrl_plot_mu = None

        self.ko_plot_sem = None
        self.ctrl_plot_sem = None


    @staticmethod
    def speed_perireward(sess: session.YMazeSession, ts_key: str = 'speed', fam: bool = True, xbounds=(-10, 3)):

        bin_edges = sess.trial_matrices['bin_edges']
        if fam:
            trial_mask = sess.trial_info['LR'] == -1 * sess.novel_arm
            rzone_front = np.argwhere((sess.rzone_fam['tfront'] <= bin_edges[1:]) * \
                                      (sess.rzone_fam['tfront'] >= bin_edges[:-1]))[0][0]
        else:
            trial_mask = sess.trial_info['LR'] == sess.novel_arm
            rzone_front = np.argwhere((sess.rzone_nov['tfront'] <= bin_edges[1:]) * \
                                      (sess.rzone_nov['tfront'] >= bin_edges[:-1]))[0][0]

        speedmat = sess.trial_matrices[ts_key][trial_mask, :]
        speedmat[np.isnan(speedmat)]=1E-3
        speedmat = sp.stats.zscore(speedmat, axis = -1)

        return speedmat[:, rzone_front + xbounds[0]:rzone_front + xbounds[1]]



    def perireward_plot(self):

        fig, ax = plt.subplots(2, self.n_days, figsize=[self.n_days * 5, 10], sharey=True)

        x = np.arange(-10, 3)
        anova_mask = (x > -5) * (x <= -1)

        # plot_mask = (x >= -10) * (x <= 1)

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
                    mu, sem = data.mean(axis=0), sp.stats.sem(data, axis=0)

                    sums[m, col] = mu[anova_mask].mean()
                    plot_mu[m, col, :] = mu
                    plot_sem[m, col, :] = sem
            return sums, plot_mu, plot_sem

        self.ko_sums, self.ko_plot_mu, self.ko_plot_sem = get_ratemap_sum(self.ko_speed)
        self.ctrl_sums, self.ctrl_plot_mu, self.ctrl_plot_sem = get_ratemap_sum(self.ctrl_speed)

        for day in range(self.n_days):
            for m in range(len(self.ko_mice)):
                ax[0, day].fill_between(x, self.ko_plot_mu[m, day, :] - self.ko_plot_sem[m, day, :],
                                        self.ko_plot_mu[m, day, :] + self.ko_plot_sem[m, day, :], color='red', alpha=.3)
            for m in range(len(self.ctrl_mice)):
                ax[0, day].fill_between(x, self.ctrl_plot_mu[m, day, :] - self.ctrl_plot_sem[m, day, :],
                                        self.ctrl_plot_mu[m, day, :] + self.ctrl_plot_sem[m, day, :], color='black',
                                        alpha=.3)

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
        ax[0, 0].set_ylabel('Norm. Speed')
        ax[1, 0].set_ylabel('Norm. Speed')

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
                df['mouse'].append(mouse + 5)

        df = pd.DataFrame(df)
        results = {}
        aov = mixed_anova(data=df, dv='sum', between='ko_ctrl', within='day', subject='mouse')
        results['anova'] = aov
        if verbose:
            print('Mixed design ANOVA results')
            print(aov)

        if group_tukey:
            ko_ctrl_tukey = pairwise_tukey(data=df, dv='sum', between='ko_ctrl')
            results['ko_ctrl_tukey'] = ko_ctrl_tukey
            if verbose:
                print('PostHoc Tukey: KO vs Ctrl')
                print(ko_ctrl_tukey)

        if day_tukey:
            day_stats = []
            print('PostHov Tukey on each day')
            for day in self.days:
                print('Day %d' % day)
                stats = pairwise_tukey(data=df[df['day'] == day], dv='sum', between='ko_ctrl')
                day_stats.append(stats)
                if verbose:
                    print(stats)
            results['day_tukey'] = day_stats

        return results


def plot_leftright_crossval_placecells_withinday(day, ts_key = 'spks', vmin = -.25, vmax = 5):
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
        l_rm, r_rm = [], []
        for mouse in mice:
            sess = u.load_single_day(mouse, day)
            if 'left' in sess.place_cell_info.keys():
                l_cellmask = sess.place_cell_info['left']['masks']
                r_cellmask= sess.place_cell_info['left']['masks']
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

            l_rm.append(np.nanmean(l_trialmat, axis=0))
            r_rm.append(np.nanmean(r_trialmat, axis=0))

            l_rm_train.append(np.nanmean(l_trialmat[::2, :, :], axis=0))
            l_rm_test.append(np.nanmean(l_trialmat[1::2, :, :], axis=0))

            r_rm_train.append(np.nanmean(r_trialmat[::2, :, :], axis=0))
            r_rm_test.append(np.nanmean(r_trialmat[1::2, :, :], axis=0))

        return np.concatenate(l_rm, axis=-1), np.concatenate(r_rm, axis=-1), \
               np.concatenate(l_rm_train, axis=-1), np.concatenate(l_rm_test, axis=-1), \
               np.concatenate(r_rm_train, axis=-1), np.concatenate(r_rm_test, axis=-1)

    def sort_norm(rm_train, rm_test):
        mu, std = np.nanmean(rm_train, axis=0, keepdims=True), np.nanstd(rm_train, axis=0, keepdims=True)
        sortvec = np.argsort(np.argmax(rm_train, axis=0))

        rm_test = (rm_test-mu)/std

        return rm_test[:, sortvec]

    ko_l, ko_r, ko_l_train, ko_l_test, ko_r_train, ko_r_test = lr_ratemaps(ymaze_sess_deets.ko_mice)
    ctrl_l, ctrl_r, ctrl_l_train, ctrl_l_test, ctrl_r_train, ctrl_r_test = lr_ratemaps(ymaze_sess_deets.ctrl_mice)

    fig, ax = plt.subplots(4,2, figsize= [10,20])
    ax[0, 0].imshow(sort_norm(ctrl_l_train, ctrl_l_test).T, cmap='pink', aspect='auto', vmin=vmin, vmax=vmax)
    ax[0, 1].imshow(sort_norm(ctrl_l, ctrl_r).T, cmap='pink', aspect='auto', vmin=vmin, vmax=vmax)
    # ax[1, 0].imshow(sort_norm(ctrl_r_train, ctrl_r_test).T, cmap='pink', aspect='auto', vmin=vmin, vmax=vmax)

    # ax[0, 0].plot([-.5, ctrl_l_train.shape[0]- .5], [-.5, ctrl_l_train.shape[1]-.5], color='blue')
    # ax[0, 1].plot([-.5, ctrl_r_train.shape[0] - .5], [-.5, ctrl_r_train.shape[1] - .5], color='blue')

    # ax[0, 0].set_title("mCherry: Left, N cells %d" % ctrl_l_test.shape[1])
    # ax[0, 1].set_title("mCherry: Right, N cells %d" % ctrl_r_test.shape[1])

    # ax[1, 0].imshow(sort_norm(ko_l_train, ko_l_test).T, cmap = 'pink', aspect = 'auto', vmin=vmin, vmax=vmax)
    # ax[1, 1].imshow(sort_norm(ko_r_train, ko_r_test).T, cmap='pink', aspect='auto', vmin=vmin, vmax=vmax)

    # ax[1, 0].plot([-.5, ko_l_train.shape[0] - .5], [-.5, ko_l_train.shape[1] - .5], color='blue')
    # ax[1, 1].plot([-.5, ko_r_train.shape[0] - .5], [-.5, ko_r_train.shape[1] - .5], color='blue')

    # ax[1, 0].set_title("Cre: Left, N cells %d" % ko_l_train.shape[1])
    # ax[1, 1].set_title("Cre: Right, N cells %d" % ko_r_train.shape[1])

    for row in [0,1]:
        for col in [0,1]:
            ax[row,col].set_yticks([])
            ax[row,col].set_ylabel('Cells')
            ax[row, col].set_xlabel('Pos')

    fig.subplots_adjust(hspace=.25, wspace=.5)
    fig.suptitle('Day %d' % day)
    return fig, ax




def plot_leftright_crossval_placecells_acrossdays(mice):
    pass
