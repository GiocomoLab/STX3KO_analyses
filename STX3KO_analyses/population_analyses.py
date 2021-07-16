# import numpy as np
# import scipy as sp
#
# from . import ymaze_sess_deets
# from . import utilities as u
# from . import session
#
# class CumCa:
#
#     def __init__(self, days=np.arange(6), ts_key='spks', fam=True, place_cells = False):
#         '''
#
#         :param days:
#         :param ts_key:
#         :param fam:
#         '''
#
#         self.ko_mice = ymaze_sess_deets.ko_mice
#         self.ctrl_mice = ymaze_sess_deets.ctrl_mice
#         self.__dict__.update({'days': days, 'ts_key': ts_key, 'fam': fam})
#         self.n_days = days.shape[0]
#
#
#         get_cum_ca = u.loop_func_over_days(self.argmax_perireward, days, ts_key=ts_key, fam=fam)
#
#         # self.ko_frac = {mouse: get_pc_max(mouse) for mouse in self.ko_mice}
#         # self.ctrl_frac = {mouse: get_pc_max(mouse) for mouse in self.ctrl_mice}
#
#         self.ko_sums = None
#         self.ctrl_sums = None
#
#         self.ko_plot_array = None
#         self.ctrl_plot_array = None
#
#     @staticmethod
#     def cum_ca(sess: session.YMazeSession, ts_key: str = 'spks'):
#
#         def _calc_cum_ca
#         starts, teleports = sess.trial_start_inds[trialmask], sess.teleport_inds[trialmask]
#         cum_ca = 0
#         for start, stop in zip(starts, teleports):
#             ts = sess.timeseries[key][:, start:stop].mean(axis=-1)
#             if place_cells:
#                 cum_ca += ts[cellmask]
#             elif non_place_cells:
#                 cum_ca += ts[~cellmask]
#             else:
#                 cum_ca += ts
#
#
# def _calc_simmat(trials_mat, popvec=False, metric="corr"):
#     if popvec:
#         trials_t = np.transpose(trials_mat, axes=(1, 0, 2))  # positions x trials x cells
#     else:
#         trials_t = np.transpose(trials_mat, axes=(2, 0, 1))  # cells x trials x positions
#
#     assert metric in ("corr", "cos"), "invalid metric, metric must be 'corr' or 'cos' "
#     if metric == "corr":
#         trials_norm = sp.stats.zscore(trials_t, axis=2)
#         return 1. / trials_t.shape[2] * np.matmul(trials_norm, np.transpose(trials_norm, axes=(0, 2, 1)))
#     if metric == "cos":
#         trials_norm = trials_t / (np.linalg.norm(trials_t, axis=2, ord=2, keepdims=True) + 1E-3)
#         return np.matmul(trials_norm, np.transpose(trials_norm, axes=(0, 2, 1)))
#
#
# def _single_sess_trial_sim(sess, trial_mask, key="spks_norm", smooth_sigma=1, nanless=True, cell_mask=None, **kwargs):
#     trials_mat = sess.trial_matrices[key][trial_mask, :, :]
#     if cell_mask is not None:
#         trials_mat = trials_mat[:, :, cell_mask]
#
#     if nanless:
#         trials_mat[np.isnan(trials_mat)] = 1E-5
#
#     if smooth_sigma > 0:
#         trials_mat = sp.ndimage.filters.gaussian_filter1d(trials_mat, smooth_sigma, axis=1)
#
#     return _calc_simmat(trials_mat, **kwargs)
#
# def _average_triu(sm):
#     ''' sm : cells/positions x trials x trials'''
#     iu = np.triu_indices(sm.shape[1], k=1)
#     return sm[:, iu[0], iu[1]].mean(axis=-1)
#
#
# def _trial_sim_violinplots(ko_sim_dict, ctrl_sim_dict):
#     fig, ax = plt.subplots(figsize=[15, 5])
#
#     n_days = len(ko_sim_dict[ko_mice[0]])
#     ko_means = np.zeros([n_days, len(ko_mice)])
#     for k, (mouse, data_list) in enumerate(ko_sim_dict.items()):
#         for day, data in enumerate(data_list):
#
#             ko_means[day, k] = data.mean()
#             parts = ax.violinplot(data, positions=[2 * day + .1 * k], showextrema=False, showmeans=False, widths=.1,
#                                   points=10)
#             _ = ax.scatter(2 * day + .1 * k, data.mean(), color='red')
#             for pc in parts['bodies']:
#                 pc.set_facecolor('red')
#                 pc.set_edgecolor('black')
#                 pc.set_alpha(.5)
#
#     ctrl_means = np.zeros([n_days, len(ctrl_mice)])
#     for k, (mouse, data_list) in enumerate(ctrl_sim_dict.items()):
#         for day, data in enumerate(data_list):
#             ctrl_means[day, k] = data.mean()
#             parts = ax.violinplot(data, positions=[2 * day + .6 + .1 * k], showextrema=False, showmeans=False,
#                                   widths=.1, points=10)
#             _ = ax.scatter(2 * day + .6 + .1 * k, data.mean(), color='black')
#             for pc in parts['bodies']:
#                 pc.set_facecolor('black')
#                 pc.set_edgecolor('black')
#                 pc.set_alpha(.5)
#
#     t, p = sp.stats.ttest_ind(ko_means, ctrl_means, axis=1)
#     return (fig, ax), ko_means, ctrl_means, (t, p)
