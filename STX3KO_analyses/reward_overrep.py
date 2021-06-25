from . import session

import scipy as sp
import numpy as np


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

def perireward_hist(ko_frac, ctrl_frac, sigma = 1):
    '''

    :param ko_frac:
    :param ctrl_frac:
    :param sigma:
    :return:
    '''




def plot_leftright_crossval_placecells_withinday(sess_list):
    pass


def plot_leftright_crossval_placecells_acrossdays(mice):
    pass
