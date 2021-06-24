import numpy as np
import scipy as sp

from TwoPUtils.spatial_analyses import *


def field_width(avg_trial_mat):
    '''
    calculate width of largest magnitude field

    :param trial_mat: trials x positions x cells
    :return: fw: field width in position bins [cells,], nan for fields at edge of track
    '''

    # normalize rate map to be 0-1
    maxnorm = np.copy(avg_trial_mat)
    _max, _min = np.amax(maxnorm, axis=0, keepdims=True), np.amin(maxnorm, axis=0, keepdims=True)
    maxnorm = (maxnorm - _min) / (_max - _min + 1E-5)
    maxinds = np.argmax(maxnorm, axis=0)


    # find indices of half max
    half_mask = maxnorm <= .5
    fw = np.zeros(avg_trial_mat.shape[-1])*np.nan
    for cell in range(maxnorm.shape[1]):
        left = np.argwhere(half_mask[:maxinds[cell], cell])
        right = np.argwhere(half_mask[maxinds[cell]:, cell])
        if left.shape[0] > 0 and right.shape[0] > 0:
            ledge, redge = left[-1][0], right[0][0] + maxinds[cell]
            fw[cell] = redge - ledge
    return fw


def max_counts(avg_trial_mat, mean_norm_thresh = 1):
    '''
    get number of local maxima in avg rate map

    :param avg_trial_mat: [pos x cells] rate map
    :param mean_norm_thresh: threshold for finding peaks
    :return: max_counts: [cells,] number of maxima for each cell
    '''

    # mean normalize
    avg_trial_mat /= avg_trial_mat.mean(axis=0, keepdims=True)
    # find number of maxima
    max_counts = []
    for cell in range(avg_trial_mat.shape[-1]):
        extm, _ = sp.signal.find_peaks(avg_trial_mat[:, cell], height=mean_norm_thresh)
        max_counts.append(extm.shape[0])

    return np.array(max_counts)


def spatial_std(avg_trial_mat):
    '''

    :param avg_trial_mat:
    :return:
    '''
    avg_trial_mat_norm = avg_trial_mat / (np.nansum(avg_trial_mat, axis=0, keepdims=True) + 1E-5)
    inds = np.arange(0, avg_trial_mat.shape[0])[:, np.newaxis]

    com = (avg_trial_mat_norm * inds).sum(axis=0, keepdims=True)
    std = np.power((np.power(inds - com, 2) * avg_trial_mat_norm).sum(axis=0), .5)
    return std.ravel()
