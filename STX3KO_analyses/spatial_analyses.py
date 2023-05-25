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

    # normalize rate map to be a distribution
    avg_trial_mat_norm = avg_trial_mat / (np.nansum(avg_trial_mat, axis=0, keepdims=True) + 1E-5)
    inds = np.arange(0, avg_trial_mat.shape[0])[:, np.newaxis]

    # center of mass / expected value
    com = (avg_trial_mat_norm * inds).sum(axis=0, keepdims=True)
    # spatial standard deviation
    std = np.power((np.power(inds - com, 2) * avg_trial_mat_norm).sum(axis=0), .5)
    return std.ravel()

def spatial_com(avg_trial_mat):
    '''

    :param avg_trial_mat:
    :return:
    '''

    # normalize rate map to be a distribution
    avg_trial_mat_norm = avg_trial_mat / (np.nansum(avg_trial_mat, axis=0, keepdims=True) + 1E-5)
    inds = np.arange(0, avg_trial_mat.shape[0])[:, np.newaxis]

    # center of mass / expected value
    return (avg_trial_mat_norm * inds).sum(axis=0)



def trial_matrix(arr_in, pos_in, tstart_inds, tstop_inds, bin_size=10, min_pos = 0,
                 max_pos=450, speed=None, speed_thr=2, perm=False,
                 mat_only=False, impute_nans = False, sum=False):
    """

    :param arr: timepoints x anything array to be put into trials x positions format
    :param pos: position at each timepoint
    :param tstart_inds: indices of trial starts
    :param tstop_inds: indices of trial stops
    :param bin_size: spatial bin size in cm
    :param max_pos: maximum position on track
    :param speed: vector of speeds at each timepoint. If None, then no speed filtering is done
    :param speed_thr: speed threshold in cm/s. Timepoints of low speed are dropped
    :param perm: bool. whether to circularly permute timeseries before binning. used for permutation testing
    :param mat_only: bool. return just spatial binned data or also occupancy, bin edges, and bin bin_centers
    :return: if mat_only
                    trial_mat - position binned data
             else
                    trial_mat
                    occ_mat - trials x positions matrix of bin occupancy
                    bin_edges - position bin edges
                    bin_centers - bin centers
    """

    arr = np.copy(arr_in)
    pos = np.copy(pos_in)

    ntrials = tstart_inds.shape[0]
    if speed is not None:  # mask out speeds below speed threshold
        pos[speed < speed_thr] = -1000
        arr[speed < speed_thr, :] = np.nan

    # make position bins
    bin_edges = np.arange(min_pos, max_pos + bin_size, bin_size)
    bin_centers = bin_edges[:-1] + bin_size / 2
    bin_edges = bin_edges.tolist()

    # if arr is a vector, expand dimension
    if len(arr.shape) < 2:
        arr = arr[:, np.newaxis]
        # arr = np.expand_dims(arr, axis=1)

    trial_mat = np.zeros([int(ntrials), len(bin_edges) - 1, arr.shape[1]])
    trial_mat[:] = np.nan
    occ_mat = np.zeros([int(ntrials), len(bin_edges) - 1])
    for trial in range(int(ntrials)):  # for each trial
        # get trial indices
        firstI, lastI = tstart_inds[trial], tstop_inds[trial]

        arr_t, pos_t = arr[firstI:lastI, :], pos[firstI:lastI]
        if perm:  # circularly permute if desired
            pos_t = np.roll(pos_t, np.random.randint(pos_t.shape[0]))

        # average within spatial bins
        for b, (edge1, edge2) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
            if np.where((pos_t > edge1) & (pos_t <= edge2))[0].shape[0] > 0:
                if sum:
                    trial_mat[trial, b] = np.nansum(arr_t[(pos_t > edge1) & (pos_t <= edge2), :], axis=0)
                else:
                    trial_mat[trial, b] = np.nanmean(arr_t[(pos_t > edge1) & (pos_t <= edge2), :], axis=0)
                # occ_mat[trial, b] = np.where((pos_t > edge1) & (pos_t <= edge2))[0].shape[0]
                occ_mat[trial, b] = (1-np.isnan(arr_t[(pos_t > edge1) & (pos_t <= edge2),0])).sum()
            else:
                pass

    if impute_nans:
        for trial in range(trial_mat.shape[0]):
            nan_inds = np.isnan(trial_mat[trial,:,0])
            _c = bin_centers[~nan_inds]
            for cell in range(trial_mat.shape[2]):
                _m = trial_mat[trial, ~nan_inds, cell]
                trial_mat[trial,:,cell] = np.interp(bin_centers, _c, _m)


    if mat_only:
        return np.squeeze(trial_mat)
    else:
        return np.squeeze(trial_mat), np.squeeze(occ_mat / (occ_mat.sum(axis=1)[:, np.newaxis] + 1E-3)), bin_edges, bin_centers

