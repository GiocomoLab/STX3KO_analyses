import os

import dill
import numpy as np
import warnings

import TwoPUtils as tpu
from . import session, ymaze_sess_deets

def loop_func_over_mice(func, mice):
    return {mouse: func(mouse) for mouse in mice}

def loop_func_over_days(func, days, **kwargs):
    # return lambda mouse: [func(load_single_day(mouse, day), **kwargs) for day in days] # uncomment if not downsampled
    return lambda mouse: [func(load_single_day(mouse, day), **kwargs) for day in days]

def center_of_mass(data, coord=None, axis=0):
    """
    Find the data's absolute center of mass (COM) along the given axis

    :param data: mass
    :param coord: coordinates of the data, from which we calculate center of mass
            -- If coord==None, coord are the indices of the data.
    :param axis: axis to calculate across
    :return:
    """

    valid_data = np.copy(data) #[np.where(~np.isnan(data))]

    if np.sum(~np.isnan(valid_data))>0:
        if coord is None:
            coord = np.indices((data.shape))[axis]
            # coord = np.arange(0, data.shape[axis])

        #valid_coord = coord[~np.isnan(data)]

        # make data positive, looking for center of upward-going mass
        mass = valid_data - np.nanmin(valid_data, axis=axis, keepdims=True)

        normalizer = np.nansum(np.abs(valid_data), axis=axis, keepdims=True)

        with np.errstate(invalid='ignore', divide='ignore'):
            COM = (
                    np.nansum(np.abs(valid_data) * coord, axis=axis, keepdims=True) / normalizer
            )
    else:
        COM = np.nan

    return COM

def field_from_thresh(trial_mat, coord, cells=None,
                      prctile=0.2,
                      axis=None,
                      sigma=2):
    """
    Calculates the field coordinates for a single place field,
    defined as activity >= prctile*(max-min)+min

    :param trial_mat: position-binned activity
    :param coord: the coordinates of the data (i.e. positions)
    :param prctile: percentile of activity change to use as threshold
    :param axis:
    :return: coordinates of the place field
    """
    fields_per_cell = {'included cells': {},
                       'number': {},
                       'widths': {},
                       'pos': {},
                       'COM': {}}

    trial_mat = np.copy(trial_mat)

    if len(trial_mat.shape) < 3:
        if len(trial_mat.shape) < 2:
            trial_mean = trial_mat
            trial_mean = np.expand_dims(trial_mean, axis=1)
            if cells is None:
                cells = [0]
        else:
            trial_mat = np.expand_dims(trial_mat, axis=2)
            trial_mean = np.nanmean(trial_mat, axis=0)
            if cells is None:
                cells = range(trial_mat.shape[2])
    else:
        trial_mean = np.nanmean(trial_mat, axis=0)
        if cells is None:
            cells = range(trial_mat.shape[2])

    fields_per_cell['included cells'] = cells

    for cell in cells:
        minmax = np.nanmax(trial_mean[:, cell], axis=axis) - \
            np.nanmin(trial_mean[:, cell], axis=axis)
        thresh = prctile * minmax + np.nanmin(trial_mean[:, cell], axis=axis)
        # trying the mean of the mean firing as the thresh

        # thresh = np.nanmean(trial_mean[:, cell])

        above_thresh = np.where(trial_mean[:, cell] > thresh)[0]

        fields_inds = find_contiguous(above_thresh, stepsize=1)
        # only accept fields longer than 2 bins
        fields_inds = fields_inds[[len(f) >= 2 for f in fields_inds]]
        fields_pos = []
        for f in fields_inds:
            fields_pos.append(coord[f])  # a list of arrays

        fields_per_cell['number'][cell] = len(fields_pos)
        fields_per_cell['widths'][cell] = [(f[-1] - f[0]) for f in fields_pos]
        fields_per_cell['pos'][cell] = fields_pos
        fields_per_cell['COM'][cell] = []

        # find center of mass of each field
        for field in fields_pos:
            data = trial_mean[:, cell]
            field_COM = center_of_mass(
                data[np.isin(coord,
                             field)
                     ],
                coord=field,
                axis=0)

            fields_per_cell['COM'][cell].append(field_COM)

    return fields_per_cell
    
def find_contiguous(data, stepsize=1, up_to_stepsize=None):
    """
    Find contiguous runs of elements based on a "stepsize" difference between them.
    Thanks to Stack Exchange for this inspiration

    :param data: array in which to look for contiguous runs
    :param stepsize: target difference between elements
    :return: an array of arrays
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=np.VisibleDeprecationWarning)
        if up_to_stepsize is not None:
            runs = np.asarray(np.split(data, np.where(np.diff(data) > stepsize)[0] + 1)) #, dtype=object)
        else:
            runs = np.asarray(np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)) #, dtype=object)
        return runs
        
def common_rois(roi_matches, inds):
    ref = roi_matches[inds[0]]
    ref_common_rois = []

    for i, targ_ind in enumerate(inds[1:]):

        #         targ = roi_matches[targ_ind][inds[0]]
        if i == 0:

            ref_common_rois = set(ref[targ_ind]['ref_inds'])
        else:
            ref_common_rois = ref_common_rois & set(ref[targ_ind]['ref_inds'])

        # find cells that are in reference match list each time
    ref_common_rois = list(ref_common_rois)

    # find matching indices
    common_roi_mapping = np.zeros([len(inds), len(ref_common_rois)]) * np.nan
    common_roi_mapping[0, :] = ref_common_rois
    for i, roi in enumerate(ref_common_rois):
        for j, targ_ind in enumerate(inds[1:]):
            #             print(j)
            ind = np.argwhere(ref[targ_ind]['ref_inds'] == roi)[0][0]
            #             print(j,roi,ind)
            common_roi_mapping[j + 1, i] = ref[targ_ind]['targ_inds'][ind]

    return common_roi_mapping.astype(int)



def get_ind_of_exp_day(sess_list, exp_day):

    index = [] # default if no matching exp_day is found

    for ind, day_dict in enumerate(sess_list):
        if type(day_dict) is tuple:
            for i, entry in enumerate(day_dict):
                # print(i)
                if entry["exp_day"] == exp_day:
                    index = ind
        else:
            if day_dict["exp_day"] == exp_day:
                index = ind

    return index
    
def load_vr_day(mouse,day, verbose = True, trial_mat_keys = ('licks','speed'), timeseries_keys = ('licks', 'speed')):
    # pkldir = os.path.join('Z:/giocomo/mplitt/2P_Data/STX3KO/YMaze_VR_Pkls/', mouse)
    if mouse in ["SparseKO_02","SparseKO_03", "SparseKO_05","SparseKO_06","SparseKO_08","SparseKO_09","SparseKO_10","SparseKO_11","SparseKO_13"]:
        pkldir = os.path.join('C:/Users/esay/data/Stx3/YMaze_VR_Pkls/', mouse)
    else:
        pkldir = os.path.join("C://Users/esay/data/Stx3/YMaze_VR_Pkls/", mouse)
        # pkldir = os.path.join('Z:/giocomo/mplitt/2P_Data/STX3KO/YMaze_VR_Pkls/', mouse)
    if mouse in ymaze_sess_deets.KO_behavior_sessions.keys():

        deets = ymaze_sess_deets.KO_behavior_sessions[mouse][day]
    elif mouse in ymaze_sess_deets.CTRL_behavior_sessions.keys():
        deets = ymaze_sess_deets.CTRL_behavior_sessions[mouse][day]
    elif mouse in ymaze_sess_deets.SparseKO_behavior_sessions.keys():
        deets = ymaze_sess_deets.SparseKO_behavior_sessions[mouse][day]
    else:
        raise Exception("invalid mouse name")

    if verbose:
        print(deets)
    if isinstance(deets, tuple):

        sess_list = []
        for _deets in deets:
            _sess = session.YMazeSession.from_file(
                os.path.join(pkldir, _deets['date'], "%s_%d.pkl" % (_deets['scene'], _deets['session'])),
                verbose=False, novel_arm=_deets['novel_arm'])

            # print(_deets['date'], _deets['scene'])
            sess_list.append(_sess)

        sess = session.ConcatYMazeSession(sess_list, None, day_inds=[0 for i in range(len(deets))],
                                          trial_mat_keys=trial_mat_keys,
                                          timeseries_keys=timeseries_keys,
                                          run_place_cells=False)
        if mouse in ['4467332.2'] and day == 0:
            mask = sess.trial_info['sess_num_ravel'] > 0
            sess.trial_info['block_number'][mask] -= 1
    else:
        sess = session.YMazeSession.from_file(
            os.path.join(pkldir, deets['date'], "%s_%d.pkl" % (deets['scene'], deets['session'])),
            verbose=False, novel_arm=deets['novel_arm'])
        # sess.add_timeseries(licks=sess.vr_data['lick']._values)
        # sess.add_pos_binned_trial_matrix('licks')
        # sess.novel_arm = deets['novel']
        # setattr(sess, 'novel_arm', deets['novel'])

        if mouse == '4467975.1' and day == 0:
            sess.trial_info['block_number'] += 1
        if mouse == '4467332.2' and day == 0:
            sess.trial_info['block_number'] += 2

    return sess


# def load_single_day(mouse,day,verbose=True,pkl_basedir = '/home/mplitt/YMazeSessPkls'):

#     if pkl_basedir=='/home/mplitt/YMazeSessPkls':
#         return load_single_day_orig(mouse,day,verbose=verbose, pkl_basedir=pkl_basedir)

#     mouse_dir = os.path.join(pkl_basedir, mouse)
#     if mouse in ymaze_sess_deets.KO_sessions.keys():
#         deets = ymaze_sess_deets.KO_sessions[mouse][day]
#     elif mouse in ymaze_sess_deets.CTRL_sessions.keys():
#         deets = ymaze_sess_deets.CTRL_sessions[mouse][day]
#     else:
#         raise Exception("invalid mouse name")

#     sess = session.YMazeSession.from_file(os.path.join(mouse_dir, deets['date'], "sess.pkl"), verbose=False, novel_arm=deets['novel_arm'])
#     return sess


def load_single_day(mouse, day, pkl_basedir = "C://Users/esay/data/Stx3/YMazeSessPkls/smooth_spks",verbose = True):
    #     mouse = '4467331.2'
    pkldir = os.path.join(pkl_basedir, mouse)
    if mouse in ymaze_sess_deets.KO_sessions.keys():

        deets = ymaze_sess_deets.KO_sessions[mouse][day]
        pkldir = os.path.join("Z://giocomo/mplitt/2P_Data/STX3KO/YMazeSessPkls", mouse)
        # pkldir = os.path.join("C://Users/esay/data/Stx3/YMazeSessPkls", mouse)

        # pkldir = os.path.join("C://Users/esay/data/Stx3/downsample_lickrate", mouse)
        # pkldir = os.path.join("C://Users/esay/data/Stx3/downsample_speed", mouse)
    elif mouse in ymaze_sess_deets.CTRL_sessions.keys():
        deets = ymaze_sess_deets.CTRL_sessions[mouse][day]
        # pkldir = os.path.join("C://Users/esay/data/Stx3/YMazeSessPkls", mouse)
        pkldir = os.path.join("Z://giocomo/mplitt/2P_Data/STX3KO/YMazeSessPkls", mouse)

        # pkldir = os.path.join("C://Users/esay/data/Stx3/downsample_lickrate", mouse)
        # pkldir = os.path.join("C://Users/esay/data/Stx3/downsample_speed", mouse)
    elif mouse in ymaze_sess_deets.SparseKO_sessions.keys():
        deets = ymaze_sess_deets.SparseKO_sessions[mouse][day]
    else:
        raise Exception("invalid mouse name")

    if verbose:
        print(deets)
    if isinstance(deets, tuple):
        
        roi_aligner_dir = os.path.join(pkl_basedir, mouse)
        with open(os.path.join(roi_aligner_dir, "roi_aligner_results.pkl"), 'rb') as file:
            match_inds = dill.load(file)

        common_roi_mapping = common_rois(match_inds, [d['ravel_ind'] for d in deets])
        sess_list = []
        for _deets in deets:
            _sess = session.YMazeSession.from_file(
                os.path.join(pkldir, _deets['date'], "%s_%d.pkl" % (_deets['scene'], _deets['session'])),
                verbose=False, novel_arm=_deets['novel_arm'])
            _sess.add_timeseries(licks=_sess.vr_data['lick']._values)
            _sess.add_pos_binned_trial_matrix('licks')
            
            # setattr(_sess,'novel_arm', _deets['novel'])
            # _sess.novel_arm = _deets['novel']
            #             _sess_list.append(sess)
            print(_deets['date'], _deets['scene'])
            sess_list.append(_sess)

        sess = session.ConcatYMazeSession(sess_list, common_roi_mapping, day_inds=[0 for i in range(len(deets))],
                                          trial_mat_keys=('F_dff', 'spks', 'F_dff_norm', 'spks_norm','licks', 'speed'),#, 'spks_nostop''spks_th',),
                                          timeseries_keys=('F_dff', 'spks',  'F_dff_norm', 'spks_norm','licks', 'speed', 
                                                           't', 'LR'),#, 'spks_nostop''spks_th','reward',, 'block_number' ),
                                          run_place_cells=True)
        if mouse in ['4467332.2'] and day == 0:
            mask = sess.trial_info['sess_num_ravel'] > 0
            sess.trial_info['block_number'][mask] -= 1
    else:
        sess = session.YMazeSession.from_file(
            os.path.join(pkldir, deets['date'], "%s_%d.pkl" % (deets['scene'], deets['session'])),
            verbose=False, novel_arm=deets['novel_arm'])
        sess.add_timeseries(licks=sess.vr_data['lick']._values)
        sess.add_pos_binned_trial_matrix('licks')

        # sess.novel_arm = deets['novel']
        # setattr(sess, 'novel_arm', deets['novel'])

        if mouse == '4467975.1' and day == 0:
            sess.trial_info['block_number'] += 1
        if mouse == '4467332.2' and day == 0:
            sess.trial_info['block_number'] += 2

    return sess

def load_single_day_noconcat(mouse, day, pkl_basedir = "C://Users/esay/data/Stx3/YMazeSessPkls",verbose = True):
    #     mouse = '4467331.2'
    pkldir = os.path.join(pkl_basedir, mouse)
    if mouse in ymaze_sess_deets.KO_sessions.keys():

        deets = ymaze_sess_deets.KO_sessions[mouse][day]
        # pkldir = os.path.join("Z://giocomo/mplitt/2P_Data/STX3KO/YMazeSessPkls", mouse)
        # pkldir = os.path.join("C://Users/esay/data/Stx3/YMazeSessPkls", mouse)

        pkldir = os.path.join("C://Users/esay/data/Stx3/downsample_speed", mouse)
        
    elif mouse in ymaze_sess_deets.CTRL_sessions.keys():
        deets = ymaze_sess_deets.CTRL_sessions[mouse][day]
        # pkldir = os.path.join("C://Users/esay/data/Stx3/YMazeSessPkls", mouse)
        # pkldir = os.path.join("Z://giocomo/mplitt/2P_Data/STX3KO/YMazeSessPkls", mouse)

        pkldir = os.path.join("C://Users/esay/data/Stx3/downsample_speed", mouse)
    elif mouse in ymaze_sess_deets.SparseKO_sessions.keys():
        deets = ymaze_sess_deets.SparseKO_sessions[mouse][day]
    else:
        raise Exception("invalid mouse name")

    # if verbose:
    #     print(deets)
    if isinstance(deets, tuple):
        deets = deets[0]
        print(deets)
        filename = os.path.join(pkldir, deets['date'], "%s_%d.pkl" % (deets['scene'], deets['session']))
        with open(filename, 'rb') as file:
            sess = dill.load(file)
    else:
        print(deets)
        sess = session.YMazeSession.from_file(
            os.path.join(pkldir, deets['date'], "%s_%d.pkl" % (deets['scene'], deets['session'])),
            verbose=False, novel_arm=deets['novel_arm'])
        sess.add_timeseries(licks=sess.vr_data['lick']._values)
        sess.add_pos_binned_trial_matrix('licks')

    # sess.novel_arm = deets['novel']
    # setattr(sess, 'novel_arm', deets['novel'])

    if mouse == '4467975.1' and day == 0:
        sess.trial_info['block_number'] += 1
    if mouse == '4467332.2' and day == 0:
        sess.trial_info['block_number'] += 2

    return sess

def single_mouse_concat_vr_sessions(mouse, date_inds=None):
    pkldir = os.path.join('C:/Users/esay/data/Stx3/YMaze_VR_Pkls/', mouse)

    if mouse in ymaze_sess_deets.KO_behavior_sessions.keys():
        sessions_deets = ymaze_sess_deets.KO_behavior_sessions[mouse]
    elif mouse in ymaze_sess_deets.CTRL_behavior_sessions.keys():
        sessions_deets = ymaze_sess_deets.CTRL_behavior_sessions[mouse]
    elif mouse in ymaze_sess_deets.SparseKO_behavior_sessions.keys():
        sessions_deets = ymaze_sess_deets.SparseKO_behavior_sessions[mouse]
    else:
        print("mouse ID typo")
        print("shenanigans")
    if date_inds is None:
        date_inds = np.arange(len(sessions_deets)).tolist()

    date_inds_ravel = []
    sess_list = []
    for date_ind in date_inds:
        deets = sessions_deets[date_ind]
        if isinstance(deets, tuple):
            _sess_list = []
            for _deets in deets:
                sess = session.YMazeSession.from_file(
                    os.path.join(pkldir, _deets['date'], "%s_%d.pkl" % (_deets['scene'], _deets['session'])),
                    verbose=False)

                sess_list.append(sess)
                date_inds_ravel.append(date_ind)

                if mouse in ['4467332.2'] and date_ind == 0:
                    mask = sess.trial_info['sess_num_ravel'] > 0
                    sess.trial_info['block_number'][mask] -= 1
        else:
            sess = session.YMazeSession.from_file(
                os.path.join(pkldir, deets['date'], "%s_%d.pkl" % (deets['scene'], deets['session'])),
                verbose=False)

            sess_list.append(sess)
            date_inds_ravel.append(date_ind)

            # print(deets['date'], deets['scene'])

            if mouse == '4467975.1' and date_ind == 0:
                sess.trial_info['block_number'] += 1
            if mouse == '4467332.2' and date_ind == 0:
                sess.trial_info['block_number'] += 2


    concat_sess = session.ConcatYMazeSession(sess_list, None, day_inds=date_inds_ravel,
                                             trial_mat_keys=['licks','nonconsum_licks','licks_sum','speed'],
                                             timeseries_keys=[ 'licks', 'nonconsum_licks','licks_sum','speed'],
                                             load_ops=False, run_place_cells=False)
    return concat_sess

def single_mouse_concat_sessions(mouse, chan, date_inds=None, load_ops = False, load_stats = True):
    pkldir = os.path.join('C:/Users/esay/data/Stx3/YMazeSessPkls/', mouse)

    with open(os.path.join(pkldir, "roi_aligner_results.pkl"), 'rb') as file:
        match_inds = dill.load(file)

    if mouse in ymaze_sess_deets.KO_sessions.keys():
        sessions_deets = ymaze_sess_deets.KO_sessions[mouse]
    elif mouse in ymaze_sess_deets.CTRL_sessions.keys():
        sessions_deets = ymaze_sess_deets.CTRL_sessions[mouse]
    elif mouse in ymaze_sess_deets.SparseKO_sessions.keys():
        sessions_deets = ymaze_sess_deets.SparseKO_sessions[mouse]
    else:
        print("mouse ID typo")
        print("shenanigans")
    if date_inds is None:
        date_inds = np.arange(len(sessions_deets)).tolist()

    date_inds_ravel = []
    roi_inds = []
    sess_list = []
    for date_ind in date_inds:
        deets = sessions_deets[date_ind]
        if isinstance(deets, tuple):
            _sess_list = []
            for _deets in deets:
                sess = session.YMazeSession.from_file(
                    os.path.join(pkldir, _deets['date'], "%s_%d.pkl" % (_deets['scene'], _deets['session'])),
                    verbose=False)
                sess.add_timeseries(licks=sess.vr_data['lick']._values)
                sess.add_pos_binned_trial_matrix('licks')
                sess.novel_arm = _deets['novel_arm']
                #             _sess_list.append(sess)
                print(_deets['date'], _deets['scene'])
                sess_list.append(sess)
                date_inds_ravel.append(date_ind)
                roi_inds.append(_deets['ravel_ind'])
                if mouse in ['4467332.2'] and date_ind == 0:
                    mask = sess.trial_info['sess_num_ravel'] > 0
                    sess.trial_info['block_number'][mask] -= 1
        else:
            sess = session.YMazeSession.from_file(
                os.path.join(pkldir, deets['date'], "%s_%d.pkl" % (deets['scene'], deets['session'])),
                verbose=False)
            sess.add_timeseries(licks=sess.vr_data['lick']._values)
            sess.add_pos_binned_trial_matrix('licks')
            sess.novel_arm = deets['novel_arm']
            sess_list.append(sess)
            date_inds_ravel.append(date_ind)
            roi_inds.append(deets['ravel_ind'])
            print(deets['date'], deets['scene'])

    common_roi_mapping = common_rois(match_inds, roi_inds)
    concat_sess = session.ConcatYMazeSession(sess_list, common_roi_mapping, day_inds=date_inds_ravel,
                                             trial_mat_keys=['F_dff', 'F_dff_norm', 'spks', 'spks_th', 'spks_norm', 'licks', 'speed', 'spks_nostop'],
                                             timeseries_keys=('F_dff', 'spks', 'spks_th', 'F_dff_norm', 'spks_norm','licks', 'speed', 
                                                           't', 'LR', 'reward', 'block_number', 'spks_nostop'),
                                             load_ops=load_ops, load_stats = load_stats)
    return concat_sess

def is_putative_interneuron(sess, ts_key='dff', method='speed',
                            prct=10, r_thresh=0.18, mux = False):
    """
    Find putative interneurons based on spatially-binned
    trial_mat values, ratio of 99th prctile to mean, per cell.
    Taking anything <10th prctile within animal as an "int".

    :param sess: session class
    :param ts_key: key of timeseries to use
    :param method: method of identifying ints: 'speed' or 'trial_mat_ratio'
    :param prct: (for method 'trial_mat_ratio') percentile of ROIs within animal to cut off
    :param r_thresh: (for method 'speed') threshold for speed correlation r
    :return: is_int (list) with a Boolean for each ROI, where
        putative interneuron is True. NOTE: ELLA SWITCHED >/< SO IS_INT HAS NON-INTERNEURONS AS TRUE 
    """

    use_trial_mat = np.copy(sess.trial_matrices[ts_key][0])
    if method == 'trial_mat_ratio':
        trial_mat_prop = []
        trial_mat_ratio = []
        for c in range(use_trial_mat.shape[2]):
            _trial_mat = use_trial_mat[:, :, c]
            nanmask = np.isnan(_trial_mat)
            trial_mat_prop.append(np.percentile(_trial_mat[~nanmask], 75))

            trial_mat_max = np.percentile(_trial_mat[~nanmask], 99)

            ratio = trial_mat_max / np.mean(_trial_mat[~nanmask])
            trial_mat_ratio.append(ratio)

        is_int = trial_mat_ratio < np.percentile(trial_mat_ratio, prct)

    elif method == 'speed':
        if mux: 
            if 'channel_0' in ts_key:
                speed = np.copy(sess.vr_data_chan0['speed']._values)
            elif 'channel_1' in ts_key:
                speed  = np.copy(sess.vr_data_chan1['speed']._values)
        else:     
            speed = np.copy(sess.vr_data['speed']._values)
            
        speed_corr = np.zeros((use_trial_mat.shape[1],))
        nanmask = ~np.isnan(sess.timeseries[ts_key][0, :])

        for c in range(use_trial_mat.shape[1]):
            speed_corr[c] = np.corrcoef(
                sess.timeseries[ts_key][c, nanmask], speed[nanmask])[0, 1]

        is_int = speed_corr < r_thresh

    elif method == 'z_score':
        ts = sess.timeseries[ts_key]
        n_cells = ts.shape[0]
        # is_int = np.zeros(n_cells, dtype = bool)
        
        zscored = np.empty_like(ts)
        for c in range(n_cells):
            trace = ts[c,:]
            if np.isnan(trace.all()):
                is_int[c] = False
                continue

            trace_z = (trace - np.nanmean(trace)) / np.nanstd(trace)
            zscored[c,:] = trace_z
            # z_thresh = np.nanpercentile(zscored, 90)
            # max_z = np.nanmax(trace_z)
            # print(max_z)
            # is_int[c] = max_z >= z_thresh
        z_thresh = np.nanpercentile(np.nanmax(zscored, axis = 1), 70)
        print(z_thresh)
        print(np.nanmax(zscored, axis = 1))
        is_int = np.nanmax(zscored, axis = 1) >=z_thresh
    
    return is_int
