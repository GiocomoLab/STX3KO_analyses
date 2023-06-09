import os

import dill
import numpy as np

from . import session, ymaze_sess_deets

def loop_func_over_mice(func, mice):
    return {mouse: func(mouse) for mouse in mice}

def loop_func_over_days(func, days, **kwargs):
    return lambda mouse: [func(load_single_day(mouse, day), **kwargs) for day in days]


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

def load_vr_day(mouse,day, verbose = True, trial_mat_keys = ('licks','speed'), timeseries_keys = ('licks', 'speed')):
    pkldir = os.path.join('/home/mplitt/YMaze_VR_Pkls/', mouse)
    if mouse in ymaze_sess_deets.KO_behavior_sessions.keys():

        deets = ymaze_sess_deets.KO_behavior_sessions[mouse][day]
    elif mouse in ymaze_sess_deets.CTRL_behavior_sessions.keys():
        deets = ymaze_sess_deets.CTRL_behavior_sessions[mouse][day]
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
    

def load_single_day(mouse, day, verbose = True, pkl_basedir = '/home/mplitt/YMazeSessPkls'):
    #     mouse = '4467331.2'
    pkldir = os.path.join(pkl_basedir, mouse)
    if mouse in ymaze_sess_deets.KO_sessions.keys():

        deets = ymaze_sess_deets.KO_sessions[mouse][day]
    elif mouse in ymaze_sess_deets.CTRL_sessions.keys():
        deets = ymaze_sess_deets.CTRL_sessions[mouse][day]
    else:
        raise Exception("invalid mouse name")

    if verbose:
        print(deets)
    if isinstance(deets, tuple):
        
        roi_aligner_dir = os.path.join('/home/mplitt/YMazeSessPkls', mouse)
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
                                          trial_mat_keys=('F_dff', 'spks', 'spks_th', 'F_dff_norm', 'spks_norm','licks', 'speed', 'spks_nostop'),
                                          timeseries_keys=('F_dff', 'spks', 'spks_th', 'F_dff_norm', 'spks_norm','licks', 'speed', 
                                                           't', 'LR', 'reward', 'block_number', 'spks_nostop'),
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



def single_mouse_concat_vr_sessions(mouse, date_inds=None):
    pkldir = os.path.join('/home/mplitt/YMaze_VR_Pkls/', mouse)

    if mouse in ymaze_sess_deets.KO_behavior_sessions.keys():
        sessions_deets = ymaze_sess_deets.KO_behavior_sessions[mouse]
    elif mouse in ymaze_sess_deets.CTRL_behavior_sessions.keys():
        sessions_deets = ymaze_sess_deets.CTRL_behavior_sessions[mouse]
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

def single_mouse_concat_sessions(mouse, date_inds=None, load_ops = False, load_stats = True):
    pkldir = os.path.join('/home/mplitt/YMazeSessPkls/', mouse)

    with open(os.path.join(pkldir, "roi_aligner_results.pkl"), 'rb') as file:
        match_inds = dill.load(file)

    if mouse in ymaze_sess_deets.KO_sessions.keys():
        sessions_deets = ymaze_sess_deets.KO_sessions[mouse]
    elif mouse in ymaze_sess_deets.CTRL_sessions.keys():
        sessions_deets = ymaze_sess_deets.CTRL_sessions[mouse]
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

            if mouse == '4467975.1' and date_ind == 0:
                sess.trial_info['block_number'] += 1
            if mouse == '4467332.2' and date_ind == 0:
                sess.trial_info['block_number'] += 2

    common_roi_mapping = common_rois(match_inds, roi_inds)
    concat_sess = session.ConcatYMazeSession(sess_list, common_roi_mapping, day_inds=date_inds_ravel,
                                             trial_mat_keys=['F_dff', 'F_dff_norm', 'spks', 'spks_th', 'spks_norm', 'licks', 'speed', 'spks_nostop'],
                                             timeseries_keys=('F_dff', 'spks', 'spks_th', 'F_dff_norm', 'spks_norm','licks', 'speed', 
                                                           't', 'LR', 'reward', 'block_number', 'spks_nostop'),
                                             load_ops=load_ops, load_stats = load_stats)
    return concat_sess
