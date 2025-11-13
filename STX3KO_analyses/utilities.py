import os

import dill
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec as gridspec

from . import session, ymaze_sess_deets

def plot_cells(trial_mat, cell_inds=None, n_cols=20):
    '''

    :param ca1:
    :param cell_inds: indices of cells to plot
    :param save_figs:
    :return:
    '''

    if cell_inds is None:
        cell_inds = np.arange(trial_mat.shape[-1])

    n_rows = int(np.ceil(cell_inds.shape[0] / n_cols))
    fig = plt.figure(figsize=[30, 3 * n_rows])
    gs = gridspec(n_rows, n_cols)
    for i, cell in enumerate(cell_inds):
        col = i % n_cols
        row = int(i / n_cols)
        ax = fig.add_subplot(gs[row, col])
        h = ax.imshow(trial_mat[:, :, cell], cmap="magma",aspect='auto')
        ax.set_title(f'cell {cell}')

        if col == 0:
            ax.set_xlabel('pos')
            ax.set_ylabel('trial #')
            if row==0:
                plt.colorbar(h,ax=ax)
        else:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.subplots_adjust(hspace=.3)
    return fig


def loop_func_over_mice(func, mice):
    return {mouse: func(mouse) for mouse in mice}

def loop_func_over_days(func, days, **kwargs):
    return lambda mouse: [func(load_single_day(mouse, day), **kwargs) for day in days]

def add_sig_field_thresh(sess, shuff_results, key='F_dff_th', min_bins=1, max_bins=25):
    '''
    add info about single field permutations to sess file
    '''
    
    field_perm_masks = {'sig_bins':{}, 'cell_masks':{}}
    for ttype in ('fam', 'nov'):
        if ttype == 'nov':
            trial_mask = sess.trial_info['LR']==sess.novel_arm
        else:
            trial_mask = sess.trial_info['LR']!=sess.novel_arm 

        sig_bins = np.nanmean(sess.trial_matrices[key][trial_mask,:,:],axis=0)>shuff_results[ttype]
        field_perm_masks['sig_bins'][ttype] = sig_bins
        sig_bin_count = (1.*sig_bins).sum(axis=0)

        field_perm_masks['cell_masks'][ttype] = (sig_bin_count>min_bins) * (sig_bin_count<max_bins)
    sess.field_perm_masks = field_perm_masks
    return sess

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


def get_cell_orders(mouse, session_deets, days, match_inds):
    if mouse in ymaze_sess_deets.ctrl_mice:
        cond_key='ctrl'
    else:
        cond_key='cre'
            
    
    cell_order = []
    for day in days:
        deets = session_deets[day]
        if isinstance(deets,tuple):
            roi_inds = []
            for _deets in deets:
                roi_inds.append(_deets['ravel_ind'])
            print(roi_inds)
            mapping = common_rois(match_inds, roi_inds)
            
            cell_order.append(mapping[0,:].tolist())
        else:
            cell_order.append([])
    return cell_order

def common_rois_adjust(mouse, days):
    if mouse in ymaze_sess_deets.ctrl_mice:
        session_deets = ymaze_sess_deets.CTRL_sessions[mouse]
    else:
        session_deets = ymaze_sess_deets.KO_sessions[mouse]
    
    
    ravel_inds = []
    day_inds = []
    for day in days:
        deets = session_deets[day]
        if isinstance(deets, tuple):
            for _deets in deets:
                ravel_inds.append(_deets['ravel_ind'])
                day_inds.append(day)
        else:
            ravel_inds.append(deets['ravel_ind'])
            day_inds.append(day)

    pkldir = os.path.join('/home/mplitt/YMazeSessPkls/', mouse)
    with open(os.path.join(pkldir, "roi_aligner_results.pkl"), 'rb') as file:
        match_inds = dill.load(file)
    

    common_roi_mapping = common_rois(match_inds, ravel_inds)
    c_rows = []
    for i, d in enumerate(day_inds):
        if d != day_inds[i-1]:
            c_rows.append(i)
    
    common_roi_mapping = common_roi_mapping[c_rows,:]
    cell_order = get_cell_orders(mouse,session_deets, days, match_inds)

    roi_mapping_adj = np.zeros(common_roi_mapping.shape)
    c_cols = []
    for row, order in enumerate(cell_order):
        croi = common_roi_mapping[row,:]

        if len(order)==0:
            roi_mapping_adj[row,:] = croi
        else:
            for col, _c in enumerate(croi):
                match = np.argwhere(order==_c)
                if len(match)>0:
                    c_cols.append(col)
                    roi_mapping_adj[row,col]=match[0][0]
                    
                    
    if len(c_cols) !=0:
        roi_mapping_adj = roi_mapping_adj[:,c_cols]
    return roi_mapping_adj.astype(int)


def load_vr_day(mouse,day, verbose = True, trial_mat_keys = ('licks','speed'), timeseries_keys = ('licks', 'speed'), pkldir = '/home/mplitt/YMaze_VR_Pkls/'):
    pkldir = os.path.join(pkldir, mouse)
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
                                          run_place_cells=False, run_field_perm_masks=False)
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
    

def load_single_day(mouse, day, pkl_basedir='/home/mplitt/YMazeSessPkls/',verbose = True,
                    trial_mat_keys=('F_dff', 'F_dff_th', 'spks', 'spks_th', 'F_dff_norm', 'F_dff_bin',
                            'spks_norm','licks', 'speed', 'spks_nostop'),
                    timeseries_keys=('F_dff', 'F_dff_th', 'spks', 'spks_th', 'F_dff_norm', 'F_dff_bin',
                            'spks_norm','licks', 'speed', 
                            't', 'LR', 'reward', 'block_number', 'spks_nostop'),):
    #     mouse = '4467331.2'
    pkldir = os.path.join(pkl_basedir, mouse)
    if mouse in ymaze_sess_deets.KO_sessions.keys():

        deets = ymaze_sess_deets.KO_sessions[mouse][day]
    elif mouse in ymaze_sess_deets.CTRL_sessions.keys():
        deets = ymaze_sess_deets.CTRL_sessions[mouse][day]
    elif mouse in ymaze_sess_deets.sparse_mice:
        deets = ymaze_sess_deets.SparseKO_sessions[mouse][day]
    else:
        raise Exception("invalid mouse name")

    if verbose:
        print(deets)
    if isinstance(deets, tuple):
        
        roi_aligner_dir = os.path.join('/home/mplitt/YMazeSessPkls/', mouse)
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
            # _sess.novel_arm = _deets['novel']ies(licks=_sess.vr_data['lick']._values)
            # _sess.a
            #             _sess_list.append(sess)
            print(_deets['date'], _deets['scene'])
            sess_list.append(_sess)

        sess = session.ConcatYMazeSession(sess_list, common_roi_mapping, day_inds=[0 for i in range(len(deets))],
                                          trial_mat_keys=trial_mat_keys,
                                          timeseries_keys=timeseries_keys,
                                          run_place_cells=True)
        for ttype in ('fam', 'nov'):
            sess.field_perm_masks['sig_bins'][ttype] = np.array(sess.field_perm_masks['sig_bins'][ttype]).sum(axis=0)==len(deets)
            sess.field_perm_masks['cell_masks'][ttype] = np.array(sess.field_perm_masks['cell_masks'][ttype]).sum(axis=0)==len(deets)
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

def single_mouse_concat_sessions(mouse, date_inds=None, load_ops = False, load_stats = True,
                                trial_mat_keys=['F_dff', 'F_dff_norm', 'F_dff_th', 'F_dff_bin', 'spks', 'spks_th', 
                                                'spks_norm', 'licks', 'speed', 'spks_nostop'],
                                timeseries_keys=('F_dff', 'spks', 'spks_th', 'F_dff_norm', 'F_dff_th', 'F_dff_bin',
                                                'spks_norm','licks', 'speed', 
                                                't', 'LR', 'reward', 'block_number', 'spks_nostop'),):
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
                                             trial_mat_keys=trial_mat_keys,
                                             timeseries_keys=timeseries_keys,
                                             load_ops=load_ops, load_stats = load_stats)
    return concat_sess


 