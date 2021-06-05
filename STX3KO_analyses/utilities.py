import os

import dill
import numpy as np


from . import session, ymaze_sess_deets, behavior


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

    return common_roi_mapping.astype(np.int)


def load_single_day(mouse, day=0):
    #     mouse = '4467331.2'
    pkldir = os.path.join('/home/mplitt/YMazeSessPkls/', mouse)
    if mouse in ymaze_sess_deets.KO_sessions.keys():

        deets = ymaze_sess_deets.KO_sessions[mouse][day]
    elif mouse in ymaze_sess_deets.CTRL_sessions.keys():
        deets = ymaze_sess_deets.CTRL_sessions[mouse][day]
    else:
        raise Exception("invalid mouse name")

    print(deets)
    if isinstance(deets, tuple):
        with open(os.path.join(pkldir, "roi_aligner_results.pkl"), 'rb') as file:
            match_inds = dill.load(file)

        common_roi_mapping = common_rois(match_inds, [d['ravel_ind'] for d in deets])
        sess_list = []
        for _deets in deets:
            _sess = session.YMazeSession.from_file(
                os.path.join(pkldir, _deets['date'], "%s_%d.pkl" % (_deets['scene'], _deets['session'])),
                verbose=False)
            _sess.add_timeseries(licks=_sess.vr_data['lick']._values)
            _sess.add_pos_binned_trial_matrix('licks')
            _sess.novel_arm = _deets['novel']
            #             _sess_list.append(sess)
            print(_deets['date'], _deets['scene'])
            sess_list.append(_sess)

        sess = Concat_Session(sess_list, common_roi_mapping, day_inds=[0 for i in range(len(deets))],
                              trial_mat_keys=('F_dff', 'spks', 'F_dff_norm', 'spks_norm'),
                              timeseries_keys=('F_dff', 'spks', 'F_dff_norm', 'spks_norm'),run_place_cells=True)
        if mouse in ['4467332.2'] and day == 0:
            mask = sess.trial_info['sess_num_ravel'] > 0
            sess.trial_info['block_number'][mask] -= 1
    else:
        sess = session.YMazeSession.from_file(
            os.path.join(pkldir, deets['date'], "%s_%d.pkl" % (deets['scene'], deets['session'])),
            verbose=False)
        sess.add_timeseries(licks=sess.vr_data['lick']._values)
        sess.add_pos_binned_trial_matrix('licks')
        sess.novel_arm = deets['novel']

        if mouse == '4467975.1' and day == 0:
            sess.trial_info['block_number'] += 1
        if mouse == '4467332.2' and day == 0:
            sess.trial_info['block_number'] += 2

    return sess


class Concat_Session():

    def __init__(self, sess_list, common_roi_mapping, trial_info_keys=['LR', 'block_number'], trial_mat_keys=['F_dff',],
                 timeseries_keys=(), run_place_cells=True, day_inds=None):
        attrs = self.concat(sess_list, common_roi_mapping, trial_info_keys, trial_mat_keys,
                            timeseries_keys, run_place_cells, day_inds)

        self.__dict__.update(attrs)
        trial_info_keys = []

    @staticmethod
    def concat(_sess_list, common_roi_mapping, t_info_keys, t_mat_keys,
               timeseries_keys, run_place_cells, day_inds):
        attrs = {}
        attrs['day_inds']=day_inds
        # same info
        #         same_attrs = ['mouse', 'novel_arm','rzone_early', 'rzone_late']
        attrs.update({'mouse': _sess_list[0].mouse,
                      'novel_arm': _sess_list[0].novel_arm,
                      'rzone_early': _sess_list[0].rzone_early,
                      'rzone_late': _sess_list[0].rzone_late
                      })
        print(t_info_keys)

        # concat basic info
        basic_info_attrs = ['date', 'scan', 'scan_info', 'scene', 'session', 'teleport_inds', 'trial_start_inds']
        attrs.update({k: [] for k in basic_info_attrs})

        if 'sess_num_ravel' not in t_info_keys:
            t_info_keys.append('sess_num_ravel')
        if 'sess_num' not in t_info_keys and day_inds is not None:
            t_info_keys.append('sess_num')

        trial_info = {k: [] for k in t_info_keys}

        trial_mat = {k: [] for k in t_mat_keys}
        trial_mat['bin_edges'] = _sess_list[0].trial_matrices['bin_edges']
        trial_mat['bin_centers'] = _sess_list[0].trial_matrices['bin_centers']

        timeseries = {k: [] for k in timeseries_keys}

        if run_place_cells:
            place_cells = {-1: {'masks': [], 'SI': [], 'p': []}, 1: {'masks': [], 'SI': [], 'p': []}}

        last_block = 0
        cum_frames = 0
        for ind, _sess in enumerate(_sess_list):

            for k in basic_info_attrs:
                if k in ('teleport_inds', 'trial_start_inds'):
                    attrs[k].append(getattr(_sess, k)+cum_frames)
                else:
                    attrs[k].append(getattr(_sess, k))

            for k in t_info_keys:

                if k == 'sess_num_ravel':
                    trial_info[k].append(np.zeros([_sess.trial_info['LR'].shape[0], ]) + ind)
                elif k == 'sess_num' and day_inds is not None:
                    trial_info[k].append(np.zeros([_sess.trial_info['LR'].shape[0], ]) + day_inds[ind])

                elif k == 'block_number' and day_inds is not None and ind > 0:
                    if _sess.trial_info[k][0] == 0 and day_inds[ind - 1] == day_inds[ind]:
                        trial_info[k].append(_sess.trial_info[k] + _sess_list[ind - 1].trial_info[k][-1] + 1)
                    else:
                        trial_info[k].append(_sess.trial_info[k])
                else:
                    trial_info[k].append(_sess.trial_info[k])

            for k in t_mat_keys:
                if len(_sess.trial_matrices[k].shape) == 3:
                    trial_mat[k].append(_sess.trial_matrices[k][:, :, common_roi_mapping[ind, :]])
                else:
                    trial_mat[k].append(_sess.trial_matrices[k])

            for k in timeseries_keys:
                if len(_sess.timeseries[k].shape) == 2:
                    timeseries[k].append(_sess.timeseries[k][common_roi_mapping[ind, :], :])
                else:
                    timeseries[k].append(_sess.timeseries[k])

            if run_place_cells:
                for lr, _lr in [[-1, 'left'], [1, 'right']]:
                    for k in ['masks', 'SI', 'p']:
                        place_cells[lr][k].append(_sess.place_cell_info[_lr][k][common_roi_mapping[ind, :]])

            cum_frames+= _sess.timeseries['spks'].shape[1]
        print(t_info_keys)
        for k in ['trial_start_inds','teleport_inds']:
            attrs[k] = np.concatenate(attrs[k])

        for k in t_info_keys:
            # print(k)
            trial_info[k] = np.concatenate(trial_info[k])
        attrs['trial_info'] = trial_info

        for k in t_mat_keys:
            trial_mat[k] = np.concatenate(trial_mat[k], axis=0)
        attrs['trial_matrices'] = trial_mat

        for k in timeseries_keys:
            timeseries[k] = np.concatenate(timeseries[k],axis=-1)
        attrs['timeseries'] = timeseries

        if run_place_cells:
            for lr in [-1, 1]:
                for k in ['masks', 'SI', 'p']:
                    place_cells[lr][k] = np.array(place_cells[lr][k])
        attrs['place_cell_info'] = place_cells

        return attrs


def single_mouse_concat_sessions(mouse, date_inds=None):
    pkldir = os.path.join('/home/mplitt/YMazeSessPkls/', mouse)

    with open(os.path.join(pkldir, "roi_aligner_results.pkl"), 'rb') as file:
        match_inds = dill.load(file)

    if mouse in ymaze_sess_deets.KO_sessions.keys():
        sessions_deets = ymaze_sess_deets.KO_sessions[mouse]
    elif mouse in ymaze_sess_deets.CTRL_sessions.keys():
        sessions_deets = ymaze_sess_deets.CTRL_sessions[mouse]
    else:
        print("mouse ID typo")

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
                sess.novel_arm = _deets['novel']
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
            sess.novel_arm = deets['novel']
            sess_list.append(sess)
            date_inds_ravel.append(date_ind)
            roi_inds.append(deets['ravel_ind'])
            print(deets['date'], deets['scene'])

            if mouse == '4467975.1' and date_ind == 0:
                sess.trial_info['block_number'] += 1
            if mouse == '4467332.2' and date_ind == 0:
                sess.trial_info['block_number'] += 2

    common_roi_mapping = common_rois(match_inds, roi_inds)
    concat_sess = Concat_Session(sess_list, common_roi_mapping, day_inds=date_inds_ravel,
                                 trial_mat_keys=['F_dff', 'F_dff_norm', 'spks', 'spks_norm', 'licks'],
                                 timeseries_keys=['F_dff', 'F_dff_norm', 'spks', 'spks_norm'])
    return concat_sess