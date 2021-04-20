import numpy as np


def antic_consum_licks(sess):
    '''

    :param sess:
    :return:
    '''
    reward_mask = sess.vr_data['reward']._values > 0
    reward_start = np.argwhere(reward_mask).ravel()
    reward_end = (reward_start + int(2 * sess.scan_info['frame_rate'])).astype(np.int)

    consum_mask = np.zeros(reward_mask.shape) > 0
    for (start, end) in zip(reward_start, reward_end):
        consum_mask[start:end] = True

    antic_licks = np.copy(sess.vr_data['lick']._values)
    antic_licks[consum_mask] = 0

    nonconsum_speed = np.copy(sess.vr_data['dz']._values)
    nonconsum_speed[consum_mask] = np.nan

    sess.add_timeseries(antic_licks=antic_licks,
                        licks=sess.vr_data['lick']._values,
                        speed=sess.vr_data['dz']._values,
                        antic_speed=nonconsum_speed)
    sess.add_pos_binned_trial_matrix(('antic_licks', 'speed', 'antic_speed'), 't', mat_only=True)

    antic_lick_positions = np.zeros(sess.timeseries['licks'].shape) * np.nan
    antic_lick_mask = sess.timeseries['antic_licks'] > 0
    antic_lick_positions[antic_lick_mask] = sess.vr_data['t']._values[antic_lick_mask.ravel()]
    sess.add_timeseries(antic_lick_positions=antic_lick_positions)


def get_probes_and_omissions(sess):
    '''

    :param sess:
    :return:
    '''
    probes = np.zeros([sess.trial_start_inds.shape[0], ])
    omissions = np.zeros([sess.trial_start_inds.shape[0], ])
    for trial, (start, stop, lr) in enumerate(zip(sess.trial_start_inds, sess.teleport_inds, sess.trial_info['LR'])):
        if sess.scene in ("YMaze_RewardReversal"):
            lr = np.copy(lr) * -1

        pos = sess.vr_data['t'].iloc[start:stop]
        licks = sess.vr_data['lick'].iloc[start:stop]
        reward = sess.vr_data['reward'].iloc[start:stop]
        if lr == 1:
            rzone = (sess.rzone_late['tfront'], sess.rzone_late['tback'])
        else:
            rzone = (sess.rzone_early['tfront'], sess.rzone_early['tback'])
        rzone_mask = (pos >= rzone[0]) & (pos <= rzone[1])

        r = reward.sum()
        rzone_licks = licks.loc[rzone_mask].sum()
        if r == 0 and rzone_licks > 0:
            probes[trial] = 1
        elif r == 0 and rzone_licks == 0:
            omissions[trial] = 1
        else:
            pass

        sess.trial_info.update({'probes': probes, 'omissions': omissions})


def single_trial_lick_metrics(sess):
    '''

    :param sess:
    :return:
    '''
    bin_lower_edges = sess.trial_matrices['bin_edges'][:-1]

    lr_early = np.nanmean(sess.trial_matrices['antic_licks'][:, (bin_lower_edges >= sess.rzone_early['t_antic']) & (
                bin_lower_edges < sess.rzone_early['tfront'] + 2)], axis=-1)
    lr_early /= np.nanmean(sess.trial_matrices['antic_licks'].ravel())
    lr_late = np.nanmean(sess.trial_matrices['antic_licks'][:, (bin_lower_edges >= sess.rzone_late['t_antic']) & (
                bin_lower_edges < sess.rzone_late['tfront'] + 2)], axis=-1)
    lr_late /= np.nanmean(sess.trial_matrices['antic_licks'].ravel())
    #     print(lr_early,lr_late)
    lr_d = (lr_early - lr_late) / (lr_early + lr_late + 1E-3)

    stem_speed = np.nanmean(sess.trial_matrices['antic_speed'][:,
                            sess.trial_matrices['bin_edges'][:-1] < sess.rzone_early['t_antic']].ravel())
    arm_speed = np.nanmean(
        sess.trial_matrices['antic_speed'][:, sess.trial_matrices['bin_edges'][:-1] > sess.rzone_early['t_antic']],
        axis=-1)
    arm_speed_norm = arm_speed / stem_speed

    accuracy = np.zeros([sess.trial_start_inds.shape[0], ])
    err = np.zeros([sess.trial_start_inds.shape[0], ])
    mean = np.zeros([sess.trial_start_inds.shape[0], ]) * np.nan
    var = np.zeros([sess.trial_start_inds.shape[0], ])

    for trial, (start, stop, lr, omission) in enumerate(
            zip(sess.trial_start_inds, sess.teleport_inds, sess.trial_info['LR'], sess.trial_info['omissions'])):
        if omission < 1:

            if sess.scene in ("YMaze_RewardReversal"):
                lr = np.copy(lr) * -1

            pos = sess.vr_data['t'].iloc[start:stop]
            licks = sess.timeseries['antic_licks'][0, start:stop]
            lick_pos = sess.timeseries['antic_lick_positions'][0, start:stop]
            if lr == 1:
                rzone = (sess.rzone_late['t_antic'], sess.rzone_late['tfront'] + 2)
                tfront = sess.rzone_late['tfront']
                otfront = sess.rzone_early['tfront']
            else:
                rzone = (sess.rzone_early['t_antic'], sess.rzone_early['tfront'] + 2)
                tfront = sess.rzone_late['tfront']
                otfront = sess.rzone_early['tfront']
            rzone_mask = (pos >= rzone[0]) & (pos <= rzone[1])

            # lick accuracy - fraction of licks in correct reward zone plust 50 cm prior
            accuracy[trial] = licks[rzone_mask].sum() / (licks.sum() + 1E-3)

            # lick position variance
            if licks.sum() > 1:
                mean[trial] = np.nanmedian(lick_pos) / tfront
                var[trial] = np.nanstd(lick_pos)
                err[trial] = np.nanmean(np.abs((lick_pos - tfront) / tfront))
            else:
                mean[trial] = np.nansum(lick_pos) / tfront
                var[trial] = 0.
                err[trial] = np.abs(13 - tfront) / tfront
    sess.trial_info.update({'lickrate_rz_early': lr_early,
                            'lickrate_rz_late': lr_late,
                            'lickrate_dprime': lr_d,
                            'lick_acc': accuracy,
                            'lick_meanpos': mean,
                            'lick_varpos': var,
                            'lick_err': err,
                            'arm_speed': arm_speed,
                            'arm_speed_norm': arm_speed_norm

                            })







