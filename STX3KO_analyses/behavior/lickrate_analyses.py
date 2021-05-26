import os

import numpy as np
import scipy as sp
from scipy import stats
from matplotlib import pyplot as plt

from sklearn import linear_model
from sklearn.linear_model import LinearRegression as linreg
from sklearn.linear_model import HuberRegressor as hubreg


import TwoPUtils


def novelarm_runningspeed(SessDict):
    arminds = slice(23 - 6, 43 - 6)
    arm_speed_dict = {}
    for m, (mouse, days) in enumerate(SessDict.items()):
        print(mouse)
        #         LR, LICKS, SPEED = [],[],[]
        transition_trials = []
        famarm_speed, novelarm_speed = [], []

        for i, day in enumerate(days):

            for sess_ind, session in enumerate(day):
                sess = TwoPUtils.sess.Session(basedir_VR=basedir_VR, mouse=mouse, date=session['date'],
                                              scene=session['scene'],
                                              session=session['session'], VR_only=True, prompt_for_keys=False)
                sess.align_VR_to_2P()

                # get LR value for each trial
                lr_trial = get_LR_trial(sess)

                # make position binned lick rates and speed
                sess.add_timeseries(licks=sess.vr_data['lick']._values, speed=sess.vr_data['dz']._values)
                sess.add_pos_binned_trial_matrix(('licks', 'speed'), 't', min_pos=6, max_pos=43, bin_size=1,
                                                 mat_only=True)

                armspeed = sess.trial_matrices['speed'][:, arminds].mean(axis=-1)

                if lr_trial[0] == 1:
                    famarm_speed.append(armspeed[lr_trial == 1])
                    novelarm_speed.append(armspeed[lr_trial == -1])
                else:
                    famarm_speed.append(armspeed[lr_trial == -1])
                    novelarm_speed.append(armspeed[lr_trial == 1])

        arm_speed_dict[mouse] = {'fam': famarm_speed, 'novel': novelarm_speed}
    return arm_speed_dict


def get_rzone_licking(SessDict):
    rzone_early = slice(25 - 6, 32 - 6)
    rzone_late = slice(35 - 6, 42 - 6)
    for mouse, days in SessDict.items():

        #         day = days[-1]
        print(mouse)

        LR, LICKS, SPEED = [], [], []
        transition_trials = []
        for day_ind, day in enumerate(days):
            #             print(day_ind)
            for sess_ind, session in enumerate(day):
                sess = TwoPUtils.sess.Session(basedir_VR=basedir_VR, mouse=mouse, date=session['date'],
                                              scene=session['scene'],
                                              session=session['session'], VR_only=True, prompt_for_keys=False)
                sess.align_VR_to_2P()

                #         print(np.amax(sess.vr_data['t']),np.amin(sess.vr_data['t']))

                # get block number for each trial
                block_number_trial, block_number_time = get_block_number(sess)

                # get LR value for each trial
                lr_trial = get_LR_trial(sess)

                # make position binned lick rates and speed
                sess.add_timeseries(licks=sess.vr_data['lick']._values, speed=sess.vr_data['dz']._values)
                sess.add_pos_binned_trial_matrix(('licks', 'speed'), 't', min_pos=6, max_pos=43, bin_size=1,
                                                 mat_only=True)

                LR.append(lr_trial)
                LICKS.append(sess.trial_matrices['licks'])
                SPEED.append(sess.trial_matrices['speed'])

                transition_trials.append(lr_trial.shape[0])
        #                 print(lr_trial.shape)
        LR, LICKS, SPEED = np.concatenate(LR, axis=0), np.concatenate(LICKS, axis=0), np.concatenate(SPEED, axis=0)
        transition_trials = np.cumsum(np.array(transition_trials)).tolist()
        print(transition_trials)

        print(rzone_early)

        licks_rz_early = LICKS[:, rzone_early].mean(axis=-1)
        licks_rz_late = LICKS[:, rzone_late].mean(axis=-1)
        f, ax = plt.subplots(1, 2, figsize=[20, 10])
        ax[0].scatter(np.arange(LR.shape[0])[LR == 1], licks_rz_late[LR == 1], color=plt.cm.winter(1.))
        ax[0].vlines(transition_trials, 0, .3, color='red')
        ax[0].set_xlabel('trials')
        ax[0].set_ylabel('Lick rate')
        ax[0].set_title(mouse)

        ax[1].scatter(np.arange(LR.shape[0])[LR == -1], licks_rz_early[LR == -1], color=plt.cm.winter(0.))
        ax[1].vlines(transition_trials, 0, .3, color='red')
        ax[1].set_xlabel('trials')
        ax[1].set_ylabel('Lick rate')
        ax[1].set_title(mouse)
        f.savefig(os.path.join(figdir_local, "RZoneEarlyLicks_%s.png" % mouse))


def first_reversal(SessDict):
    rzone_early = slice(25 - 6, 32 - 6)
    rzone_late = slice(35 - 6, 42 - 6)
    RZONE_LICKS = {}
    slopes = np.zeros([len(SessDict.keys()), ])
    for m, (mouse, days) in enumerate(SessDict.items()):
        print(mouse)
        LR, LICKS, SPEED = [], [], []
        transition_trials = []
        early_rzone_licks = []
        for i, day in enumerate(days[:2]):

            for sess_ind, session in enumerate(day):
                sess = TwoPUtils.sess.Session(basedir_VR=basedir_VR, mouse=mouse, date=session['date'],
                                              scene=session['scene'],
                                              session=session['session'], VR_only=True, prompt_for_keys=False)
                sess.align_VR_to_2P()

                # get LR value for each trial
                lr_trial = get_LR_trial(sess)

                # make position binned lick rates and speed
                sess.add_timeseries(licks=sess.vr_data['lick']._values, speed=sess.vr_data['dz']._values)
                sess.add_pos_binned_trial_matrix(('licks', 'speed'), 't', min_pos=6, max_pos=43, bin_size=1,
                                                 mat_only=True)

                licks_rz_early = sess.trial_matrices['licks'][:, rzone_early].mean(axis=-1)

                if i == 0 and sess_ind == 0:
                    baseline = np.mean(licks_rz_early[lr_trial == -1])
                else:
                    licks = licks_rz_early[lr_trial == -1] / baseline
                    licks[np.isnan(licks)] = 0
                    early_rzone_licks.append(licks)

        #         f, ax = plt.subplots()
        early_rzone_licks = np.concatenate(early_rzone_licks)
        lr = linreg().fit(np.arange(40)[:, np.newaxis], early_rzone_licks[:40])
        slopes[m] = lr.coef_
        #         ax.plot(early_rzone_licks)

        #         ax.plot(sp.ndimage.filters.gaussian_filter1d(early_rzone_licks,5))

        RZONE_LICKS[mouse] = early_rzone_licks
    return RZONE_LICKS, slopes
    # fit a spline to the data


def second_reversal(SessDict):
    rzone_early = slice(25 - 6, 32 - 6)
    RZONE_LICKS = {}
    for m, (mouse, days) in enumerate(SessDict.items()):
        print(mouse)
        LR, LICKS, SPEED = [], [], []
        transition_trials = []
        early_rzone_licks = []
        day = days[-1]

        for sess_ind, session in enumerate(day):
            sess = TwoPUtils.sess.Session(basedir_VR=basedir_VR, mouse=mouse, date=session['date'],
                                          scene=session['scene'],
                                          session=session['session'], VR_only=True, prompt_for_keys=False)
            sess.align_VR_to_2P()

            # get LR value for each trial
            lr_trial = get_LR_trial(sess)

            # make position binned lick rates and speed
            sess.add_timeseries(licks=sess.vr_data['lick']._values, speed=sess.vr_data['dz']._values)
            sess.add_pos_binned_trial_matrix(('licks', 'speed'), 't', min_pos=6, max_pos=43, bin_size=1, mat_only=True)

            licks_rz_early = sess.trial_matrices['licks'][:, rzone_early].mean(axis=-1)

            if sess_ind == 0:
                baseline = np.mean(licks_rz_early[lr_trial == 1])
            else:
                licks = licks_rz_early[lr_trial == 1] / baseline
                licks[np.isnan(licks)] = 0
                early_rzone_licks.append(licks)

        RZONE_LICKS[mouse] = np.concatenate(early_rzone_licks)
    return RZONE_LICKS