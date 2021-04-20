import os
import numpy as np
from .. import ymaze_sess_deets, session
from . import trial_metrics


def get_session_dicts(pklbase='/home/mplitt/YMazeSessPkls/'):
    '''


    :param pklbase:
    :return:
    '''

    def combine_sessions(trial_info_list):
        '''

        :param trial_info_list:
        :return:
        '''
        combined_dict = {}
        for k, v in trial_info_list[0].items():
            combined_dict[k] = [v]

        for i, _dicts in enumerate(trial_info_list[1:]):
            for k, v in _dicts.items():
                if k == 'block_number':
                    #                 print(trial_info_list[i]['block_number'][-1])
                    combined_dict[k].append(v + trial_info_list[i]['block_number'][-1] + 1)
                else:
                    combined_dict[k].append(v)

        for k, v in combined_dict.items():
            combined_dict[k] = np.concatenate(combined_dict[k])

        return combined_dict

    def build_dict(sessions_dict):
        out_dict = {}
        keys = ['antic_licks', 'speed', 'antic_speed']  # keys to save
        for mouse, sessions in sessions_dict.items():
            # print(sessions)
            pkldir = os.path.join(pklbase, mouse)
            out_dict[mouse] = []
            for deets in sessions:
                print(mouse, deets)
                if isinstance(deets, tuple):
                    sess_list = []
                    for _deets in deets:
                        sess = session.YMazeSession.from_file(
                            os.path.join(pkldir, _deets['date'], "%s_%d.pkl" % (_deets['scene'], _deets['session'])),
                            verbose=False)
                        trial_metrics.antic_consum_licks(sess)
                        trial_metrics.get_probes_and_omissions(sess)
                        trial_metrics.single_trial_lick_metrics(sess)
                        trial_matrices = {k: sess.trial_matrices[k] for k in keys}

                        sess_list.append({**_deets, **sess.trial_info, **trial_matrices})
                    combined = combine_sessions(sess_list)
                    out_dict[mouse].append({**combined, 'trial_number': np.arange(combined['LR'].shape[0])})
                else:

                    sess = session.YMazeSession.from_file(
                        os.path.join(pkldir, deets['date'], "%s_%d.pkl" % (deets['scene'], deets['session'])),
                        verbose=False)
                    trial_metrics.antic_consum_licks(sess)
                    trial_metrics.get_probes_and_omissions(sess)
                    trial_metrics.single_trial_lick_metrics(sess)
                    trial_matrices = {k: sess.trial_matrices[k] for k in keys}
                    out_dict[mouse].append({**deets, **sess.trial_info, **trial_matrices,
                                            'trial_number': np.arange(sess.trial_info['LR'].shape[0])})

        return out_dict

    return build_dict(ymaze_sess_deets.KO_sessions), build_dict(ymaze_sess_deets.CTRL_sessions)


def pick_best_model(ll_cv, pval, p_thresh=.01, llr_thresh=1):
    '''


    :param ll_cv:
    :param pval:
    :param p_thresh:
    :param llr_thresh
    :return:
    '''

    # first stage
    # M0 baseline model
    # M1 groupwise intercept
    # M2 groupwise slope
    # M3 groupwise slope and intercept
    # M4 groupwise asymptote
    # M5 groupwise intercept and asymptote
    # M6 groupwise slope and asymptote
    # M7 groupwise intercept, slope, and asymptote

    onep_inds = [1, 2, 4]
    onep_LLR = ll_cv[onep_inds] - ll_cv[0]
    onep_pval = pval[onep_inds]

    best_onep_ind = np.argmax(onep_LLR)
    if (onep_pval[best_onep_ind] < .01) and onep_LLR[best_onep_ind] > 5:

        if best_onep_ind == 0:
            twop_inds = [3, 5]
            twop_LLR = ll_cv[twop_inds] - ll_cv[onep_inds[0]]
            twop_pval = pval[twop_inds]


        elif best_onep_ind == 1:
            twop_inds = [3, 6]
            twop_LLR = ll_cv[twop_inds] - ll_cv[onep_inds[1]]
            twop_pval = pval[twop_inds]
        elif best_onep_ind == 2:
            twop_inds = [5, 6]
            twop_LLR = ll_cv[twop_inds] - ll_cv[onep_inds[2]]
            twop_pval = pval[twop_inds]
        else:
            raise Exception("One paramater model improperly indexed")

        assert isinstance(twop_LLR, object)
        best_twop_ind = np.argmax(twop_LLR)
        if (twop_LLR[best_twop_ind] > 5) and (twop_pval[best_twop_ind] < .01):

            threep_LLR = ll_cv[-1] - ll_cv[twop_inds[best_twop_ind]]
            if (threep_LLR > 5) and (pval[-1] < .01):
                bestmodel = 7
            else:
                bestmodel = twop_inds[best_twop_ind]

        else:
            bestmodel = onep_inds[best_onep_ind]

    else:
        bestmodel = 0

    return bestmodel
