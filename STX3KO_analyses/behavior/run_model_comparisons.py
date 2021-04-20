from . import model_comparison, models
import numpy as np
from matplotlib import pyplot as plt

## familiar sessions
def fam_run_fit(all_sessions, l0, l1, metric, day, crossval=True):
    '''


    :param all_sessions:
    :param l0:
    :param l1:
    :param metric:
    :param day:
    :param crossval:
    :return:
    '''
    x = []
    y = []
    for mouse, d in {mouse: all_sessions[mouse][day] for mouse in l0}.items():

        if mouse == '4467975.4' and day == 0:
            mask = np.ones(d['trial_number'].shape) > 0
            mask[24 - 5:68 - 5] = False
        else:
            mask = np.ones(d['trial_number'].shape) > 0

        mask = mask * (d['block_number'] < 5) * (d['probes'] == 0) * (d['omissions'] == 0)
        _y = d[metric][mask]
        trials = np.arange(mask.shape[0])
        trials = trials[mask]

        y.append(_y)
        _x = np.zeros([2, trials.shape[0]])
        _x[0, :] = trials
        x.append(_x)

    for mouse, d in {mouse: all_sessions[mouse][day] for mouse in l1}.items():

        if mouse == '4467975.4' and day == 0:
            mask = np.ones(d['trial_number'].shape) > 0
            mask[24 - 5:68 - 5] = False
        else:
            mask = np.ones(d['trial_number'].shape) > 0

        mask = mask * (d['block_number'] < 5) * (d['probes'] == 0) * (d['omissions'] == 0)
        _y = d[metric][mask]
        trials = np.arange(mask.shape[0])
        trials = trials[mask]

        y.append(_y)
        _x = np.ones([2, trials.shape[0]])
        _x[0, :] = trials
        x.append(_x)

    y = np.concatenate(y, axis=-1)
    x = np.concatenate(x, axis=-1)

    random_order = np.random.permutation(int(y.shape[0]))
    x = x[:, random_order]
    y = y[random_order]
    return models.fit_models(x, y, crossval=crossval)
    # bic_vec, SSE, dof, popt_list, SSE_cv = fit_models(X, acc, crossval=True)


def run_model_comparisons_familiar(ko_sessions, ctrl_sessions, metric):
    '''


    :param ko_sessions:
    :param ctrl_sessions:
    :param metric:
    :return:
    '''
    ko_mice = [k for k in ko_sessions.keys()]
    ctrl_mice = [k for k in ctrl_sessions.keys()]

    perms = model_comparison.generate_perms(ko_mice, ctrl_mice)
    all_sessions = {**ko_sessions, **ctrl_sessions}

    results = []
    for day in range(5):

        bic_vec, ll, dof, popt_list, ll_cv = fam_run_fit(all_sessions, ko_mice, ctrl_mice, metric, day, crossval=True)
        print('log likelihood', ll_cv)
        perm_ll = []
        for p, (l0, l1) in enumerate(perms):
            if p % 50 == 0:
                print('perm', p)
            _, _, _, _, _ll_cv = fam_run_fit(all_sessions, l0, l1, metric, day, crossval=True)
            perm_ll.append(_ll_cv)

        perm_ll = np.array(perm_ll)
        pvec = []
        for col in range(perm_ll.shape[1]):
            print("Day %d" % day)
            true_ll = ll_cv[col]
            _perm_ll = perm_ll[:, col]
            p_val = np.float((true_ll + 1E-3 < _perm_ll).sum()) / _perm_ll.shape[0]
            print("M%d, true log likelihood %f, lowest perm log likelihood %f, 'p' value %f" % (
            col, true_ll, np.amax(_perm_ll), p_val))
            pvec.append(p_val)

        results.append({'bic_vec': bic_vec, 'popt_list': popt_list, 'LL_cv': ll_cv, 'p_val': pvec})

    return results


def plot_fam(ko_sessions, ctrl_sessions, results, metric):
    '''


    :param ko_sessions:
    :param ctrl_sessions:
    :param results:
    :param metric:
    :return:
    '''
    fig, ax = plt.subplots(1, 5, figsize=[25, 5], sharey=True)
    for day in range(5):

        for mouse, d_list in ko_sessions.items():
            d = d_list[day]
            y = d[metric][(d['block_number'] < 5) * (d['probes'] == 0)]
            trials = np.arange(y.shape[0])

            ax[day].scatter(trials, y, alpha=.3, color='red')

        for mouse, d_list in ctrl_sessions.items():
            d = d_list[day]
            y = d[metric][(d['block_number'] < 5) * (d['probes'] == 0)]
            trials = np.arange(y.shape[0])

            ax[day].scatter(trials, y, alpha=.3, color='black')

        ll_cv = results[day]['LL_cv']
        pval = np.array(results[day]['p_val'])
        bestmodel, m = model_comparison.pick_best_model(ll_cv, pval, p_thresh=.01, llr_thresh=1)

        x = np.zeros([2, 80])
        x[0, :] = np.arange(80)

        if not bestmodel == 0:

            ax[day].plot(x[0, :], m(x, *results[day]['popt_list'][bestmodel]), color='red', linewidth=5)

            x[1, :] = 1
            ax[day].plot(x[0, :], m(x, *results[day]['popt_list'][bestmodel]), color='black', linewidth=5)

            ax[day].set_title("best model : %d , p_val %f" % (bestmodel, results[day]['p_val'][bestmodel]))
        else:
            ax[day].plot(x[0, :], m(x, *results[day]['popt_list'][bestmodel]), color='blue', linewidth=5)

    return fig, ax


## novel arm blocks


if __name__ == '__main__':

    ko_sessions, ctrl_sessions = model_comparison.get_session_dicts()
