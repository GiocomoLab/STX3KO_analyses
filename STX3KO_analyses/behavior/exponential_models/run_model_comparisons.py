from STX3KO_analyses.behavior.exponential_models import model_comparison, models
import numpy as np
from matplotlib import pyplot as plt


## familiar sessions
def fam_run_fit(all_sessions, l0, l1, metric, day, crossval=True, n_folds='mice'):
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
    mouse_counter = 0
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
        _x = np.zeros([3, trials.shape[0]])
        _x[0, :] = trials
        _x[2, :] = mouse_counter
        x.append(_x)
        mouse_counter += 1

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
        _x = np.ones([3, trials.shape[0]])
        _x[0, :] = trials
        _x[2, :] = mouse_counter

        x.append(_x)
        mouse_counter += 1

    y = np.concatenate(y, axis=-1)
    x = np.concatenate(x, axis=-1)

    random_order = np.random.permutation(int(y.shape[0]))
    x = x[:, random_order]
    y = y[random_order]
    return models.fit_models(x, y, crossval=crossval, n_folds=n_folds)


def run_model_comparisons_familiar(ko_sessions, ctrl_sessions, metric, p_thresh=.05, llr_thresh=1):
    '''


    :param llr_thresh:
    :param p_thresh:
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
        print("Day %d" % day)
        for col in range(perm_ll.shape[1]):
            true_ll = ll_cv[col]
            _perm_ll = perm_ll[:, col]
            p_val = np.float((true_ll < _perm_ll + 1E-5).sum()) / _perm_ll.shape[0]
            print("M%d, true log likelihood %f, highest perm log likelihood %f, 'p' value %f" % (
                col, true_ll, np.amax(_perm_ll), p_val))
            pvec.append(p_val)

        pvec = np.array(pvec)
        bestmodel, m = model_comparison.pick_best_model(ll_cv, pvec, p_thresh=p_thresh, llr_thresh=llr_thresh)
        results.append({'bic_vec': bic_vec,
                        'popt_list': popt_list,
                        'LL_cv': ll_cv,
                        'p_val': pvec,
                        'best_model_index': bestmodel,
                        'best_model_func': m})

    return results


## novel arm blocks
def nov_run_fit(all_sessions, l0, l1, metric, day, crossval=True, n_folds='mice'):
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
    mouse_counter = 0
    for mouse, d_list in {mouse: all_sessions[mouse] for mouse in l0}.items():
        novel = -1 * d_list[0]['LR'][0]
        d = d_list[day]
        mask = (np.ones(d['trial_number'].shape) > 0) * (d['block_number'] == 5) * (d['probes'] == 0) * (
                d['omissions'] == 0) * (d['LR'] == novel)
        _y = d[metric][mask]
        trials = np.arange(mask.shape[0])
        trials = trials[mask]

        y.append(_y)
        _x = np.zeros([3, trials.shape[0]])
        _x[0, :] = trials
        _x[2, :] = mouse_counter
        x.append(_x)
        if _y.shape[0] > 0:
            mouse_counter += 1

    for mouse, d_list in {mouse: all_sessions[mouse] for mouse in l1}.items():
        novel = -1 * d_list[0]['LR'][0]
        d = d_list[day]
        mask = (np.ones(d['trial_number'].shape) > 0) * (d['block_number'] == 5) * (d['probes'] == 0) * (
                d['omissions'] == 0) * (d['LR'] == novel)
        _y = d[metric][mask]
        trials = np.arange(mask.shape[0])
        trials = trials[mask]

        y.append(_y)
        _x = np.ones([3, trials.shape[0]])
        _x[0, :] = trials
        _x[2, :] = mouse_counter

        x.append(_x)
        if _y.shape[0] > 0:
            mouse_counter += 1

    y = np.concatenate(y, axis=-1)
    x = np.concatenate(x, axis=-1)

    random_order = np.random.permutation(int(y.shape[0]))
    x = x[:, random_order]
    y = y[random_order]
    return models.fit_models(x, y, crossval=crossval, n_folds=n_folds)


def run_model_comparisons_novel(ko_sessions, ctrl_sessions, metric, p_thresh=.01, llr_thresh=1):
    '''


    :param llr_thresh:
    :param p_thresh:
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

        bic_vec, ll, dof, popt_list, ll_cv = nov_run_fit(all_sessions, ko_mice, ctrl_mice, metric, day, crossval=True,
                                                         n_folds='mice')
        print('log likelihood', ll_cv)
        perm_ll = []
        for p, (l0, l1) in enumerate(perms):
            if p % 50 == 0:
                print('perm', p)
            _, _, _, _, _ll_cv = nov_run_fit(all_sessions, l0, l1, metric, day, crossval=True, n_folds='mice')
            perm_ll.append(_ll_cv)

        perm_ll = np.array(perm_ll)
        pvec = []
        print("Day %d" % day)
        for col in range(perm_ll.shape[1]):
            true_ll = ll_cv[col]
            _perm_ll = perm_ll[:, col]
            p_val = np.float((true_ll < _perm_ll + 1E-5).sum()) / _perm_ll.shape[0]
            print("M%d, true log likelihood %f, highest perm log likelihood %f, 'p' value %f" % (
                col, true_ll, np.amax(_perm_ll), p_val))
            pvec.append(p_val)

        pvec = np.array(pvec)
        bestmodel, m = model_comparison.pick_best_model(ll_cv, pvec, p_thresh=p_thresh, llr_thresh=llr_thresh)
        results.append({'bic_vec': bic_vec,
                        'popt_list': popt_list,
                        'LL_cv': ll_cv,
                        'p_val': pvec,
                        'best_model_index': bestmodel,
                        'best_model_func': m})

    return results


def plot_famnov_results(ko_sessions, ctrl_sessions, results, metric):
    '''


    :param ko_sessions:
    :param ctrl_sessions:
    :param results:
    :param metric:
    :return:
    '''
    fig, ax = plt.subplots(1, 5, figsize=[25, 5], sharey=True)
    for day in range(5):
        ymax = 0
        for mouse, d_list in ko_sessions.items():
            d = d_list[day]
            y = d[metric][(d['block_number'] < 5) * (d['probes'] == 0)]
            trials = np.arange(y.shape[0])

            ax[day].scatter(trials, y, alpha=.3, color='red')
            ymax = max(ymax, y.shape[0])

        for mouse, d_list in ctrl_sessions.items():
            d = d_list[day]
            y = d[metric][(d['block_number'] < 5) * (d['probes'] == 0)]
            trials = np.arange(y.shape[0])

            ax[day].scatter(trials, y, alpha=.3, color='black')
            ymax = max(ymax, y.shape[0])

        ll_cv = results[day]['LL_cv']
        pval = np.array(results[day]['p_val'])
        bestmodel, m = model_comparison.pick_best_model(ll_cv, pval, p_thresh=.01, llr_thresh=1)

        x = np.zeros([2, ymax])
        x[0, :] = np.arange(ymax)

        if not bestmodel == 0:

            ax[day].plot(x[0, :], m(x, *results[day]['popt_list'][bestmodel]), color='red', linewidth=5)

            x[1, :] = 1
            ax[day].plot(x[0, :], m(x, *results[day]['popt_list'][bestmodel]), color='black', linewidth=5)

            ax[day].set_title("best model : %d , p_val %f" % (bestmodel, results[day]['p_val'][bestmodel]))
        else:
            ax[day].plot(x[0, :], m(x, *results[day]['popt_list'][bestmodel]), color='blue', linewidth=5)

    return fig, ax


## reward reversal
def concat_rz_lickrate(d_list):
    '''


    :param d_list:
    :return:
    '''
    lr_rz_early_norm = []
    lr_rz_late_norm = []
    for j, d in enumerate(d_list):
        if j == 0:
            baseline_early_inds = (d['block_number'] < 2) * (d['LR'] == -1)
            baseline_late_inds = (d['block_number'] < 2) * (d['LR'] == 1)
            baseline_early = d['lickrate_rz_early'][baseline_early_inds].mean()
            baseline_late = d['lickrate_rz_late'][baseline_late_inds].mean()

            nonbaseline_inds = d['block_number'] >= 2
            lr_rz_early_norm.append(d['lickrate_rz_early'][nonbaseline_inds * (d['LR'] == -1)] / baseline_early)
            lr_rz_late_norm.append(d['lickrate_rz_late'][nonbaseline_inds * (d['LR'] == 1)] / baseline_late)

        elif j == 1:
            nonbaseline_inds = d['LR'] == -1
            lr_rz_early_norm.append(d['lickrate_rz_early'][d['LR'] == -1] / baseline_early)
            lr_rz_late_norm.append(d['lickrate_rz_late'][d['LR'] == 1] / baseline_late)
        else:
            nonbaseline_inds = d['block_number'] < 2
            lr_rz_early_norm.append(d['lickrate_rz_early'][nonbaseline_inds * (d['LR'] == -1)] / baseline_early)
            lr_rz_late_norm.append(d['lickrate_rz_late'][nonbaseline_inds * (d['LR'] == 1)] / baseline_late)

    lr_rz_early_norm = np.concatenate(lr_rz_early_norm)
    lr_rz_late_norm = np.concatenate(lr_rz_late_norm)

    inds_early = np.arange(lr_rz_early_norm.shape[0])
    inds_late = np.arange(lr_rz_late_norm.shape[0])

    return {'early_rz_inds': inds_early, 'early_rz_lr': lr_rz_early_norm, 'late_rz_inds': inds_late,
            'late_rz_lr': lr_rz_late_norm}


def rev_model_fit(all_reversal_lrs, l0, l1, metric, crossval=True, n_folds='mice'):
    '''


    :param all_reversal_lrs:
    :param l0:
    :param l1:
    :param metric:
    :return:
    '''
    y = []
    x = []
    mouse_counter = 0
    for mouse, _y in {mouse: all_reversal_lrs[mouse][metric] for mouse in l0}.items():
        trials = np.arange(_y.shape[0])
        y.append(_y)

        _x = np.zeros([3, trials.shape[0]])
        _x[0, :] = trials
        _x[2, :] = mouse_counter
        x.append(_x)
        mouse_counter += 1

    for mouse, _y in {mouse: all_reversal_lrs[mouse][metric] for mouse in l1}.items():
        trials = np.arange(_y.shape[0])
        y.append(_y)
        _x = np.ones([3, trials.shape[0]])
        _x[0, :] = trials
        _x[2, :] = mouse_counter
        x.append(_x)
        mouse_counter += 1

    y = np.concatenate(y, axis=-1)
    x = np.concatenate(x, axis=-1)
    random_order = np.random.permutation(int(y.shape[0]))
    x = x[:, random_order]
    y = y[random_order]

    return models.fit_models(x, y, crossval=crossval, n_folds=n_folds)


def _run_model_comparisons_rev_lr(ko_sessions, ctrl_sessions, metric, p_thresh=.01, llr_thresh=1):
    ko_mice = [k for k in ko_sessions.keys()]
    ctrl_mice = [k for k in ctrl_sessions.keys()]

    ko_reversal_lrs = {mouse: concat_rz_lickrate(d_list[-3:]) for mouse, d_list in ko_sessions.items()}
    ctrl_reversal_lrs = {mouse: concat_rz_lickrate(d_list[-3:]) for mouse, d_list in ctrl_sessions.items()}
    all_reversal_lrs = {**ko_reversal_lrs, **ctrl_reversal_lrs}
    bic_vec, sse, dof, popt_list, ll_cv = rev_model_fit(all_reversal_lrs, ko_mice, ctrl_mice, metric)
    print('log-likelihood', ll_cv)
    perms = model_comparison.generate_perms(ko_mice, ctrl_mice)
    perm_ll = []
    print('running permutations')
    for p, (l0, l1) in enumerate(perms):
        if p % 50 == 0:
            print('perm ', p)
        _, _, _, _, _ll_cv = rev_model_fit(all_reversal_lrs, l0, l1, metric)
        perm_ll.append(_ll_cv)

    perm_ll = np.array(perm_ll)
    pvec = []
    for col in range(perm_ll.shape[1]):
        true_ll = ll_cv[col]
        _perm_ll = perm_ll[:, col]
        p_val = np.float((true_ll < _perm_ll + 1E-5).sum()) / _perm_ll.shape[0]
        print("M%d, true log likelihood %f, highest perm log likelihood %f, 'p' value %f" % (
            col, true_ll, np.amax(_perm_ll), p_val))
        pvec.append(p_val)
    pvec = np.array(pvec)
    ##

    bestmodel, m = model_comparison.pick_best_model(ll_cv, pvec, p_thresh=p_thresh, llr_thresh=llr_thresh)
    print('best model ', bestmodel)
    return {'bic_vec': bic_vec,
            'popt_list': popt_list,
            'LL_cv': ll_cv,
            'p_val': pvec,
            'best_model_index': bestmodel,
            'best_model_func': m}


def run_model_comparisons_rev_lr(ko_sessions, ctrl_sessions, metrics=None):
    if metrics is None:
        metrics = ('early_rz_lr', 'late_rz_lr')
    return [_run_model_comparisons_rev_lr(ko_sessions, ctrl_sessions, metric, p_thresh=.01, llr_thresh=1) for metric in
            metrics]


def plot_reversal_lrs(ko_sessions, ctrl_sessions, early_results, late_results):
    fig, ax = plt.subplots(1, 2, figsize=[20, 10], sharey=True)
    ko_reversal_lrs = {mouse: concat_rz_lickrate(d_list[-3:]) for mouse, d_list in ko_sessions.items()}
    ctrl_reversal_lrs = {mouse: concat_rz_lickrate(d_list[-3:]) for mouse, d_list in ctrl_sessions.items()}

    ymax = 0
    for mouse, data in ko_reversal_lrs.items():
        ax[0].scatter(data['early_rz_inds'], data['early_rz_lr'], color='red', alpha=.3)
        ax[1].scatter(data['late_rz_inds'], data['late_rz_lr'], color='red', alpha=.3)
        ymax = max(ymax, data['early_rz_lr'].shape[0], data['late_rz_lr'].shape[0])
    for mouse, data in ctrl_reversal_lrs.items():
        ax[0].scatter(data['early_rz_inds'], data['early_rz_lr'], color='black', alpha=.3)
        ax[1].scatter(data['late_rz_inds'], data['late_rz_lr'], color='black', alpha=.3)
        ymax = max(ymax, data['early_rz_lr'].shape[0], data['late_rz_lr'].shape[0])

    for ax_ind, results in enumerate((early_results, late_results)):
        p_val = np.array(results['p_val'])
        best_model, m = results['best_model_index'], results['best_model_func']

        x = np.zeros([2, ymax])
        x[0, :] = np.arange(ymax)

        if not best_model == 0:

            ax[ax_ind].plot(x[0, :], m(x, *results['popt_list'][best_model]), color='red', linewidth=5)
            x[1, :] = 1
            ax[ax_ind].plot(x[0, :], m(x, *results['popt_list'][best_model]), color='black', linewidth=5)

            ax[ax_ind].set_title("best model : %d , p_val %f" % (best_model, p_val[best_model]))
        else:
            ax[ax_ind].plot(x[0, :], m(x, *results['popt_list'][best_model]), color='blue', linewidth=5)

    return fig, ax
