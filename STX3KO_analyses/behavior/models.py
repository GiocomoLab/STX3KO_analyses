<<<<<<< HEAD
import numpy as np
import scipy as sp
from scipy.optimize import curve_fit
from sklearn.model_selection import KFold


# licking accuracy vs trial for familiar arm on each day
def monoexp_decay_wint(x, a, b, c):
    '''
    monoexponential decay

    a*np.exp(-b*x) + c
    :param x:
    :param a:
    :param b:
    :param c:
    :return:
    '''
    return a * np.exp(-b * x) + c


def m0(x, a, b, c):
    '''
    mono exponential

    :param x: [2 x m], first row is independent variable, second row is grouping variable for other models
    :param a:
    :param b:
    :param c:
    :return:
    '''
    # # M0 - n params 3
    return a * np.exp(-b * x[0, :]) + c


def m1(x, a0, a1, b, c):
    '''
    monoexp_decay_wint with group dependent intercept

    4 paramaters

    :param x: np.array, [2 x m], first row is independent variable, second row is grouping variable for other models
    :param a0:
    :param a1:
    :param b:
    :param c:
    :return:
    '''
    return (a0 * x[1, :] + a1 * (1 - x[1, :])) * np.exp(-b * x[0, :]) + c


def m2(x, a, b0, b1, c):
    '''
    mono exponential with group dependent decay rate

    4 parameters

    :param x: np.array, [2 x m], first row is independent variable, second row is grouping variable for other models
    :param a:
    :param b0:
    :param b1:
    :param c:
    :return:
    '''
    return a * np.exp(-1 * (b0 * x[1, :] + b1 * (1 - x[1, :])) * x[0, :]) + c


def m3(x, a0, a1, b0, b1, c):
    '''
    mono exponential with group dependent intercept and decay rate

    5 parameters

    :param x: np.array, [2 x m], first row is independent variable, second row is grouping variable for other models
    :param a0:
    :param a1:
    :param b0:
    :param b1:
    :param c:
    :return:
    '''

    return (a0 * x[1, :] + a1 * (1 - x[1, :])) * np.exp(-1 * (b0 * x[1, :] + b1 * (1 - x[1, :])) * x[0, :]) + c


def m4(x, a, b, c0, c1):
    '''
    mono exponential with group dependent asymptote

    4 parameters
    :param x: np.array, [2 x m], first row is independent variable, second row is grouping variable for other models
    :param a:
    :param b:
    :param c0:
    :param c1:
    :return:
    '''

    return a * np.exp(-b * x[0, :]) + c0 * x[1, :] + c1 * (1 - x[1, :])


def m5(x, a0, a1, b, c0, c1):
    '''
    mono exponential with group dependent intercept and asymptote
    
    5 parameters 
    :param x: np.array, [2 x m], first row is independent variable, second row is grouping variable for other models
    :param a0: 
    :param a1: 
    :param b: 
    :param c0: 
    :param c1: 
    :return: 
    '''

    return (a0 * x[1, :] + a1 * (1 - x[1, :])) * np.exp(-b * x[0, :]) + c0 * x[1, :] + c1 * (1 - x[1, :])


def m6(x, a, b0, b1, c0, c1):
    '''
    mono exponential with group dependent slope and aysmptote

    5 parameters

    :param x: np.array, [2 x m], first row is independent variable, second row is grouping variable for other models
    :param a:
    :param b0:
    :param b1:
    :param c0:
    :param c1:
    :return:
    '''

    return a * np.exp(-1 * (b0 * x[1, :] + b1 * (1 - x[1, :])) * x[0, :]) + c0 * x[1, :] + c1 * (1 - x[1, :])


def m7(x, a0, a1, b0, b1, c0, c1):
    '''
    mono exponential with group all parameters group dependent (i.e. fits two different mono exponentials)

    6 parameters

    :param x: np.array, [2 x m], first row is independent variable, second row is grouping variable for other models
    :param a0: 
    :param a1: 
    :param b0: 
    :param b1: 
    :param c0: 
    :param c1: 
    :return: 
    '''

    return (a0 * x[1, :] + a1 * (1 - x[1, :])) * np.exp(-1 * (b0 * x[1, :] + b1 * (1 - x[1, :])) * x[0, :]) + \
           c0 * x[1, :] + c1 * (1 - x[1, :])


def sample_stderr(y, yhat, k):
    '''
    sample standard error between y and yhat

    :param y: array-like, [n,]
    :param yhat: array-like, [n,]
    :param k: degrees of freedom correction/number of parameters
    :return:
    '''
    return np.sqrt(((y - yhat) ** 2).sum() / (yhat.shape[0] - k))


def squared_error_log_likelihood(y, yhat, k):
    '''
    gaussian log likelihood (base 10) of yhat if real data is y
    assume uniform variance

    :param y: array-like, [n,]
    :param yhat: array-like, [n,]
    :param k: degrees of freedom correction
    :return:
    '''

    return np.log10(sp.stats.norm.pdf(y - yhat, loc=0, scale=sample_stderr(y, yhat, k))).sum()


def bic(y, yhat, k):
    '''
    Bayesian information criterion assuming squared error loss

    :param y: array-like, [n,], true data
    :param yhat: array-like, [n,], predicted data
    :param k: degrees of freedeom
    :return: bic,

    '''
    return k * np.log(y.shape[0]) - 2. * squared_error_log_likelihood(y, yhat, k) / np.log10(np.e)


def fit_models(x, y, crossval=False, n_folds='mice'):
    '''
    fit parameters of M0-M7

    :param x: np.array, [2 x m], first row is independent variable, second row is grouping variable for other models
              OR
              np.array, [3 x m], last row is mouse ID, used when n_folds == 'mice' for leave one mouse out
    :param y: np.array, [m,], dependent data
    :param crossval: estimate log-likelihood of model using K Fold cross validation (default False)
    :param n_folds: int, number of folds for cross validation (default 10)_
    :return: BIC: np.array, [8,], Bayesian information criterion for each model
            ll: np.array, [8,], log likelihood of data for each model
            dof: np.array, [8,], degrees of freedom of each model
            popt_list: optimal parameters for full model, output of sp.optimize.curve_fit
            ll_cv: returned if crossval is True, cross-validated log likelihood
    '''
    if n_folds == 'LOO':
        n_folds = y.shape[0]

    # initialize metrics
    bic_vec = []
    popt_list = []
    ll = []
    dof = []

    # baseline
    popt, pcov = curve_fit(m0, x, y, maxfev=int(1E6), p0=[2, .05, .75], bounds=(-10, 10))  # fit model
    bic_vec.append(bic(y, m0(x, *popt), 3))
    ll.append(squared_error_log_likelihood(y, m0(x, *popt), 1))
    dof.append(y.shape[0] - 3)
    popt_list.append(popt)

    # groupwise intercept
    popt, pcov = curve_fit(m1, x, y, maxfev=int(1E6), p0=[2, 2, .05, .75], bounds=(-10, 10))
    bic_vec.append(bic(y, m1(x, *popt), 5))
    ll.append(squared_error_log_likelihood(y, m1(x, *popt), 1))
    dof.append(y.shape[0] - 5)
    popt_list.append(popt)

    # groupwise slope
    popt, pcov = curve_fit(m2, x, y, maxfev=int(1E6), p0=[2, .05, .05, .75], bounds=(-10, 10))
    bic_vec.append(bic(y, m2(x, *popt), 5))
    ll.append(squared_error_log_likelihood(y, m2(x, *popt), 1))
    dof.append(y.shape[0] - 5)
    popt_list.append(popt)

    # groupwise slope and intercept
    popt, pcov = curve_fit(m3, x, y, maxfev=int(1E6), p0=[2, 2, .05, .05, .75], bounds=(-10, 10))
    bic_vec.append(bic(y, m3(x, *popt), 6))
    ll.append(squared_error_log_likelihood(y, m3(x, *popt), 1))
    dof.append(y.shape[0] - 6)
    popt_list.append(popt)

    # groupwise asymptote
    popt, pcov = curve_fit(m4, x, y, maxfev=int(1E6), p0=[2, .05, .75, .75], bounds=(-10, 10))
    bic_vec.append(bic(y, m4(x, *popt), 5))
    ll.append(squared_error_log_likelihood(y, m4(x, *popt), 1))
    dof.append(y.shape[0] - 5)
    popt_list.append(popt)

    # groupwise intercept and asymptote
    popt, pcov = curve_fit(m5, x, y, maxfev=int(1E6), p0=[2, 2, .05, .75, .75], bounds=(-10, 10))
    bic_vec.append(bic(y, m5(x, *popt), 6))
    ll.append(squared_error_log_likelihood(y, m5(x, *popt), 1))
    dof.append(y.shape[0] - 6)
    popt_list.append(popt)

    # groupwise slope and asymptote
    popt, pcov = curve_fit(m6, x, y, maxfev=int(1E6), p0=[2, 2, .05, .75, .75], bounds=(-10, 10))
    bic_vec.append(bic(y, m6(x, *popt), 6))
    ll.append(squared_error_log_likelihood(y, m6(x, *popt), 1))
    dof.append(y.shape[0] - 6)
    popt_list.append(popt)

    # groupwise intercept, slope, and asymptote
    popt, pcov = curve_fit(m7, x, y, maxfev=int(1E6), p0=[2, 2, .05, .05, .75, .75], bounds=(-10, 10))
    bic_vec.append(bic(y, m7(x, *popt), 7))
    ll.append(squared_error_log_likelihood(y, m7(x, *popt), 1))
    dof.append(y.shape[0] - 7)
    popt_list.append(popt)

    bic_vec = np.array(bic_vec) - bic_vec[0]

    def cv_train_test(_x_train, _y_train, _x_test, _y_test):
        _ll = np.zeros([8, ])
        # baseline
        popt, pcov = curve_fit(m0, _x_train, _y_train, maxfev=int(1E5), p0=[2, .05, .75],
                               bounds=(-10, 10))
        _ll[0] = squared_error_log_likelihood(_y_test, m0(_x_test, *popt), 1).sum()

        # groupwise intercept
        popt, pcov = curve_fit(m1, _x_train, _y_train, maxfev=int(1E5), p0=[2, 2, .05, .75],
                               bounds=(-10, 10))
        _ll[1] = squared_error_log_likelihood(_y_test, m1(_x_test, *popt), 1).sum()

        # groupwise slope
        popt, pcov = curve_fit(m2, _x_train, _y_train, maxfev=int(1E5), p0=[2, .05, .05, .75],
                               bounds=(-10, 10))
        _ll[2] = squared_error_log_likelihood(_y_test, m2(_x_test, *popt), 1).sum()

        # groupwise slope and intercept
        popt, pcov = curve_fit(m3, _x_train, _y_train, maxfev=int(1E5), p0=[2, 2, .05, .05, .75],
                               bounds=(-10, 10))
        _ll[3] = squared_error_log_likelihood(_y_test, m3(_x_test, *popt), 1).sum()

        # groupwise asymptote
        popt, pcov = curve_fit(m4, _x_train, _y_train, maxfev=int(1E5), p0=[2, .05, .75, .75],
                               bounds=(-10, 10))
        _ll[4] = squared_error_log_likelihood(_y_test, m4(_x_test, *popt), 1).sum()

        # groupwise intercept and asymptote
        popt, pcov = curve_fit(m5, _x_train, _y_train, maxfev=int(1E5), p0=[2, 2, .05, .75, .75],
                               bounds=(-10, 10))
        _ll[5] = squared_error_log_likelihood(_y_test, m5(_x_test, *popt), 1).sum()

        # groupwise slope and asymptote
        popt, pcov = curve_fit(m6, _x_train, _y_train, maxfev=int(1E5), p0=[2, .05, .05, .75, .75],
                               bounds=(-10, 10))
        _ll[6] = squared_error_log_likelihood(_y_test, m6(_x_test, *popt), 1).sum()

        # groupwise intercept, slope, and asymptote
        popt, pcov = curve_fit(m7, _x_train, _y_train, maxfev=int(1E5), p0=[2, 2, .05, .05, .75, .75],
                               bounds=(-10, 10))
        _ll[7] = squared_error_log_likelihood(_y_test, m7(_x_test, *popt), 1).sum()
        return _ll

    if crossval:
        if n_folds == 'mice':
            ll_cv = np.zeros([int(np.unique(x[2, :]).shape[0]), bic_vec.shape[0]])
            for fold in np.unique(x[2, :]).tolist():
                mask = np.ones([y.shape[0], ]) > 0
                mask[x[2, :] == fold] = False

                x_train, y_train = x[:, mask], y[mask]
                x_test, y_test = x[:, ~mask], y[~mask]
                ll_cv[int(fold), :] = cv_train_test(x_train, y_train, x_test, y_test)

        else:
            kf = KFold(n_splits=n_folds, shuffle=True)
            ll_cv = np.zeros([n_folds, bic_vec.shape[0]])
            for fold, (train, test) in enumerate(kf.split(x.T)):
                x_train, y_train = x[:, train], y[train]
                x_test, y_test = x[:, test], y[test]

                ll_cv[fold, :] = cv_train_test(x_train, y_train, x_test, y_test)

        return bic_vec, np.array(ll), np.array(dof), popt_list, ll_cv.sum(axis=0)

    else:
        return bic_vec, np.array(ll), np.array(dof), popt_list
=======
import numpy as np
import scipy as sp
from scipy.optimize import curve_fit
from sklearn.model_selection import KFold


# licking accuracy vs trial for familiar arm on each day
def monoexp_decay_wint(x, a, b, c):
    '''
    monoexponential decay

    a*np.exp(-b*x) + c
    :param x:
    :param a:
    :param b:
    :param c:
    :return:
    '''
    return a * np.exp(-b * x) + c


def m0(x, a, b, c):
    '''
    mono exponential

    :param x: [2 x m], first row is independent variable, second row is grouping variable for other models
    :param a:
    :param b:
    :param c:
    :return:
    '''
    # # M0 - n params 3
    return a * np.exp(-b * x[0, :]) + c


def m1(x, a0, a1, b, c):
    '''
    monoexp_decay_wint with group dependent intercept

    4 paramaters

    :param x: np.array, [2 x m], first row is independent variable, second row is grouping variable for other models
    :param a0:
    :param a1:
    :param b:
    :param c:
    :return:
    '''
    return (a0 * x[1, :] + a1 * (1 - x[1, :])) * np.exp(-b * x[0, :]) + c


def m2(x, a, b0, b1, c):
    '''
    mono exponential with group dependent decay rate

    4 parameters

    :param x: np.array, [2 x m], first row is independent variable, second row is grouping variable for other models
    :param a:
    :param b0:
    :param b1:
    :param c:
    :return:
    '''
    return a * np.exp(-1 * (b0 * x[1, :] + b1 * (1 - x[1, :])) * x[0, :]) + c


def m3(x, a0, a1, b0, b1, c):
    '''
    mono exponential with group dependent intercept and decay rate

    5 parameters

    :param x: np.array, [2 x m], first row is independent variable, second row is grouping variable for other models
    :param a0:
    :param a1:
    :param b0:
    :param b1:
    :param c:
    :return:
    '''

    return (a0 * x[1, :] + a1 * (1 - x[1, :])) * np.exp(-1 * (b0 * x[1, :] + b1 * (1 - x[1, :])) * x[0, :]) + c


def m4(x, a, b, c0, c1):
    '''
    mono exponential with group dependent asymptote

    4 parameters
    :param x: np.array, [2 x m], first row is independent variable, second row is grouping variable for other models
    :param a:
    :param b:
    :param c0:
    :param c1:
    :return:
    '''

    return a * np.exp(-b * x[0, :]) + c0 * x[1, :] + c1 * (1 - x[1, :])


def m5(x, a0, a1, b, c0, c1):
    '''
    mono exponential with group dependent intercept and asymptote
    
    5 parameters 
    :param x: np.array, [2 x m], first row is independent variable, second row is grouping variable for other models
    :param a0: 
    :param a1: 
    :param b: 
    :param c0: 
    :param c1: 
    :return: 
    '''

    return (a0 * x[1, :] + a1 * (1 - x[1, :])) * np.exp(-b * x[0, :]) + c0 * x[1, :] + c1 * (1 - x[1, :])


def m6(x, a, b0, b1, c0, c1):
    '''
    mono exponential with group dependent slope and aysmptote

    5 parameters

    :param x: np.array, [2 x m], first row is independent variable, second row is grouping variable for other models
    :param a:
    :param b0:
    :param b1:
    :param c0:
    :param c1:
    :return:
    '''

    return a * np.exp(-1 * (b0 * x[1, :] + b1 * (1 - x[1, :])) * x[0, :]) + c0 * x[1, :] + c1 * (1 - x[1, :])


def m7(x, a0, a1, b0, b1, c0, c1):
    '''
    mono exponential with group all parameters group dependent (i.e. fits two different mono exponentials)

    6 parameters

    :param x: np.array, [2 x m], first row is independent variable, second row is grouping variable for other models
    :param a0: 
    :param a1: 
    :param b0: 
    :param b1: 
    :param c0: 
    :param c1: 
    :return: 
    '''

    return (a0 * x[1, :] + a1 * (1 - x[1, :])) * np.exp(-1 * (b0 * x[1, :] + b1 * (1 - x[1, :])) * x[0, :]) + \
           c0 * x[1, :] + c1 * (1 - x[1, :])


def sample_stderr(y, yhat, k):
    '''
    sample standard error between y and yhat

    :param y: array-like, [n,]
    :param yhat: array-like, [n,]
    :param k: degrees of freedom correction/number of parameters
    :return:
    '''
    return np.sqrt(((y - yhat) ** 2).sum() / (yhat.shape[0] - k))


def squared_error_log_likelihood(y, yhat, k):
    '''
    gaussian log likelihood (base 10) of yhat if real data is y
    assume uniform variance

    :param y: array-like, [n,]
    :param yhat: array-like, [n,]
    :param k: degrees of freedom correction
    :return:
    '''

    return np.log10(sp.stats.norm.pdf(y - yhat, loc=0, scale=sample_stderr(y, yhat, k))).sum()


def bic(y, yhat, k):
    '''
    Bayesian information criterion assuming squared error loss

    :param y: array-like, [n,], true data
    :param yhat: array-like, [n,], predicted data
    :param k: degrees of freedeom
    :return: bic,

    '''
    return k * np.log(y.shape[0]) - 2. * squared_error_log_likelihood(y, yhat, k) / np.log10(np.e)


def fit_models(x, y, crossval=False, n_folds='mice'):
    '''
    fit parameters of M0-M7

    :param x: np.array, [2 x m], first row is independent variable, second row is grouping variable for other models
              OR
              np.array, [3 x m], last row is mouse ID, used when n_folds == 'mice' for leave one mouse out
    :param y: np.array, [m,], dependent data
    :param crossval: estimate log-likelihood of model using K Fold cross validation (default False)
    :param n_folds: int, number of folds for cross validation (default 10)_
    :return: BIC: np.array, [8,], Bayesian information criterion for each model
            ll: np.array, [8,], log likelihood of data for each model
            dof: np.array, [8,], degrees of freedom of each model
            popt_list: optimal parameters for full model, output of sp.optimize.curve_fit
            ll_cv: returned if crossval is True, cross-validated log likelihood
    '''
    if n_folds == 'LOO':
        n_folds = y.shape[0]

    # initialize metrics
    bic_vec = []
    popt_list = []
    ll = []
    dof = []

    # baseline
    popt, pcov = curve_fit(m0, x, y, maxfev=int(1E6), p0=[2, .05, .75], bounds=(-20, 20))  # fit model
    bic_vec.append(bic(y, m0(x, *popt), 3))
    ll.append(squared_error_log_likelihood(y, m0(x, *popt), 1))
    dof.append(y.shape[0] - 3)
    popt_list.append(popt)

    # groupwise intercept
    popt, pcov = curve_fit(m1, x, y, maxfev=int(1E6), p0=[2, 2, .05, .75], bounds=(-20, 20))
    bic_vec.append(bic(y, m1(x, *popt), 5))
    ll.append(squared_error_log_likelihood(y, m1(x, *popt), 1))
    dof.append(y.shape[0] - 5)
    popt_list.append(popt)

    # groupwise slope
    popt, pcov = curve_fit(m2, x, y, maxfev=int(1E6), p0=[2, .05, .05, .75], bounds=(-20, 20))
    bic_vec.append(bic(y, m2(x, *popt), 5))
    ll.append(squared_error_log_likelihood(y, m2(x, *popt), 1))
    dof.append(y.shape[0] - 5)
    popt_list.append(popt)

    # groupwise slope and intercept
    popt, pcov = curve_fit(m3, x, y, maxfev=int(1E5), p0=[2, 2, .05, .05, .75], bounds=(-20, 20))
    bic_vec.append(bic(y, m3(x, *popt), 6))
    ll.append(squared_error_log_likelihood(y, m3(x, *popt), 1))
    dof.append(y.shape[0] - 6)
    popt_list.append(popt)

    # groupwise asymptote
    popt, pcov = curve_fit(m4, x, y, maxfev=int(1E5), p0=[2, .05, .75, .75], bounds=(-20, 20))
    bic_vec.append(bic(y, m4(x, *popt), 5))
    ll.append(squared_error_log_likelihood(y, m4(x, *popt), 1))
    dof.append(y.shape[0] - 5)
    popt_list.append(popt)

    # groupwise intercept and asymptote
    popt, pcov = curve_fit(m5, x, y, maxfev=int(1E5), p0=[2, 2, .05, .75, .75], bounds=(-20, 20))
    bic_vec.append(bic(y, m5(x, *popt), 6))
    ll.append(squared_error_log_likelihood(y, m5(x, *popt), 1))
    dof.append(y.shape[0] - 6)
    popt_list.append(popt)

    # groupwise slope and asymptote
    popt, pcov = curve_fit(m6, x, y, maxfev=int(1E5), p0=[2, 2, .05, .75, .75], bounds=(-20, 20))
    bic_vec.append(bic(y, m6(x, *popt), 6))
    ll.append(squared_error_log_likelihood(y, m6(x, *popt), 1))
    dof.append(y.shape[0] - 6)
    popt_list.append(popt)

    # groupwise intercept, slope, and asymptote
    popt, pcov = curve_fit(m7, x, y, maxfev=int(1E5), p0=[2, 2, .05, .05, .75, .75], bounds=(-20, 20))
    bic_vec.append(bic(y, m7(x, *popt), 7))
    ll.append(squared_error_log_likelihood(y, m7(x, *popt), 1))
    dof.append(y.shape[0] - 7)
    popt_list.append(popt)

    bic_vec = np.array(bic_vec) - bic_vec[0]

    def cv_train_test(_x_train, _y_train, _x_test, _y_test):
        _ll = np.zeros([8, ])
        # baseline
        popt, pcov = curve_fit(m0, _x_train, _y_train, maxfev=int(1E5), p0=[2, .05, .75],
                               bounds=(-20, 20))
        _ll[0] = squared_error_log_likelihood(_y_test, m0(_x_test, *popt), 1).sum()

        # groupwise intercept
        popt, pcov = curve_fit(m1, _x_train, _y_train, maxfev=int(1E5), p0=[2, 2, .05, .75],
                               bounds=(-20, 20))
        _ll[1] = squared_error_log_likelihood(_y_test, m1(_x_test, *popt), 1).sum()

        # groupwise slope
        popt, pcov = curve_fit(m2, _x_train, _y_train, maxfev=int(1E5), p0=[2, .05, .05, .75],
                               bounds=(-20, 20))
        _ll[2] = squared_error_log_likelihood(_y_test, m2(_x_test, *popt), 1).sum()

        # groupwise slope and intercept
        popt, pcov = curve_fit(m3, _x_train, _y_train, maxfev=int(1E5), p0=[2, 2, .05, .05, .75],
                               bounds=(-20, 20))
        _ll[3] = squared_error_log_likelihood(_y_test, m3(_x_test, *popt), 1).sum()

        # groupwise asymptote
        popt, pcov = curve_fit(m4, _x_train, _y_train, maxfev=int(1E5), p0=[2, .05, .75, .75],
                               bounds=(-20, 20))
        _ll[4] = squared_error_log_likelihood(_y_test, m4(_x_test, *popt), 1).sum()

        # groupwise intercept and asymptote
        popt, pcov = curve_fit(m5, _x_train, _y_train, maxfev=int(1E5), p0=[2, 2, .05, .75, .75],
                               bounds=(-20, 20))
        _ll[5] = squared_error_log_likelihood(_y_test, m5(_x_test, *popt), 1).sum()

        # groupwise slope and asymptote
        popt, pcov = curve_fit(m6, _x_train, _y_train, maxfev=int(1E5), p0=[2, .05, .05, .75, .75],
                               bounds=(-20, 20))
        _ll[6] = squared_error_log_likelihood(_y_test, m6(_x_test, *popt), 1).sum()

        # groupwise intercept, slope, and asymptote
        popt, pcov = curve_fit(m7, _x_train, _y_train, maxfev=int(1E5), p0=[2, 2, .05, .05, .75, .75],
                               bounds=(-20, 20))
        _ll[7] = squared_error_log_likelihood(_y_test, m7(_x_test, *popt), 1).sum()
        return _ll

    if crossval:
        if n_folds == 'mice':
            ll_cv = np.zeros([np.amax(x[2, :]), bic_vec.shape[0]])
            for fold in np.unique(x[2, :]).tolist():
                mask = np.ones([y.shape[0], ]) > 0
                mask[x[2, :] == fold] = False

                x_train, y_train = x[:, mask], y[mask]
                x_test, y_test = x[:, ~mask], y[~mask]
                ll_cv[fold, :] = cv_train_test(x_train, y_train, x_test, y_test)

        else:
            kf = KFold(n_splits=n_folds, shuffle=True)
            ll_cv = np.zeros([n_folds, bic_vec.shape[0]])
            for fold, (train, test) in enumerate(kf.split(x.T)):
                x_train, y_train = x[:, train], y[train]
                x_test, y_test = x[:, test], y[test]

                ll_cv[fold, :] = cv_train_test(x_train, y_train, x_test, y_test)

        return bic_vec, np.array(ll), np.array(dof), popt_list, ll_cv.sum(axis=0)

    else:
        return bic_vec, np.array(ll), np.array(dof), popt_list
>>>>>>> a15b0e0 (local)
