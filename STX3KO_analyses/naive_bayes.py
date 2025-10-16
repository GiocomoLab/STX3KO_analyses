from STX3KO_analyses import utilities as u

import numpy as np
import scipy as sp


gamma = lambda x,kappa:  1/(sp.special.gamma(kappa)+1E-5)* np.exp(-x)*np.power(x,kappa-1)
poiss = lambda k,lam:  1/(sp.special.gamma(k)+1E-5)* np.exp(-lam)*np.power(lam,k)

def logsumexp(x,axis=-1):
    c = x.max(axis = axis, keepdims=True) + 1E-5
    return c + np.log(np.sum(np.exp(x - c ),axis=axis, keepdims=True))


def crossval(sess, n_cells = -1, poisson = True, nov = False, chan=None):
    
    # assuming uniform prior over bins
    if nov:
        trial_mask = sess.trial_info['LR']==sess.novel_arm
    else:
        trial_mask = sess.trial_info['LR']==-1*sess.novel_arm
    
    if chan is None:
        trial_mat = sess.trial_matrices['F_dff'][trial_mask,:,:]#*10
    else:
        trial_mat = sess.trial_matrices[f'{chan}_F_dff'][trial_mask,:,:]#*10
    
    trial_mat -= np.nanmin(trial_mat,axis=-1,keepdims=True)
    trial_mat[np.isnan(trial_mat)]=0
    trial_mat += 1E-3

    # select cells
    if n_cells ==-1:
#         print('all_cells')
        pass
    else:
        rng = np.random.default_rng()
        trial_mat = trial_mat[:,:,rng.permutation(trial_mat.shape[-1])[:n_cells]]
    
    posterior = np.zeros([trial_mat.shape[0], trial_mat.shape[1], trial_mat.shape[1]])
    for trial in range(trial_mat.shape[0]):
        
        mask = np.zeros((trial_mat.shape[0],))<1
        mask[trial] = False
        
        trial_mat_mean = np.nanmean(trial_mat[mask,:,:],axis=0)
        
        y = np.copy(trial_mat[trial,:,:])
        y[np.isnan(y)]=1E-3
        log_likelihood = 0
        for cell in range(trial_mat.shape[-1]):
            if poisson:
                log_likelihood += np.log(poiss(y[:,cell:cell+1], trial_mat_mean[:,cell:cell+1].T)+1E-3)
            else:
                log_likelihood += np.log(gamma(y[:,cell:cell+1], trial_mat_mean[:,cell:cell+1].T)+1E-3)
        
        posterior[trial,:,:] = np.exp(log_likelihood - logsumexp(log_likelihood))
        

    
    return posterior


def test(_y, trial_mat_mean, poisson=False):
    log_likelihood = 0
    for cell in range(trial_mat_mean.shape[-1]):
        yc, tmm = _y[:,cell], trial_mat_mean[:,cell]
        if poisson:
            log_likelihood += np.log(poiss(yc[:,np.newaxis], tmm[np.newaxis,:])+1E-3)
        else:
            log_likelihood += np.log(gamma(yc[:,np.newaxis], tmm[np.newaxis,:])+1E-3)
    return np.exp(log_likelihood - logsumexp(log_likelihood))

def abs_err(post):
    err = np.zeros([*post.shape[:2]])*np.nan
    true_pos = np.arange(0,post.shape[1])
    for trial in range(post.shape[0]):
        decode_pos = np.argmax(post[trial,:,:],axis=-1)
        err[trial,:] = np.abs(decode_pos-true_pos)
    return err

def run_ncells_baseline_crossval(mice, day = 0):
    results = {}
    for mouse in mice:
        print('mouse', mouse)
        results[mouse]={}
        sess = u.load_single_day(mouse,day=day)
        for n_cells in [2**n for n in range(3,9)]:
            print('n cells', n_cells)
            fam_post = []
            nov_post = []
            for rep in range(50):
                fam_post.append(crossval(sess,n_cells=n_cells, nov=False))
                nov_post.append(crossval(sess,n_cells=n_cells, nov = True))
            fam_post = np.concatenate(fam_post,axis=0)
            nov_post = np.concatenate(nov_post,axis=0)
            results[mouse][n_cells]={'fam': fam_post, 'nov':nov_post}
    return results

def run_alldays_baseline_crossval(mice, n_cells):
    results = {}
    for mouse in mice:
        print('mouse', mouse)
        results[mouse]={}
        for day in range(6):
            results[mouse][day]={}
            sess = u.load_single_day(mouse,day=day)
            
            fam_post = []
            nov_post = []
            for rep in range(10):
                fam_post.append(crossval(sess,n_cells=n_cells, nov=False))
                nov_post.append(crossval(sess, n_cells=n_cells, nov=True))
            fam_post = np.concatenate(fam_post, axis=0)
            nov_post = np.concatenate(nov_post, axis=0)
            
            results[mouse][day] = {'fam': fam_post, 'nov': nov_post}
    return results

def run_alldays_baseline_crossval_sparse(mice, n_cells):
    results = {}
    for mouse in mice:
        print('mouse', mouse)
        results[mouse]={}
        for day in range(6):
            if mouse == 'SparseKO_09' and day ==2:
                continue
            results[mouse][day]={}
            sess = u.load_single_day(mouse,day=day)
            
            for chan in ('channel_0', 'channel_1'):
                results[mouse][day][chan]={}
                fam_post = []
                nov_post = []
                for rep in range(10):
                    fam_post.append(crossval(sess,n_cells=n_cells, nov=False, chan=chan))
                    nov_post.append(crossval(sess, n_cells=n_cells, nov=True, chan=chan))
                fam_post = np.concatenate(fam_post, axis=0)
                nov_post = np.concatenate(nov_post, axis=0)
            
                results[mouse][day][chan] = {'fam': fam_post, 'nov': nov_post}
    return results