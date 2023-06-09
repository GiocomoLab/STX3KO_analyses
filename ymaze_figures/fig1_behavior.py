import os
import math
import dill
from itertools import permutations

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from matplotlib import gridspec
import pandas as pd
from pingouin import mixed_anova, anova, pairwise_tukey, pairwise_ttests
from statsmodels.regression.mixed_linear_model import MixedLM

import suite2p as s2p

import TwoPUtils as tpu
import STX3KO_analyses as stx
from STX3KO_analyses import utilities as u

ko_mice = stx.ymaze_sess_deets.ko_vr_mice
ctrl_mice = stx.ymaze_sess_deets.ctrl_vr_mice
plt.rcParams['pdf.fonttype']=42


fig_dir = "/mnt/BigDisk/YMazeResults/behavior"
os.makedirs(fig_dir,exist_ok=True)
dt = 15.46


def left_right_avg_lick_plots():
    # Left-right licking plots across days
    fig, ax = plt.subplots(2,6,figsize=[30,10], sharey=True)

    sess = u.load_vr_day(ctrl_mice[0], 0, 
                         trial_mat_keys = ('licks','nonconsum_licks','speed'),
                         verbose = False)
    
    t_bins = sess.trial_matrices['bin_edges'][1:]
    rzone_early = (t_bins>=sess.rzone_early['tfront']-3)*(t_bins<=sess.rzone_early['tfront'])
    rzone_late = (t_bins>=sess.rzone_late['tfront']-3)*(t_bins<=sess.rzone_late['tfront'])
    
    # mark positions of reward zones
    for day in range(6):
        ax[1,day].fill_betweenx([0,5], sess.rzone_late['tfront'], sess.rzone_late['tback'],zorder=0, color='green',alpha=.3)
        ax[0,day].fill_betweenx([0,5],sess.rzone_early['tfront'], sess.rzone_early['tback'],zorder=0, color='blue',alpha=.3)
    


    for day in range(6):
        
        for z, (mice, plot_color) in enumerate(zip((ctrl_mice, ko_mice), ('black', 'red'))):
            for m, mouse in enumerate(mice):
                sess = u.load_vr_day(mouse,day, trial_mat_keys = ('licks','nonconsum_licks','speed'),verbose = False)
            
                if m==0 and z==0:
                    trial_mask = sess.trial_info['LR']==-1
                    if mice == ko_mice and day==0 and i == 3:
                        trial_mask[35:66]=False

                ax[0,day].plot(sess.trial_matrices['bin_centers'], dt*sess.trial_matrices['licks'][trial_mask,:].mean(axis=0), color=plot_color)
                ax[1,day].plot(sess.trial_matrices['bin_centers'], dt*sess.trial_matrices['licks'][sess.trial_info['LR']==1,:].mean(axis=0), color=plot_color)

    path = os.path.join(fig_dir, 'left_right_lickrate_all_days.pdf')
    fig.savefig(path, format='pdf')
    return None


def fam_nov_lickrate_across_days(key='nonconsum_licks'):
# fam vs nov lick rates across days


    # build dataframe for anova
    # build dictionary for plots
    df = pd.DataFrame({'mouse':[],
                        'cond':[],
                        'day':[],
                        'nov':[],
                        'lr': [],
                        'lickrate':[], 
                        })

    lr = {'ctrl': {}, 'ko':{}}
    x = np.arange(-10,3)
    for cond, mice in zip(('ctrl', 'cre'), (ctrl_mice, ko_mice)):
        for day in range(6):
            lr[cond][day] = {'fam': [], 'nov': []}
    
            for mouse in mice:
                sess = u.load_vr_day(mouse,day, trial_mat_keys = ('licks','nonconsum_licks','licks_sum','speed'),verbose = False)
                bin_edges = sess.trial_matrices['bin_edges']
                
                for ax_ind, leftright in enumerate([-1, 1]):
                    
                    trial_mask = sess.trial_info['LR']== leftright
                    mu = dt*np.nanmean(sess.trial_matrices[key][trial_mask,:], axis=0)

                    # beginning of reward zone
                    fam_rzone_front = np.argwhere((sess.rzone_fam['tfront']<=bin_edges[1:]*(sess.rzone_fam['tfront']>=bin_edges[:-1])))[0][0]
                    nov_rzone_front = np.argwhere((sess.rzone_nov['tfront']<=bin_edges[1:]*(sess.rzone_nov['tfront']>=bin_edges[:-1])))[0][0]
                    
                    if leftright == sess.novel_arm:
                        
                        # append to plot dictionary
                        lr[cond][day]['nov'].append(mu[nov_rzone_front-x[0]:nov_rzone_front+x[-1]])
                        
                        # average anticipatory lick rate
                        _nov = np.nanmean(mu[nov_rzone_front-3:nov_rzone_front+1])
                        
                        # append to dictionary for dataframe
                        df['mouse'].append(mouse)
                        df['cond'].append(cond)
                        df['day'].append(day)
                        df['nov'].append(1)
                        df['lr'].append(leftright)
                        df['lickrate'].append(_nov)

    
                    else:
                        # append to plot dictionary
                        lr[cond][day]['fam'].append(mu[fam_rzone_front-x[0]:fam_rzone_front+x[-1]])
                       
                       # average anticipatory lick rate
                        _fam = np.nanmean(mu[fam_rzone_front-3:fam_rzone_front+1])

                        # append to dictionary for dataframe
                        df['mouse'].append(mouse)
                        df['cond'].append(cond)
                        df['day'].append(day)
                        df['nov'].append(0)
                        df['lr'].append(leftright)
                        df['lickrate'].append(_nov)                     
            
    df = pd.DataFrame(df)
    
    # plots 
    fig,ax = plt.subplots(2,6, figsize = [30, 10],sharey=True)
    for day in range(6):
        for ax_ind, fn in enumerate(['fam', 'nov']):
            
            # cre plots
            arr = np.array(lr['cre'][day][fn])
            mu, sem = np.nanmean(arr,axis=0), sp.stats.sem(arr, axis=0, nan_policy='omit')
            ax[ax_ind,day].fill_between(x, mu- sem, mu+sem,color='red', alpha = .3, label='Cre')

            # control plots
            arr = np.array(lr['ctrl'][day][fn])
            mu, sem = np.nanmean(arr,axis=0), sp.stats.sem(arr, axis=0, nan_policy='omit')
            ax[ax_ind, day].fill_between(x, mu- sem, mu+sem,color='black', alpha = .3, label='Control')
        
            ax[ax_ind, day].fill_betweenx([0,5], 0, 2, zorder=-1, color='purple',alpha=.3)
            ax[ax_ind, day].fill_betweenx([0,5], 0, 2, zorder=-1, color='purple',alpha=.3)

            ax[ax_ind, day].set_xlabel('Position')
            ax[ax_ind, day].set_xlabel('Position')
            ax[ax_ind, day].set_ylabel('Lick Rate')
            ax[ax_ind, day].spines['top'].set_visible(False)
            ax[ax_ind, day].spines['top'].set_visible(False)
            ax[ax_ind, day].spines['right'].set_visible(False)
            ax[ax_ind, day].spines['right'].set_visible(False)
            ax[ax_ind, day].set_ylim([0,5.5])
    ax[0,1].legend(loc = 'upper left')

    fig.savefig(os.path.join(fig_dir, 'famnov_lickrate_alldays.pdf'))


    fig, ax = plt.subplots(1,2, figsize=[15,5], sharey=True)

    lw = 5 # line width
    s = 10 # dot size
    for day in range(6):
        
        mask = (df['cond']=='ctrl') & (df['day']==day) & (df['nov']==0)
        lickrate = df['lickrate'].loc[mask]._values
        ax[0].scatter(5*day -1 + np.linspace(-0.05, 0.05, num=lickrate.shape[0]), lickrate, color='black', s=s)
        ax[0].plot(5*day -1 + np.array([-0.2, .2]), lickrate.mean()*np.ones([2,]), color='black', linewidth=lw,alpha = .3)
        
        mask = (df['cond']=='ctrl') & (df['day']==day) & (df['nov']==1)
        lickrate = df['lickrate'].loc[mask]._values
        ax[1].scatter(5*day -1 + np.linspace(-0.05, 0.05, num=lickrate.shape[0]), lickrate, color='black', s=s, alpha = 1)
        ax[1].plot(5*day -1 + np.array([-0.2, .2]), lickrate.mean()*np.ones([2,]), color='black', linewidth=lw,alpha = .3)
        
        mask = (df['cond']=='cre') & (df['day']==day) & (df['nov']==0)
        lickrate = df['lickrate'].loc[mask]._values
        ax[0].scatter(5*day +1 + np.linspace(-0.05, 0.05, num=lickrate.shape[0]), lickrate, color='red', s=s)
        ax[0].plot(5*day + 1 + np.array([-0.2, .2]), lickrate.mean()*np.ones([2,]), color='red', linewidth=lw,alpha = .3)
        
        mask = (df['cond']=='cre') & (df['day']==day)& (df['nov']==1)
        lickrate = df['lickrate'].loc[mask]._values
        ax[1].scatter(5*day +1 + np.linspace(-0.05, 0.05, num=lickrate.shape[0]), lickrate, color='red', s=s, alpha=1)
        ax[1].plot(5*day + 1 + np.array([-0.2, .2]), lickrate.mean()*np.ones([2,]), color='red', linewidth=lw,alpha = .3)
        

    for a in range(2):

        ax[a].set_xticks(np.arange(0,6*5,5))
        ax[a].set_xticklabels([x for x in range(1,7)])

        ax[a].spines['top'].set_visible(False)
        ax[a].spines['right'].set_visible(False)

        ax[a].set_ylabel('Antic. Lick Rate')
        ax[a].set_xlabel('Day')

    fig.savefig(os.path.join(fig_dir, 'famnov_average_lickrate_summary.pdf'))


    aov = mixed_anova(data=df[df['nov']==0], dv='lickrate', between='ko', within='day', subject='mouse')
    
    posthoc = pairwise_ttests(data=df[df['nov']==0], dv='lickrate', between='ko', within='day', subject='mouse', padjust = 'holm')
    
    return aov, posthoc


