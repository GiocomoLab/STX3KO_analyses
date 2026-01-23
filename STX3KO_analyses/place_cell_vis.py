import os
import math
from itertools import permutations

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from matplotlib import gridspec
import pandas as pd

import TwoPUtils as tpu
import STX3KO_analyses as stx
from STX3KO_analyses import utilities as u


def plot_single_trial_place_cells_dense(mouse, day, fam=True):

    sess = u.load_single_day(mouse,day)
    
    if fam:
        trial_mask = sess.trial_info['LR']!=sess.novel_arm
        pc_mask = sess.fam_place_cell_mask()
        # pc_mask = sess.field_perm_masks['cell_masks']['fam']
    else:
        trial_mask = sess.trial_info['LR']==sess.novel_arm
        pc_mask = sess.nov_place_cell_mask()
        # pc_mask = sess.field_perm_masks['cell_masks']['nov']

    avg_trial_mat = np.nanmean(sess.trial_matrices['F_dff'][trial_mask,:,:], axis=0)[:, pc_mask]
    sort = np.argsort(np.argmax(avg_trial_mat,axis=0))[::-1]
    # make gridspec
    fig = plt.figure(figsize=[12,5])
    gs = gridspec.GridSpec(3,12)

    reward = sess.timeseries['reward'][0,:]
    licks = sess.timeseries['licks'][0,:]
    licks[licks>0]=1
    licks[licks<1]=np.nan
    
    
    # nan ITI for behavior
    t =  sess.timeseries['t'][0,:]
    for (stop, start) in zip(sess.teleport_inds[:-1], sess.trial_start_inds[1:]):
        t[stop:start] = np.nan
    
    t_reward = 10*(t-sess.trial_matrices['bin_centers'][0])*reward
    t_reward[t_reward<1]=np.nan
    
    t_licks = 10*(t-sess.trial_matrices['bin_centers'][0])*licks
    t_licks[t_licks<1]=np.nan
    
    
    t_starts = sess.trial_start_inds[trial_mask]
    t_stops = sess.teleport_inds[trial_mask]
    
    start = t_starts[1]-10
    stop = t_stops[8]+10

    window = slice(start,stop)
    spks = sess.timeseries['F_dff'][pc_mask,:]
    ax_spks = fig.add_subplot(gs[0:2,:11])
    img = ax_spks.imshow(spks[sort, window], aspect='auto',cmap='Greys', vmin=0, vmax=.4)
    
    ax_clrbr = fig.add_subplot(gs[0,11])
    ax_clrbr.set_yticks([])
    ax_clrbr.set_xticks([])
    plt.colorbar(img,ax=ax_clrbr)

    
    x = np.arange(stop-start)
    
    # set iti to nan for position
    t = t[window] #sess.timeseries['t'][0,window]
    t[t<sess.trial_matrices['bin_edges'][0]] = np.nan
    t = 10*(t- sess.trial_matrices['bin_centers'][0])

    ax_pos = fig.add_subplot(gs[2,:11], sharex=ax_spks)
    ax_pos.plot(x, t)
    
    
    ax_pos.scatter(x,t_reward[window], color='orange', zorder=10,s=80, marker='o')
    ax_pos.scatter(x, licks[window]+10, color='red', zorder = 8, alpha=1, s=100, marker='|')
    
    
    for (stop, start) in zip(sess.teleport_inds[1:9], sess.trial_start_inds[2:9]):
        stop = stop - sess.trial_start_inds[1] +9
        start = start - sess.trial_start_inds[1] +9
        ax_pos.fill_betweenx([-10,300], stop, start, color='black', alpha=.3, edgecolor='white')
        
       
    ax_spks.set_ylabel('Cells')
    ax_pos.set_ylabel('Position')
    ax_pos.set_xlabel('Time (s)')

    dt = 1./sess.scan_info['frame_rate']
    labels = (dt*x)[::250]

    ax_pos.set_xticks(x[::250], labels=[str(int(l)) for l in labels])

    ax_pos.spines['top'].set_visible(False)
    ax_pos.spines['right'].set_visible(False)
    


    fig.suptitle(f"mouse: {mouse}, day: {day}")
    
    return fig, (ax_pos, ax_spks)
    

def plot_cell_across_days(sess, cell, plot_roi=True):
    
    fig = plt.figure(figsize = [25, 5])
    gs = gridspec.GridSpec(6, 7, figure=fig)
    mean = np.nanmean(sess.trial_matrices['F_dff'][:,:,cell].ravel())
    for day in range(6):
        fam_mask = (sess.trial_info['sess_num']==day) & (sess.trial_info['LR']==-1*sess.novel_arm)
        nov_mask = (sess.trial_info['sess_num']==day) & (sess.trial_info['LR']==sess.novel_arm)
        fam_trialmat = sess.trial_matrices['F_dff'][fam_mask,:,:][:,:,cell]
        nov_trialmat = sess.trial_matrices['F_dff'][nov_mask,:,:][:,:,cell]
        
    
        fam_trialmat[np.isnan(fam_trialmat)]=1E-3
        nov_trialmat[np.isnan(nov_trialmat)]=1E-3
        
        day_inds = sess.trial_info['sess_num'][sess.trial_info['LR']==-1*sess.novel_arm]==day
        fam_ax = fig.add_subplot(gs[day,0])
        h = fam_ax.imshow(fam_trialmat/mean, cmap = 'magma', aspect='auto', vmin = 0, vmax=5)
        
        nov_ax = fig.add_subplot(gs[day,1])
        nov_ax.imshow(nov_trialmat/mean, cmap = 'magma', aspect='auto', vmin = 0, vmax=5)
        
        if day<5:
            fam_ax.set_xticks([])
            nov_ax.set_xticks([])
        else:
            fam_ax.set_xticks((-.5, 9.5, 19.5, 29.5))
            fam_ax.set_xticklabels((0,100,200,300))
            
            nov_ax.set_xticks((-.5, 9.5, 19.5, 29.5))
            nov_ax.set_xticklabels((0,100,200,300))
    
    
    cbar_ax = fig.add_subplot(gs[2:4,2])
    plt.colorbar(h,ax=cbar_ax)
    
    if plot_roi:
        day0_com = (sess.s2p_stats[0][cell]['xpix'].mean(), sess.s2p_stats[0][cell]['ypix'].mean())
        day5_com = (sess.s2p_stats[len(sess.s2p_stats)-1][cell]['xpix'].mean(), sess.s2p_stats[len(sess.s2p_stats)-1][cell]['ypix'].mean())
        
        sz = 30
        day0img = np.zeros([int(sz*2), int(sz*2),3])
        

        
        x_edges = (int(day0_com[1]-sz), int(day0_com[1]+sz))
        y_edges = ( int(day0_com[0]-sz), int(day0_com[0]+sz))
        g_img = sess.s2p_ops[0]['meanImg'][x_edges[0]:x_edges[1], y_edges[0]:y_edges[1]]
        r_img = sess.s2p_ops[0]['meanImg_chan2'][x_edges[0]:x_edges[1], y_edges[0]:y_edges[1]]
        
        g_img = np.minimum(g_img/np.percentile(g_img,100),1)
        r_img = np.minimum(r_img/np.percentile(r_img, 100),1)
        
        day0_ax_g = fig.add_subplot(gs[0:3,3])
        day0_ax_g.imshow(g_img,cmap='Greys_r')
        day0_ax_g.set_xticks([])
        day0_ax_g.set_yticks([])
        
        day0_ax_r = fig.add_subplot(gs[3:, 3])
        day0_ax_r.imshow(r_img, cmap='Greys_r')
        day0_ax_r.set_xticks([])
        day0_ax_r.set_yticks([])
        
        day0img[:,:,1] = g_img
        day0img[:,:,0] = r_img
        day0img[:,:,2] = .8*r_img + .2*g_img
        
        day0_ax = fig.add_subplot(gs[2:5,4])
        day0_ax.imshow(day0img,cmap='Greys_r')
        day0_ax.set_xticks([])
        day0_ax.set_yticks([])
        
        circle = plt.Circle((sz, sz), 7, fill=False, color='blue',linewidth=3)
        day0_ax.add_patch(circle)
        
        circle = plt.Circle((sz, sz), 7, fill=False, color='blue',linewidth=3)
        day0_ax_g.add_patch(circle)
        circle = plt.Circle((sz, sz), 7, fill=False, color='blue',linewidth=3)
        day0_ax_r.add_patch(circle)
        
        
        day5img = np.zeros([int(sz*2), int(sz*2), 3])

        x_edges = (int(day5_com[1]-sz), int(day5_com[1]+sz))
        y_edges = ( int(day5_com[0]-sz), int(day5_com[0]+sz))
        g_img = sess.s2p_ops[5]['meanImg'][x_edges[0]:x_edges[1], y_edges[0]:y_edges[1]]
        r_img = sess.s2p_ops[5]['meanImg_chan2'][x_edges[0]:x_edges[1], y_edges[0]:y_edges[1]]
        
        g_img = np.minimum(g_img/np.percentile(g_img,100),1)
        r_img = np.minimum(r_img/np.percentile(r_img, 100),1)
        
        day5_ax_g = fig.add_subplot(gs[0:3,5])
        day5_ax_g.imshow(g_img,cmap='Greys_r')
        day5_ax_g.set_xticks([])
        day5_ax_g.set_yticks([])
        
        day5_ax_r = fig.add_subplot(gs[3:, 5])
        day5_ax_r.imshow(r_img, cmap='Greys_r')
        day5_ax_r.set_xticks([])
        day5_ax_r.set_yticks([])
        
        day5img[:,:,1] = 1*g_img
        day5img[:,:,0] = r_img #.5*(r_img+g_img)
        day5img[:,:,2] = .8*r_img+.2*g_img
        
        day5_ax = fig.add_subplot(gs[2:5,6])
        day5_ax.imshow(day5img,cmap='Greys_r')
        day5_ax.set_xticks([])
        day5_ax.set_yticks([])
        
        circle = plt.Circle((sz, sz), 7, fill=False, color='blue',linewidth=3)
        day5_ax.add_patch(circle)
        
        circle = plt.Circle((sz, sz), 7, fill=False, color='blue',linewidth=3)
        day5_ax_g.add_patch(circle)
        circle = plt.Circle((sz, sz), 7, fill=False, color='blue',linewidth=3)
        day5_ax_r.add_patch(circle)
    
        return fig, (fam_ax, nov_ax, cbar_ax, day0_ax, day5_ax)
    else:
        return fig, (fam_ax, nov_ax, cbar_ax)


def plot_cell_across_days_sparse(sess, cell, channel, days, plot_roi=True):
    
    fig = plt.figure(figsize = [25, 5])
    gs = gridspec.GridSpec(6, 7, figure=fig)
    mean = np.nanmean(sess.trial_matrices[f'{channel}_F_dff'][:,:,cell].ravel())
    for day in days:
        fam_mask = (sess.trial_info['sess_num_ravel']==day) & (sess.trial_info['LR']==-1*sess.novel_arm)
        nov_mask = (sess.trial_info['sess_num_ravel']==day) & (sess.trial_info['LR']==sess.novel_arm)
        fam_trialmat = sess.trial_matrices[f'{channel}_F_dff'][fam_mask,:,:][:,:,cell]
        nov_trialmat = sess.trial_matrices[f'{channel}_F_dff'][nov_mask,:,:][:,:,cell]
        
    
        fam_trialmat[np.isnan(fam_trialmat)]=1E-3
        nov_trialmat[np.isnan(nov_trialmat)]=1E-3
        
        day_inds = sess.trial_info['sess_num_ravel'][sess.trial_info['LR']==-1*sess.novel_arm]==day
        fam_ax = fig.add_subplot(gs[day,0])
        h = fam_ax.imshow(fam_trialmat/mean, cmap = 'magma', aspect='auto', vmin = 0, vmax=5)
        
        nov_ax = fig.add_subplot(gs[day,1])
        nov_ax.imshow(nov_trialmat/mean, cmap = 'magma', aspect='auto', vmin = 0, vmax=5)
        
        if day < days[-1]:
            fam_ax.set_xticks([])
            nov_ax.set_xticks([])
        else:
            fam_ax.set_xticks((-.5, 9.5, 19.5, 29.5))
            fam_ax.set_xticklabels((0,100,200,300))
            
            nov_ax.set_xticks((-.5, 9.5, 19.5, 29.5))
            nov_ax.set_xticklabels((0,100,200,300))
    
    
    cbar_ax = fig.add_subplot(gs[2:4,2])
    plt.colorbar(h,ax=cbar_ax)
    
    if plot_roi:
        day0_com = (sess.s2p_stats[channel][0][cell]['xpix'].mean(), 
                    sess.s2p_stats[channel][0][cell]['ypix'].mean())
        dayN_com = (sess.s2p_stats[channel][len(sess.s2p_stats[channel])-1][cell]['xpix'].mean(), 
                    sess.s2p_stats[channel][len(sess.s2p_stats[channel])-1][cell]['ypix'].mean())
        
        sz = 30
        
        

        
        x_edges = (int(day0_com[1]-sz), int(day0_com[1]+sz))
        y_edges = ( int(day0_com[0]-sz), int(day0_com[0]+sz))
        img = sess.s2p_ops[channel][0]['meanImg'][x_edges[0]:x_edges[1], y_edges[0]:y_edges[1]]
       
        img = np.minimum(img/np.percentile(img,100),1)
        
        day0_ax = fig.add_subplot(gs[1:5,3])
        day0_ax.imshow(img,cmap='Greys_r')
        day0_ax.set_xticks([])
        day0_ax.set_yticks([])
        
       
        circle = plt.Circle((sz, sz), 7, fill=False, color='blue',linewidth=3)
        day0_ax.add_patch(circle)
        
       

        x_edges = (int(dayN_com[1]-sz), int(dayN_com[1]+sz))
        y_edges = ( int(dayN_com[0]-sz), int(dayN_com[0]+sz))
        img = sess.s2p_ops[channel][-1]['meanImg'][x_edges[0]:x_edges[1], y_edges[0]:y_edges[1]]
        
        img = np.minimum(img/np.percentile(img,100),1)
        
        dayN_ax = fig.add_subplot(gs[1:5,5])
        dayN_ax.imshow(img,cmap='Greys_r')
        dayN_ax.set_xticks([])
        dayN_ax.set_yticks([])
        
       
        
        circle = plt.Circle((sz, sz), 7, fill=False, color='blue',linewidth=3)
        dayN_ax.add_patch(circle)

        fig.suptitle(f'Cell {cell}, channel {channel}')

        return fig, (fam_ax, nov_ax, cbar_ax, day0_ax, dayN_ax)
    else:
        return fig, (fam_ax, nov_ax, cbar_ax)


def plot_crossval_placecells_across_days(mice, max_sess = 6, key = 'F_dff'):
    '''
    
    '''
    
    fam_dict, nov_dict = {d: {d: [] for d in range(max_sess)} for d in range(max_sess)}, {d: {d: [] for d in range(max_sess)} for d in range(max_sess)}
    fam_sort_dict, nov_sort_dict = {d: [] for d in range(max_sess)}, {d: [] for d in range(max_sess)}
    fam_mu_dict, nov_mu_dict = {d: [] for d in range(max_sess)}, {d: [] for d in range(max_sess)}
    fam_std_dict, nov_std_dict = {d: [] for d in range(max_sess)}, {d: [] for d in range(max_sess)}
    
    for mouse in mice:

        # load data
        concat_sess = u.single_mouse_concat_sessions(mouse,date_inds=np.arange(0,max_sess),
                                                     trial_mat_keys=[key,], timeseries_keys=[key,], load_stats=False)
        
        # for each day that is being used for sorting
        for sort_day in range(max_sess):

            # place_cell_mask = (np.array(concat_sess.field_perm_masks['cell_masks']['fam']).sum(axis=0)==6) + \
            #                   (np.array(concat_sess.field_perm_masks['cell_masks']['nov']).sum(axis=0)==6)
            place_cell_mask = concat_sess.fam_place_cell_mask() + concat_sess.nov_place_cell_mask()
            
            trial_mask = (concat_sess.trial_info['sess_num']==sort_day)*(concat_sess.trial_info['LR']==-1*concat_sess.novel_arm)
            trial_inds = np.arange(trial_mask.shape[0])
            trial_inds = trial_inds[trial_mask]
            fr = np.nanmean(concat_sess.trial_matrices[key][trial_inds[::2],:,:],axis=0)[:,place_cell_mask]+1E-5
            fam_mu, fam_std = fr.mean(axis=0,keepdims=True), fr.std(axis=0,keepdims=True)
            fam_mu_dict[sort_day].append(fam_mu)
            fam_std_dict[sort_day].append(fam_std)
            
            trial_mask = (concat_sess.trial_info['sess_num']==sort_day)*(concat_sess.trial_info['LR']==concat_sess.novel_arm)
            trial_inds = np.arange(trial_mask.shape[0])
            trial_inds = trial_inds[trial_mask]
            fr = np.nanmean(concat_sess.trial_matrices[key][trial_inds[::2],:,:],axis=0)[:,place_cell_mask]+1E-5
            nov_mu, nov_std = fr.mean(axis=0,keepdims=True), fr.std(axis=0,keepdims=True)
            nov_mu_dict[sort_day].append(nov_mu)
            nov_std_dict[sort_day].append(nov_std)
            
            for day in range(max_sess):
                trial_mask = (concat_sess.trial_info['sess_num']==day)*(concat_sess.trial_info['LR']==-1*concat_sess.novel_arm)
                trial_inds = np.arange(trial_mask.shape[0])
                trial_inds = trial_inds[trial_mask]
                if sort_day==day:
                    fam_sort_dict[sort_day].append(sp.stats.zscore(np.nanmean(concat_sess.trial_matrices[key][trial_inds[::2],:,:]+1E-5,axis=0)[:,place_cell_mask],axis=1))
                    trial_inds = trial_inds[1::2]
#                 fr = sp.stats.zscore(np.nanmean(concat_sess.trial_matrices['F_dff'][trial_inds,:,:],axis=0)[:,place_cell_mask],axis=1)
                fr = np.nanmean(concat_sess.trial_matrices[key][trial_inds,:,:],axis=0)[:,place_cell_mask]
                fam_dict[sort_day][day].append(fr)
                
                trial_mask = (concat_sess.trial_info['sess_num']==day)*(concat_sess.trial_info['LR']==concat_sess.novel_arm)
                trial_inds = np.arange(trial_mask.shape[0])
                trial_inds = trial_inds[trial_mask]
                if sort_day==day:
                    nov_sort_dict[sort_day].append(sp.stats.zscore(np.nanmean(concat_sess.trial_matrices[key][trial_inds[::2],:,:]+1E-5,axis=0)[:,place_cell_mask],axis=1))
                    trial_inds = trial_inds[1::2]
                fr = np.nanmean(concat_sess.trial_matrices[key][trial_inds,:,:],axis=0)[:,place_cell_mask]
                nov_dict[sort_day][day].append(fr)

    for sort_day in range(max_sess):
        fam_sort_dict[sort_day] = np.concatenate(fam_sort_dict[sort_day],axis = -1)
        nov_sort_dict[sort_day] = np.concatenate(nov_sort_dict[sort_day],axis = -1)
        
        fam_mu_dict[sort_day] = np.concatenate(fam_mu_dict[sort_day],axis = -1)
        nov_mu_dict[sort_day] = np.concatenate(nov_mu_dict[sort_day],axis = -1)
        
        fam_std_dict[sort_day] = np.concatenate(fam_std_dict[sort_day],axis = -1)
        nov_std_dict[sort_day] = np.concatenate(nov_std_dict[sort_day],axis = -1)
        
        for day in range(max_sess):
            fam_dict[sort_day][day] = np.concatenate(fam_dict[sort_day][day],axis = -1)
            nov_dict[sort_day][day] = np.concatenate(nov_dict[sort_day][day],axis = -1)

    fig_fam_fam, ax_fam_fam  = plt.subplots(max_sess,max_sess, figsize= [30,30])
    fig_fam_fam.suptitle('Familiar act, Familiar sort')
    
    fig_fam_nov, ax_fam_nov  = plt.subplots(max_sess,max_sess, figsize= [30,30])
    fig_fam_nov.suptitle('Familiar act, Novel sort')
    
    fig_nov_fam, ax_nov_fam  = plt.subplots(max_sess,max_sess, figsize= [30,30])
    fig_nov_fam.suptitle('Novel act, Familiar sort')
    
    fig_nov_nov, ax_nov_nov  = plt.subplots(max_sess,max_sess, figsize= [30,30])
    fig_nov_nov.suptitle('Novel act, Novel sort')
    for sort_day in range(max_sess):
        sort_fam  = np.argsort(np.argmax(fam_sort_dict[sort_day],axis=0))
        sort_nov  = np.argsort(np.argmax(nov_sort_dict[sort_day],axis=0))
        for day in range(max_sess):
            fam_fam = sp.stats.zscore(fam_dict[sort_day][day], axis=0) #(fam_dict[sort_day][day]-fam_mu_dict[sort_day])/fam_std_dict[sort_day]
            fam_nov = sp.stats.zscore(fam_dict[sort_day][day], axis=0) #(fam_dict[sort_day][day] - nov_mu_dict[sort_day])/nov_std_dict[sort_day]
            nov_fam = sp.stats.zscore(nov_dict[sort_day][day], axis=0) #(nov_dict[sort_day][day] - fam_mu_dict[sort_day])/fam_std_dict[sort_day]
            nov_nov = sp.stats.zscore(nov_dict[sort_day][day], axis=0) #(nov_dict[sort_day][day] - nov_mu_dict[sort_day])/nov_std_dict[sort_day]
            

            h = ax_fam_fam[sort_day, day].imshow(fam_fam[:,sort_fam].T, aspect = 'auto', vmax = 4, vmin  =0., cmap = 'pink')
            ax_fam_fam[sort_day,day].set_title(fam_fam.shape)
            plt.colorbar(h,ax=ax_fam_fam[sort_day,day])
            
            h = ax_fam_nov[sort_day, day].imshow(fam_nov[:,sort_nov].T, aspect = 'auto', vmax = 4, vmin  =0., cmap = 'pink')
            ax_fam_nov[sort_day,day].set_title(fam_nov.shape)
            plt.colorbar(h,ax=ax_fam_nov[sort_day,day])
            
            h = ax_nov_fam[sort_day, day].imshow(nov_fam[:,sort_fam].T, aspect = 'auto', vmax = 4, vmin  =0., cmap = 'pink')
            ax_nov_fam[sort_day,day].set_title(nov_fam.shape)
            plt.colorbar(h,ax=ax_nov_fam[sort_day,day])
            
            h = ax_nov_nov[sort_day, day].imshow(nov_nov[:,sort_nov].T, aspect = 'auto', vmax = 4, vmin  =0., cmap = 'pink')
            ax_nov_nov[sort_day,day].set_title(nov_nov.shape)
            plt.colorbar(h,ax=ax_nov_nov[sort_day,day])
            
            
    return (fig_fam_fam, ax_fam_fam), (fig_fam_nov, ax_fam_nov), (fig_nov_fam, ax_nov_fam), (fig_nov_nov, ax_nov_nov)