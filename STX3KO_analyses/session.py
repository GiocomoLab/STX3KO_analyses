import dill
import numpy as np
import scipy as sp
import pandas as pd

from sklearn.linear_model import LinearRegression
import TwoPUtils

from suite2p.extraction import dcnv
from scipy.interpolate import interp1d as spline

def p_from_pop_shuffle(cell_SIs, perm_SIs):
    """
    Calculate place cell p values for spatial information 
    relative to shuffles from the whole population

    :param cell_SIs: spatial information for each cell; array of shape N, or Nx1
    :param perm_SIs: permuted spatial information per shuffle;
                     array of shape n_perms x n_cells (but any shape works)
    :return p_pop: array of p values per cell

    """

    all_SI_perms = np.ravel(perm_SIs)
    p_pop = np.ones((cell_SIs.shape[0],))
    for cell in range(cell_SIs.shape[0]):
        p_pop[cell] = (cell_SIs[cell] <= all_SI_perms).sum()/len(all_SI_perms)

    return p_pop
    
class YMazeSession(TwoPUtils.sess.Session):

    def __init__(self, prev_sess=None, **kwargs):

        """

        :param self:
        :param kwargs:
        :return:
        """
        self.trial_info = None
        self.z2t_spline = None
        self.t2z_spline = None
        self.t2x_spline = None
        self.rzone_early = None
        self.rzone_late = None
        self.novel_arm = None
        self.place_cell_info = {}

        if prev_sess is not None:
            for attr in dir(prev_sess):
                if not attr.startswith('__') and not callable(getattr(prev_sess, attr)):
                    kwargs[attr] = getattr(prev_sess, attr)
                    # setattr(self, attr, getattr(prev_sess, attr))

        super(YMazeSession, self).__init__(**kwargs)

        self._get_pos2t_spline()
        if self.novel_arm is not None:
            if self.novel_arm == -1:
                self.rzone_nov = self.rzone_early
                self.rzone_fam = self.rzone_late
            elif self.novel_arm == 1:
                self.rzone_fam = self.rzone_early
                self.rzone_nov = self.rzone_late

        if isinstance(self.iscell, pd.DataFrame):
            self.mcherry_curated = True
        else:
            self.mcherry_curated = False



    @classmethod
    def from_file(cls, filename, **kwargs):
        '''
        initialize class from previous instance

        :param filename:
        :return:
        '''
        with open(filename, 'rb') as file:
            return cls(prev_sess=dill.load(file), **kwargs)

    def get_trial_info(self):
        """

        :return:
        """
        self.trial_info = {'block_number': self._get_block_number(), 'LR': self._get_LR_trial()}

    def _get_block_number(self):
        """
        
        :param self: 
        :return: 
        """
        tport_times = self.vr_data['time'].iloc[self.teleport_inds]._values
        tstart_times = self.vr_data['time'].iloc[self.trial_start_inds]._values
        iti = tstart_times[1:] - tport_times[:-1]
        block_number_trial = np.zeros(tport_times.shape)
        block_number_counter = 0

        for i in range(1, tport_times.shape[-1]):

            # Ella changed threshold from > 60 and added rounding
            if round(iti[i - 1]) >= 59:
                block_number_counter += 1
                print(block_number_counter)
            block_number_trial[i] = block_number_counter

        return block_number_trial

    def _get_LR_trial(self):
        """

        :return:
        """
        lr_trial = np.zeros(self.teleport_inds.shape)
        for i, (start, stop) in enumerate(zip(self.trial_start_inds.tolist(), self.teleport_inds.tolist())):
            lr_trial[i] = self.vr_data['LR'].iloc[start + 10]
        return lr_trial

    def _get_pos2t_spline(self):
        '''

        :return:
        '''
        control_points = np.array([[0., .5, -70],
                                   [0, .5, -25.],
                                   [0, .5, -5],
                                   [0, .5, 0],
                                   [0, .5, 110],
                                   [24.14, .5, 174.14],
                                   [110.6, .5, 259.6],
                                   [194.14, .5, 344.14]])
        tvec = np.zeros([8, ])
        for ind in range(1, 8):
            tvec[ind] = self._get_t(tvec[ind - 1], control_points[ind - 1], control_points[ind])

        t = np.linspace(6.71, 43.2, num=100)
        trajectory = np.array([self._catmulrom(_t, tvec, control_points) for _t in t])

        self.z2t_spline = spline(trajectory[:, 2], t)
        self.t2x_spline = spline(t, trajectory[:, 0])
        self.t2z_spline = spline(t, trajectory[:, 2])

        self.rzone_early = {'xcenter': 31.6, 'zcenter': 181.6, 'scale': 25}
        self.rzone_late = {'xcenter': 88.2, 'zcenter': 238.2, 'scale': 25}

        self.rzone_early['zfront'] = self.rzone_early['zcenter'] - self.rzone_early['zcenter'] / self.rzone_early[
            'scale'] / 2
        self.rzone_early['zback'] = self.rzone_early['zfront'] + 25
        self.rzone_early.update({'tfront': self.z2t_spline(self.rzone_early['zfront']),
                                 'tback': self.z2t_spline(self.rzone_early['zback'])})
        self.rzone_early['t_antic'] = self.rzone_early['tfront'] - 7
        self.rzone_early['z_antic'] = self.t2z_spline(self.rzone_early['t_antic'])

        self.rzone_late['zfront'] = self.rzone_late['zcenter'] - self.rzone_late['zcenter'] / self.rzone_late[
            'scale'] / 2
        self.rzone_late['zback'] = self.rzone_late['zfront'] + 25
        self.rzone_late.update(
            {'tfront': self.z2t_spline(self.rzone_late['zfront']), 'tback': self.z2t_spline(self.rzone_late['zback'])})
        self.rzone_late['t_antic'] = self.rzone_late['tfront'] - 7
        self.rzone_late['z_antic'] = self.t2z_spline(self.rzone_late['t_antic'])

    @staticmethod
    def _get_t(t, p0, p1, alpha=.5):
        '''

        :param t:
        :param p0:
        :param p1:
        :param alpha:
        :return:
        '''
        a = (p0 - p1) ** 2
        b = a.sum() ** (alpha * .5)
        return b + t

    @staticmethod
    def _catmulrom(_t, tvec, control_points):
        '''

        :param _t:
        :param tvec:
        :param control_points:
        :return:
        '''
        if tvec[1] <= _t < tvec[2]:
            ind = 0
        elif tvec[2] <= _t < tvec[3]:
            ind = 1
        elif tvec[3] <= _t < tvec[4]:
            ind = 2
        elif tvec[4] <= _t < tvec[5]:
            ind = 3
        elif tvec[5] <= _t < tvec[6]:
            ind = 4
        else:
            _t = tvec[2]
            ind = 1
        #     print(ind)
        p0 = control_points[ind, :]
        p1 = control_points[ind + 1, :]
        p2 = control_points[ind + 2, :]
        p3 = control_points[ind + 3, :]

        t0, t1, t2, t3 = tvec[ind], tvec[ind + 1], tvec[ind + 2], tvec[ind + 3]

        a1 = (t1 - _t) / (t1 - t0) * p0 + (_t - t0) / (t1 - t0) * p1
        a2 = (t2 - _t) / (t2 - t1) * p1 + (_t - t1) / (t2 - t1) * p2
        a3 = (t3 - _t) / (t3 - t2) * p2 + (_t - t2) / (t3 - t2) * p3

        b1 = (t2 - _t) / (t2 - t0) * a1 + (_t - t0) / (t2 - t0) * a2
        b2 = (t3 - _t) / (t3 - t1) * a2 + (_t - t1) / (t3 - t1) * a3

        c = (t2 - _t) / (t2 - t1) * b1 + (_t - t1) / (t2 - t1) * b2
        return c

    def add_pos_binned_trial_matrix(self, ts_name, pos_key='t', min_pos=13, max_pos=43, bin_size=1, mat_only=True, 
                                    **trial_matrix_kwargs):
        """

        :param ts_name:
        :param pos_key:
        :param min_pos:
        :param max_pos:
        :param bin_size:
        :param mat_only:
        :param trial_matrix_kwargs:
        :return:
        """
        super(YMazeSession, self).add_pos_binned_trial_matrix(ts_name, pos_key,
                                                              min_pos=min_pos,
                                                              max_pos=max_pos,
                                                              bin_size=bin_size,
                                                              mat_only=mat_only,
                                                              **trial_matrix_kwargs)

        if 'bin_edges' not in self.trial_matrices.keys() or 'bin_centers' not in self.trial_matrices.keys():
            self.trial_matrices['bin_edges'] = np.arange(min_pos, max_pos + bin_size, bin_size)
            self.trial_matrices['bin_centers'] = self.trial_matrices['bin_edges'][:-1] + bin_size / 2

    def add_pos_binned_trial_matrix_mux(self, ts_name, pos_key='t', min_pos=13, max_pos=43, bin_size=1, mat_only=True, 
                                    **trial_matrix_kwargs):
        """

        :param ts_name:
        :param pos_key:
        :param min_pos:
        :param max_pos:
        :param bin_size:
        :param mat_only:
        :param trial_matrix_kwargs:
        :return:
        """
        super(YMazeSession, self).add_pos_binned_trial_matrix_mux(ts_name, pos_key,
                                                              min_pos=min_pos,
                                                              max_pos=max_pos,
                                                              bin_size=bin_size,
                                                              mat_only=mat_only,
                                                              **trial_matrix_kwargs)

        if 'bin_edges' not in self.trial_matrices.keys() or 'bin_centers' not in self.trial_matrices.keys():
            self.trial_matrices['bin_edges'] = np.arange(min_pos, max_pos + bin_size, bin_size)
            self.trial_matrices['bin_centers'] = self.trial_matrices['bin_edges'][:-1] + bin_size / 2

    def neuropil_corrected_dff(self, Fkey='F', Fneukey='Fneu', spks_key=None, Fneu_coef=.7, tau=None, key_out=None, chan_mask = None, **dff_kwargs):
        """

        :return:
        """
        if key_out is None:
            key_out = Fkey + '_dff'    
        
        if tau is None:
            if self.n_channels >1:
                tau = self.s2p_ops['tau']['channel_0']['tau']
            else:
                tau = self.s2p_ops['tau']
                
        if spks_key is None:
            spks_key = 'spks'
            
 
        Freg = np.zeros(self.timeseries[Fkey].shape) * np.nan
        dff = np.zeros(self.timeseries[Fkey].shape) * np.nan
        spks = np.zeros(self.timeseries[Fkey].shape) * np.nan
        # lr = LinearRegression(fit_intercept=True)
        for block in np.unique(self.trial_info['block_number']).tolist():
            start_ind = self.trial_start_inds[self.trial_info['block_number'] == block][0]
            stop_ind = self.teleport_inds[self.trial_info['block_number'] == block][-1]
            print(start_ind, stop_ind)

            curr_idx = range(start_ind, stop_ind)
            
            Freg[:, start_ind:stop_ind] = self.timeseries[Fkey][:, start_ind:stop_ind] - Fneu_coef * self.timeseries[
                                                                                                         Fneukey][:,
                                                                                                     start_ind:stop_ind] + Fneu_coef * np.amin(
                self.timeseries[Fneukey][:, start_ind:stop_ind], axis=1, keepdims=True)

            Freg[:, start_ind:stop_ind] = sp.ndimage.median_filter(Freg[:, start_ind:stop_ind], size=(1, 7))
            dff[:, start_ind:stop_ind] = TwoPUtils.utilities.dff(Freg[:, start_ind:stop_ind], **dff_kwargs)
            

            spks[:, start_ind:stop_ind] = dcnv.oasis(dff[:, start_ind:stop_ind], 2000, tau,
                                                        self.scan_info['frame_rate'])

        
        self.add_timeseries(**{key_out: dff, spks_key: spks})
        self.add_pos_binned_trial_matrix(key_out)
        self.add_pos_binned_trial_matrix(spks_key)
        
    def neuropil_corrected_dff_ES(self, Fkey='F', Fneukey='Fneu', spks_key=None, Fneu_coef=.7, tau=None, key_out=None, chan_mask=None, **dff_kwargs):
        """
        Neuropil-corrected dF/F calculation with an optional channel mask.
        """
        if key_out is None:
            key_out = Fkey + '_dff'
        if tau is None:
            tau = self.s2p_ops['tau']['channel_0']['tau'] if self.n_channels > 1 else self.s2p_ops['tau']
        if spks_key is None:
            spks_key = 'spks'
            
        # Initialize arrays filled with NaN
        Freg = np.zeros(self.timeseries[Fkey].shape) * np.nan
        dff = np.zeros(self.timeseries[Fkey].shape) * np.nan
        spks = np.zeros(self.timeseries[Fkey].shape) * np.nan
        
        for block in np.unique(self.trial_info['block_number']).tolist():
            start_ind = self.trial_start_inds[self.trial_info['block_number'] == block][0]
            stop_ind = self.teleport_inds[self.trial_info['block_number'] == block][-1]
            print(start_ind, stop_ind)
            
            if chan_mask is not None:
                # Ensure mask is the correct length
                if len(chan_mask) != self.timeseries[Fkey].shape[1]:
                    raise ValueError("chan_mask length does not match time series length")
                mask_range = chan_mask[start_ind:stop_ind]  # Mask within this range
            else:
                mask_range = np.ones(stop_ind - start_ind, dtype=bool)  # Default to all True
                
            if not np.any(mask_range):
                continue

                
            curr_idx = np.arange(start_ind, stop_ind)
            
            # Extract masked segment
            if chan_mask is not None:
                curr_idx = curr_idx[chan_mask[curr_idx]]
                print(curr_idx)
            if len(curr_idx) == 0:
                continue
            
            # Apply Neuropil correction only where mask is True
            Freg[:, curr_idx] = self.timeseries[Fkey][:, curr_idx] - Fneu_coef * self.timeseries[Fneukey][:, curr_idx] \
                                + Fneu_coef * np.nanmin(self.timeseries[Fneukey][:, curr_idx], axis=1, keepdims=True)

            Freg[:, curr_idx] = sp.ndimage.median_filter(Freg[:, curr_idx], size=(1, 7))
            dff[:, curr_idx] = TwoPUtils.utilities.dff(Freg[:, curr_idx])
            spks[:, curr_idx] = dcnv.oasis(dff[:, curr_idx], 2000, tau, self.scan_info['frame_rate'])
            
        # Add processed data back to time series while preserving NaNs
        #changed to muxing version 
        self.add_timeseries(**{key_out: dff, spks_key: spks})
        self.add_pos_binned_trial_matrix(key_out)
        self.add_pos_binned_trial_matrix(spks_key)

    
    def neuropil_corrected_dff_mux(self, Fkey_='F', Fneukey_='Fneu', spks_key_=None, Fneu_coef=.7, tau=None, key_out=None, channels=None, **dff_kwargs):
        """
        Neuropil-corrected dF/F calculation for multi-chan muxed data.
        """

        for chan in channels:
            Fkey = chan + '_' + Fkey_
            Fneukey = chan + '_' + Fneukey_
            tau = self.s2p_ops[chan]['tau']
            spks_key = chan + '_' + spks_key_

            key_out = chan + '_' + Fkey_ + '_dff'
            
            # Initialize arrays filled with NaN
            Freg = np.zeros(self.timeseries[Fkey].shape) * np.nan
            dff = np.zeros(self.timeseries[Fkey].shape) * np.nan
            spks = np.zeros(self.timeseries[Fkey].shape) * np.nan

            start_inds = self.trial_starts[chan]
            end_inds = self.trial_ends[chan]
            
            for block in np.unique(self.trial_info['block_number']).tolist():
                start_ind = start_inds[self.trial_info['block_number'] == block][0]
                stop_ind = end_inds[self.trial_info['block_number'] == block][-1]
                print(start_ind, stop_ind)

                curr_idx = np.arange(start_ind, stop_ind)
    
                Freg[:, curr_idx] = self.timeseries[Fkey][:, curr_idx] - Fneu_coef * self.timeseries[Fneukey][:, curr_idx] \
                                    + Fneu_coef * np.nanmin(self.timeseries[Fneukey][:, curr_idx], axis=1, keepdims=True)
    
                Freg[:, curr_idx] = sp.ndimage.median_filter(Freg[:, curr_idx], size=(1, 7))
                dff[:, curr_idx] = TwoPUtils.utilities.dff(Freg[:, curr_idx])
                spks[:, curr_idx] = dcnv.oasis(dff[:, curr_idx], 2000, tau, self.scan_info['frame_rate'])
                
            self.add_timeseries_mux(**{key_out: dff, spks_key: spks})
            self.add_pos_binned_trial_matrix_mux(key_out, channel = chan)
            self.add_pos_binned_trial_matrix_mux(spks_key, channel = chan)


        
    def place_cells_calc(self, Fkey='F_dff', trial_mask=None, lr_split=True, out_key=None, min_pos=13, max_pos=43,
                         bin_size=1, mux = False, **pc_kwargs):

        # choose appropriate target dictionary
        if out_key is None:
            d = self.place_cell_info
        else:
            self.place_cell_info.update({out_key: {}})
            d = self.place_cell_info[out_key]

        if trial_mask is None:
            trial_mask = np.ones(self.trial_start_inds.shape) > 0

        if lr_split:
            if mux :
                if 'channel_0' in Fkey:
                    vr_data = self.vr_data_chan0
                    start_inds = self.trial_starts['channel_0']
                    end_inds = self.trial_ends['channel_0']
                elif 'channel_1' in Fkey:
                    vr_data = self.vr_data_chan1
                    start_inds = self.trial_starts['channel_1']
                    end_inds = self.trial_ends['channel_1']

                lr_masks = {'left': (self.trial_info['LR'] == -1) * trial_mask,
                            'right': (self.trial_info['LR'] == 1) * trial_mask}
                
                for key, mask in lr_masks.items():

                    masks, SI, p = TwoPUtils.spatial_analyses.place_cells_calc(self.timeseries[Fkey].T, vr_data['t'],
                                                                               start_inds[mask],
                                                                               end_inds[mask],
                                                                               min_pos=min_pos, max_pos=max_pos,
                                                                               bin_size=bin_size, **pc_kwargs)
    
                    d[key] = {'masks': masks, 'SI': SI, 'p': p}

            else:
                
                lr_masks = {'left': (self.trial_info['LR'] == -1) * trial_mask,
                            'right': (self.trial_info['LR'] == 1) * trial_mask}
                for key, mask in lr_masks.items():
                    masks, SI, p = TwoPUtils.spatial_analyses.place_cells_calc(self.timeseries[Fkey].T, self.vr_data['t'],
                                                                               self.trial_start_inds[mask],
                                                                               self.teleport_inds[mask],
                                                                               min_pos=min_pos, max_pos=max_pos,
                                                                               bin_size=bin_size, **pc_kwargs)
    
                    d[key] = {'masks': masks, 'SI': SI, 'p': p}
                    
        else:
            mask = trial_mask
            masks, SI, p = TwoPUtils.spatial_analyses.place_cells_calc(self.timeseries[Fkey].T, self.vr_data['t'],
                                                                       self.trial_start_inds[mask],
                                                                       self.teleport_inds[mask],
                                                                       min_pos=min_pos, mas_pos=max_pos,
                                                                       bin_size=bin_size, **pc_kwargs)
            d.update({'masks': masks, 'SI': SI, 'p': p})

            
    def place_cells_calc_pop(self, Fkey='F_dff', trial_mask=None, lr_split=True, out_key=None, min_pos=13, max_pos=43,
                         bin_size=1, p_thr = 0.5, output_shuffle = False, shuffle_method = "individual",**pc_kwargs):

        # choose appropriate target dictionary
        if out_key is None:
            d = self.place_cell_info
        else:
            self.place_cell_info.update({out_key: {}})
            d = self.place_cell_info[out_key]

        if (shuffle_method == 'population') or output_shuffle:
            tmp_output_shuffle = True
        else:
            tmp_output_shuffle = False

        if trial_mask is None:
            trial_mask = np.ones(self.trial_start_inds.shape) > 0

        if lr_split:

            lr_masks = {'left': (self.trial_info['LR'] == -1) * trial_mask,
                        'right': (self.trial_info['LR'] == 1) * trial_mask}
            for key, mask in lr_masks.items():
                masks, SI, p, perms, SI_perms = TwoPUtils.spatial_analyses.place_cells_calc(self.timeseries[Fkey].T, self.vr_data['t'],
                                                                           self.trial_start_inds[mask],
                                                                           self.teleport_inds[mask],
                                                                           min_pos=min_pos, max_pos=max_pos,
                                                                           bin_size=bin_size, output_shuffle=tmp_output_shuffle, **pc_kwargs)

                d[key] = {'masks': masks, 'SI': SI, 'SI_pop': SI_perms, 'p': p}

                if shuffle_method == "population":
                    print(f"updating p with population shuffle for {key}")
                    p_adj = p_from_pop_shuffle(SI, SI_perms)
                    d[key]['p'] = p_adj
                    d[key]['masks'] = p_adj < p_thr
                
        else:
            mask = trial_mask
            masks, SI, p, perms, SI_perms = TwoPUtils.spatial_analyses.place_cells_calc(self.timeseries[Fkey].T, self.vr_data['t'],
                                                                       self.trial_start_inds[mask],
                                                                       self.teleport_inds[mask],
                                                                       min_pos=min_pos, max_pos=max_pos,
                                                                       bin_size=bin_size, output_shuffle=tmp_output_shuffle,**pc_kwargs)
            d.update({'masks': masks, 'SI': SI, 'SI_perms':SI_perms, 'p': p})

            if shuffle_method == "population":
                print("updating p with population shuffle")
                p0 = p_from_pop_shuffle(SI, SI_perms)
                d['p'] = p0 < p_thr
                d['masks'] = p0 <p_thr
    
    
            
    def fam_place_cell_mask(self, mux = False, chan = None, key = 'F_dff'):
        '''

        :return:
        '''
        if mux:
            if self.novel_arm == -1:
                return self.place_cell_info[f"{chan}_{key}"]['right']['masks']
            elif self.novel_arm == 1:
                return self.place_cell_info[f"{chan}_{key}"]['left']['masks']
            else:
                return None
        else:
            if self.novel_arm == -1:
                return self.place_cell_info['right']['masks']
            elif self.novel_arm == 1:
                return self.place_cell_info['left']['masks']
            else:
                return None

    def nov_place_cell_mask(self, mux = False, chan = None, key = 'F_dff'):
        '''

        :return:
        '''
        if mux: 
            if self.novel_arm == 1:
                return self.place_cell_info[f"{chan}_{key}"]['right']['masks']
            elif self.novel_arm == -1:
                return self.place_cell_info[f"{chan}_{key}"]['left']['masks']
            else:
                return None
        else:
            if self.novel_arm == 1:
                return self.place_cell_info['right']['masks']
            elif self.novel_arm == -1:
                return self.place_cell_info['left']['masks']
            else:
                return None

    def mcherry_pos_timeseries(self, fkey):
        '''

        :param fkey:
        :return:
        '''
        return self.timeseries[fkey][self.iscell.loc[:, 'Cre'].to_numpy()>0,:]

    def mcherry_neg_timeseries(self, fkey):
        '''

        :param fkey:
        :return:
        '''
        return self.timeseries[fkey][self.iscell.loc[:, 'NotCre'].to_numpy()>0,:]

    def mcherry_pos_trialmatrix(self, fkey):
        '''
        :param fkey:
        :return:
        '''
        return self.trial_matrices[fkey][:, :, self.iscell.loc[:, 'Cre'].to_numpy()>0]

    def mcherry_neg_trialmatrix(self, fkey):
        '''
        :param fkey:
        :return:
        '''
        return self.trial_matrices[fkey][:, :, self.iscell.loc[:, 'NotCre'].to_numpy()>0]




class ConcatYMazeSession:

    def __init__(self, sess_list, common_roi_mapping, trial_info_keys=['LR', 'block_number'],
                 trial_mat_keys=['F_dff', ],
                 timeseries_keys=(), run_place_cells=True, day_inds=None,
                 load_ops=False, load_stats = False):

        attrs = self.concat(sess_list, common_roi_mapping, trial_info_keys, trial_mat_keys,
                            timeseries_keys, run_place_cells, day_inds, load_ops, load_stats)


        self.__dict__.update(attrs)
        trial_info_keys = []

    @staticmethod
    def concat(_sess_list, common_roi_mapping, t_info_keys, t_mat_keys,
               timeseries_keys, run_place_cells, day_inds, load_ops, load_stats):
        attrs = {}
        attrs['day_inds'] = day_inds
        # same info
        #         same_attrs = ['mouse', 'novel_arm','rzone_early', 'rzone_late']
        attrs.update({'mouse': _sess_list[0].mouse,
                      'novel_arm': _sess_list[0].novel_arm,
                      'rzone_early': _sess_list[0].rzone_early,
                      'rzone_late': _sess_list[0].rzone_late
                      })
        if attrs['novel_arm'] == -1:
            attrs.update({'rzone_nov': attrs['rzone_early'],
                          'rzone_fam': attrs['rzone_late']})
        elif attrs['novel_arm'] == 1:
            attrs.update({'rzone_fam': attrs['rzone_early'],
                          'rzone_nov': attrs['rzone_late']})

        # print(t_info_keys)

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


        cell_info_attrs= ['s2p_stats', 's2p_ops']
        attrs.update({k:[] for k in cell_info_attrs})
        last_block = 0
        cum_frames = 0
        for ind, _sess in enumerate(_sess_list):
            if load_ops:
                attrs['s2p_ops'].append(_sess.s2p_ops)
            if load_stats:
                attrs['s2p_stats'].append(_sess.s2p_stats[common_roi_mapping[ind,:]])
            for k in basic_info_attrs:
                if k in ('teleport_inds', 'trial_start_inds'):
                    attrs[k].append(getattr(_sess, k) + cum_frames)
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
                if len(_sess.timeseries[k].shape) == 2 and _sess.timeseries[k].shape[0] > 1:
                    # if _sess.timeseries[k].shape[0] > 1:
                    timeseries[k].append(_sess.timeseries[k][common_roi_mapping[ind, :], :])
                elif len(_sess.timeseries[k].shape) == 2 and _sess.timeseries[k].shape[0] == 1:
                    timeseries[k].append(_sess.timeseries[k])
                else:
                    timeseries[k].append(_sess.timeseries[k][np.newaxis, :])

            if run_place_cells:
                for lr, _lr in [[-1, 'left'], [1, 'right']]:
                    for k in ['masks', 'SI', 'p']:
                        place_cells[lr][k].append(_sess.place_cell_info[_lr][k][common_roi_mapping[ind, :]])

            cum_frames += _sess.timeseries['licks'].shape[1]
        # print(t_info_keys)
        for k in ['trial_start_inds', 'teleport_inds']:
            attrs[k] = np.concatenate(attrs[k])

        for k in t_info_keys:
            # print(k)
            trial_info[k] = np.concatenate(trial_info[k])
        attrs['trial_info'] = trial_info

        for k in t_mat_keys:
            trial_mat[k] = np.concatenate(trial_mat[k], axis=0)
        attrs['trial_matrices'] = trial_mat

        for k in timeseries_keys:
            timeseries[k] = np.concatenate(timeseries[k], axis=-1)
        attrs['timeseries'] = timeseries

        if run_place_cells:
            for lr in [-1, 1]:
                for k in ['masks', 'SI', 'p']:
                    place_cells[lr][k] = np.array(place_cells[lr][k])
            attrs['place_cell_info'] = place_cells

        return attrs
    
    def fam_place_cell_mask(self):
        '''

        :return:
        '''
        if self.novel_arm == -1:
            if 'right' in self.place_cell_info.keys():
                return self.place_cell_info['right']['masks']
            else:
                return self.place_cell_info[1]['masks'].sum(axis=0) > 0
        else:
            if 'left' in self.place_cell_info.keys():
                return self.place_cell_info['left']['masks']
            else:
                return self.place_cell_info[-1]['masks'].sum(axis=0) > 0

    def nov_place_cell_mask(self):
        '''

        :return:
        '''
        if self.novel_arm == 1:
            if 'right' in self.place_cell_info.keys():
                return self.place_cell_info['right']['masks']
            else:
                return self.place_cell_info[1]['masks'].sum(axis=0) > 0
        else:
            if 'left' in self.place_cell_info.keys():
                return self.place_cell_info['left']['masks']
            else:
                return self.place_cell_info[-1]['masks'].sum(axis=0) > 0


class MorphSession(TwoPUtils.sess.Session):

    def __init__(self, **kwargs):

        """

        :param self:
        :param kwargs:
        :return:
        """
        self.trial_info = None
        self.place_cell_info = {}

        super(MorphSession, self).__init__(**kwargs)

    def get_trial_info(self):
        """

        :return:
        """

        morph_shared, wall_jitter = self._get_morph()
        self.trial_info = {'morph_shared': morph_shared, 'morph': morph_shared + wall_jitter}

    def _get_morph(self):
        """

        :return:
        """
        morph_trial, wall_jitter = np.zeros(self.teleport_inds.shape), np.zeros(self.teleport_inds.shape)
        for i, (start, stop) in enumerate(zip(self.trial_start_inds.tolist(), self.teleport_inds.tolist())):
            morph_trial[i] = self.vr_data['morph'].iloc[start + 10]
            wall_jitter[i] = self.vr_data['wallJitter'].iloc[start + 10]
        return morph_trial, wall_jitter

    def neuropil_corrected_dff(self, Fkey='F', Fneukey='Fneu', Fneu_coef=0.7, key_out=None, **dff_kwargs):
        """

        :return:
        """
        if key_out is None:
            key_out = Fkey + '_dff'

        # Freg = np.zeros(self.timeseries[Fkey].shape) * np.nan
        dff = np.zeros(self.timeseries[Fkey].shape) * np.nan

        lr = LinearRegression(fit_intercept=False)

        # for cell in range(self.timeseries[Fkey].shape[0]):
        #     lr.fit(self.timeseries[Fneukey][cell:cell + 1, :].T,
        #            self.timeseries[Fkey][cell, :])
        #     Freg[cell, :] = self.timeseries[Fkey][cell, :] - lr.predict(
        #         self.timeseries[Fneukey][cell:cell + 1, :].T)

        Freg = self.timeseries[Fkey] - Fneu_coef*self.timeseries[Fneukey]
        dff = sp.ndimage.median_filter(Freg, size=(1, 7))
        dff = TwoPUtils.utilities.dff(dff, **dff_kwargs)

        spks = dcnv.oasis(dff, 2000, self.s2p_ops['tau'], self.scan_info['frame_rate'])

        self.add_timeseries(**{key_out: dff, 'spks': spks})
        self.add_pos_binned_trial_matrix(key_out, 'pos')
        self.add_pos_binned_trial_matrix('spks', 'pos')

    def add_pos_binned_trial_matrix(self, ts_name, pos_key='pos', min_pos=0, max_pos=450, bin_size=10, mat_only=True,
                                    **trial_matrix_kwargs):
        """

        :param ts_name:
        :param pos_key:
        :param min_pos:
        :param max_pos:
        :param bin_size:
        :param mat_only:
        :param trial_matrix_kwargs:
        :return:
        """
        super(MorphSession, self).add_pos_binned_trial_matrix(ts_name, pos_key,
                                                              min_pos=min_pos,
                                                              max_pos=max_pos,
                                                              bin_size=bin_size,
                                                              mat_only=mat_only,
                                                              **trial_matrix_kwargs)

        if 'bin_edges' not in self.trial_matrices.keys() or 'bin_centers' not in self.trial_matrices.keys():
            self.trial_matrices['bin_edges'] = np.arange(min_pos, max_pos + bin_size, bin_size)
            self.trial_matrices['bin_centers'] = self.trial_matrices['bin_edges'][:-1] + bin_size / 2

    def place_cells_calc(self, Fkey='spks', morph_split=True, out_key=None, bin_size=10, **pc_kwargs):

        # choose appropriate target dictionary
        if out_key is None:
            d = self.place_cell_info
        else:
            self.place_cell_info.update({out_key: {}})
            d = self.place_cell_info[out_key]

        if morph_split:

            morph_masks = {m: (self.trial_info['morph_shared'] == m) for m in [0, .25, .5, .75, 1]}

            for key, mask in morph_masks.items():
                masks, si, p = TwoPUtils.spatial_analyses.place_cells_calc(self.timeseries[Fkey].T, self.vr_data['pos'],
                                                                           self.trial_start_inds[mask],
                                                                           self.teleport_inds[mask],
                                                                           bin_size=bin_size, **pc_kwargs)

                d[key] = {'masks': masks, 'SI': si, 'p': p}
        else:

            masks, si, p = TwoPUtils.spatial_analyses.place_cells_calc(self.timeseries[Fkey].T, self.vr_data['pos'],
                                                                       self.trial_start_inds,
                                                                       self.teleport_inds,
                                                                       **pc_kwargs)
            d.update({'masks': masks, 'SI': si, 'p': p})
