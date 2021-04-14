import dill
import numpy as np
import scipy as sp
from sklearn.linear_model import LinearRegression
import TwoPUtils

from suite2p.extraction import dcnv
from scipy.interpolate import interp1d as spline


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
        self.place_cell_info = {}

        if prev_sess is not None:
            for attr in dir(prev_sess):
                if not attr.startswith('__') and not callable(getattr(prev_sess, attr)):
                    kwargs[attr] = getattr(prev_sess, attr)
                    # setattr(self, attr, getattr(prev_sess, attr))

        super(YMazeSession, self).__init__(**kwargs)

        self._get_pos2t_spline()

    @classmethod
    def from_file(cls, filename):
        return cls(prev_sess=dill.load(filename))

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
            if iti[i - 1] > 60:
                block_number_counter += 1
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
        self.rzone_early['zback'] = self.rzone_early['zfront'] + 25 / 2 ** .5
        self.rzone_early.update({'tfront': self.z2t_spline(self.rzone_early['zfront']),
                                 'tback': self.z2t_spline(self.rzone_early['zback'])})
        self.rzone_early['t_antic'] = self.rzone_early['tfront'] - 5
        self.rzone_early['z_antic'] = self.t2z_spline(self.rzone_early['t_antic'])

        self.rzone_late['zfront'] = self.rzone_late['zcenter'] - self.rzone_late['zcenter'] / self.rzone_late[
            'scale'] / 2
        self.rzone_late['zback'] = self.rzone_late['zfront'] + 25 / 2 ** .5
        self.rzone_late.update(
            {'tfront': self.z2t_spline(self.rzone_late['zfront']), 'tback': self.z2t_spline(self.rzone_late['zback'])})
        self.rzone_late['t_antic'] = self.rzone_late['tfront'] - 5

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

    def neuropil_corrected_dff(self, Fkey='F', Fneukey='Fneu', Fneu_coef=.7, key_out=None, **dff_kwargs):
        """

        :return:
        """
        if key_out is None:
            key_out = Fkey + '_dff'

        Freg = np.zeros(self.timeseries[Fkey].shape) * np.nan
        dff = np.zeros(self.timeseries[Fkey].shape) * np.nan
        spks = np.zeros(self.timeseries[Fkey].shape) * np.nan
        # lr = LinearRegression(fit_intercept=False)
        for block in np.unique(self.trial_info['block_number']).tolist():
            start_ind = self.trial_start_inds[self.trial_info['block_number'] == block][0]
            stop_ind = self.teleport_inds[self.trial_info['block_number'] == block][-1]
            print(start_ind, stop_ind)

            Freg[:, start_ind:stop_ind] = self.timeseries[Fkey][:, start_ind:stop_ind] - Fneu_coef * self.timeseries[
                                                                                                         Fneukey][:,
                                                                                                     start_ind:stop_ind]
            # for cell in range(self.timeseries[Fkey].shape[0]):
            #     # lr.fit(self.timeseries[Fneukey][cell:cell + 1, start_ind:stop_ind].T,
            #     #        self.timeseries[Fkey][cell, start_ind:stop_ind])
            #     Freg[cell, start_ind:stop_ind] = self.timeseries[Fkey][cell, start_ind:stop_ind] - lr.predict(
            #         self.timeseries[Fneukey][cell:cell + 1, start_ind:stop_ind].T)

            Freg[:, start_ind:stop_ind] = sp.ndimage.median_filter(Freg[:, start_ind:stop_ind], size=(1, 7))
            dff[:, start_ind:stop_ind] = TwoPUtils.utilities.dff(Freg[:, start_ind:stop_ind], **dff_kwargs)

            spks[:, start_ind:stop_ind] = dcnv.oasis(dff[:, start_ind:stop_ind], 2000, self.s2p_ops['tau'],
                                                     self.scan_info['frame_rate'])

        self.add_timeseries(**{key_out: dff, 'spks': spks})
        self.add_pos_binned_trial_matrix(key_out)
        self.add_pos_binned_trial_matrix('spks')

    def place_cells_calc(self, Fkey='F_dff', trial_mask=None, lr_split=True, out_key=None, min_pos=13, max_pos=43,
                         bin_size=1, **pc_kwargs):

        # choose appropriate target dictionary
        if out_key is None:
            d = self.place_cell_info
        else:
            self.place_cell_info.update({out_key: {}})
            d = self.place_cell_info[out_key]

        if trial_mask is None:
            trial_mask = np.ones(self.trial_start_inds.shape) > 0

        if lr_split:

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

    def neuropil_corrected_dff(self, Fkey='F', Fneukey='Fneu', key_out=None, **dff_kwargs):
        """

        :return:
        """
        if key_out is None:
            key_out = Fkey + '_dff'

        Freg = np.zeros(self.timeseries[Fkey].shape) * np.nan
        dff = np.zeros(self.timeseries[Fkey].shape) * np.nan

        lr = LinearRegression(fit_intercept=False)

        for cell in range(self.timeseries[Fkey].shape[0]):
            lr.fit(self.timeseries[Fneukey][cell:cell + 1, :].T,
                   self.timeseries[Fkey][cell, :])
            Freg[cell, :] = self.timeseries[Fkey][cell, :] - lr.predict(
                self.timeseries[Fneukey][cell:cell + 1, :].T)

        dff = sp.ndimage.median_filter(Freg, size=(1, 7))
        dff = TwoPUtils.utilities.dff(dff, **dff_kwargs)

        spks = dcnv.oasis(dff, 2000, self.s2p_ops['tau'], self.scan_info['frame_rate'])

        self.add_timeseries(**{key_out: dff, 'spks': spks})
        self.add_pos_binned_trial_matrix(key_out, 'pos')
        self.add_pos_binned_trial_matrix('spks', 'pos')

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
                                                                       min_pos=min_pos, mas_pos=max_pos,
                                                                       bin_size=bin_size, **pc_kwargs)
            d.update({'masks': masks, 'SI': si, 'p': p})
