import numpy as np
from sklearn.linear_model import LinearRegression
import TwoPUtils




class YMazeSession(TwoPUtils.sess.Session):

    def __init__(self, **kwargs):

        """

        :param self:
        :param kwargs:
        :return:
        """
        self.trial_info = None
        super(YMazeSession,self).__init__(**kwargs)


    def get_trial_info(self):
        """

        :return:
        """
        self.trial_info = {'block_number':self._get_block_number, 'LR':self._get_LR_trial}
        
    
    
    def _get_block_number(self):
        """
        
        :param self: 
        :return: 
        """
        tport_times = self.vr_data['time'].iloc[self.teleport_inds]._values
        tstart_times = self.vr_data['time'].iloc[self.trial_start_inds]._values
        ITI = tstart_times[1:] - tport_times[:-1]
        block_number_trial = np.zeros(tport_times.shape)
        block_number_counter = 0


        for i in range(1, tport_times.shape[-1]):
            if ITI[i - 1] > 60:
                block_number_counter += 1
            block_number_trial[i] = block_number_counter

        return block_number_trial

    def _get_LR_trial(self):
        """

        :return:
        """
        lr_trial = np.zeros(self.teleport_inds.shape)
        for i, (start, stop) in enumerate(zip(self.trial_start_inds.tolist(), self.teleport_inds.tolist())):
            lr_trial[i] = self.vr_data['LR'].iloc[start + 1]
        return lr_trial

    def add_pos_binned_trial_matrix(self,ts_name, pos_key='t', min_pos=6,max_pos=43,bin_size=1,mat_only=True,  **trial_matrix_kwargs):
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
        super(YMazeSession,self).add_pos_binned_trial_matrix(ts_name,pos_key,
                                                             min_pos=min_pos,
                                                             max_pos=max_pos,
                                                             bin_size=bin_size,
                                                             mat_only=mat_only,
                                                             **trial_matrix_kwargs)

    def neuropil_corrected_dff(self, Fkey, Fneukey, key_out=None, **dff_kwargs):
        """

        :return:
        """
        if key_out is None:
            key_out = Fkey+'_dff'

        Freg = np.zeros(self.timeseries[Fkey].shape) * np.nan
        dff = np.zeros(self.timeseries[Fkey].shape) * np.nan
        lr = LinearRegression()
        for block in np.unique(self.trial_info['block_number']).tolist():

            start_ind, stop_ind = self.trial_start_inds[self.trial_info['block_number'] == block][0], \
                                  self.teleport_inds[self.trial_info['block_number'] == block][-1]
            print(start_ind, stop_ind)

            for cell in range(self.timeseries[Fkey].shape[0]):
                lr.fit(self.timeseries[Fneukey][cell:cell + 1, start_ind:stop_ind].T,
                       self.timeseries[Fkey][cell, start_ind:stop_ind])
                Freg[cell, start_ind:stop_ind] = self.timeseries[Fkey][cell, start_ind:stop_ind] - lr.predict(
                    self.timeseries[Fneukey][cell:cell + 1, start_ind:stop_ind].T)
            dff[:, start_ind:stop_ind] = TwoPUtils.utilities.dff(Freg[:, start_ind:stop_ind].T).T

        self.add_timeseries(**{key_out: dff})
        self.add_pos_binned_trial_matrix(key_out)


