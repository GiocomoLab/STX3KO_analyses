import numpy as np
from scipy.interpolate import interp1d as spline
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec as gridspec

import TwoPUtils as tpu


class KWTA():

    def __init__(self, n_pos=30, w_max=100, n_ca3=1000, n_ca1=1000, n_winners=100,
                 eta=1E-4, tau=1E-5, eta_ctrl=None, max_pos=10, ca3_sigma_mag=.08,
                 weight_dist='lognormal', w_sigma_mag=1E-3, w_norm_decay=1E-3, ca1_noise=1):
        '''
        K-Winners-Take-All model with Hebbian learning for inheriting frozen CA3 linear track place cells
        representations to CA1

        example usage: kwta = KWTA() # initialize
                    ca1 = kwta.forward() # get CA1 activity and update weights

        :param n_pos: number of position bins
        :param w_max: maximum synaptic weight (min is 0)
        :param n_ca3: number of CA3 neurons
        :param n_ca1: number of CA1 neurons
        :param n_winners: number of 'winners' (k) on each pass
        :param eta: learning rate
        :param tau: uniform decay rate
        :param eta_ctrl: control points for position modulated gain in learning rate
        :param max_pos: maximum track positions
        :param ca3_sigma_mag: gain for ca3 activation noise
        :param weight_dist: initial weight distribution (either 'lognormal' or 'uniform')
        :param w_sigma_mag: gain for noise on weight updates
        :param w_norm_decay: if non-zero, decay weights by this constant*l2 norm of the weight vector onto each CA1 cell
        :param ca1_noise: constant for uniform noise added to activity of CA1 neurons
        '''

        # initialize properties
        self.n_pos = n_pos
        self.pos = np.linspace(0, 10, num=n_pos)[np.newaxis, :]  # position on track
        self.max_pos = max_pos
        self.w_max = w_max
        self.n_ca3 = n_ca3
        self.n_ca1 = n_ca1
        self.n_winners = n_winners
        self.eta = eta
        self.tau = tau
        self.ca3_sigma_mag = ca3_sigma_mag
        self.w_sigma_mag = w_sigma_mag
        self.w_norm_decay = w_norm_decay
        self.ca1_noise = ca1_noise
        self.rng_ = np.random.default_rng()

        # make ca3 place fields
        self.mu = np.linspace(0, self.max_pos, num=n_ca3)[:, np.newaxis]  # centers of place fields
        self.ca3 = tpu.utilities.gaussian(self.mu, .5, self.pos)  # cells by positions

        # initialize weights
        if weight_dist == 'uniform':
            self.w = self.rng_.random(size=[n_ca1, n_ca3])  #
        elif weight_dist == 'lognormal':
            self.w = self.rng_.lognormal(sigma=.5, size=[n_ca1, n_ca3])
        else:
            pass

        if eta_ctrl is not None:  # set position dependent gain in learning rate
            ctrl_x = np.linspace(0, 10, num=np.array(eta_ctrl).shape[0])
            self.eta_gain = spline(ctrl_x, eta_ctrl)
            self.eta_gain_mat = np.eye(n_pos) * self.eta_gain(self.pos).T
        else:
            self.eta_gain_mat = np.eye(n_pos)

    def winners(self):
        '''
        determine k-winners
        :return:
        '''

        # initialize CA1 activity
        ca1 = np.zeros([self.n_ca1, self.n_pos])
        # feedforward linear activations
        activations = np.matmul(self.w,
                                self.ca3 + self.ca3_sigma_mag * self.rng_.standard_normal(
                                    size=[self.n_ca3, self.n_pos]))  # noise added to CA3 activity
        winners = np.argsort(activations, axis=0)[::-1, :]  # sort by decreasing activation
        for pos_ind in range(self.n_pos):
            ca1[winners[:self.n_winners, pos_ind], pos_ind] = activations[
                winners[:self.n_winners, pos_ind], pos_ind]  # winners are active
            ca1 += self.ca1_noise * self.rng_.random(size=[self.n_ca1, self.n_pos])  # add noise to all neurons
            ca1 = np.maximum(0, ca1)  # ensure no negative activity
        return ca1

    def forward(self):
        '''
        forward pass of activity and update weights
        :return:
        '''
        ca1 = self.winners()  # get CA1 activity
        # update weights
        self.w += self.eta * np.matmul(np.matmul(ca1, self.eta_gain_mat),
                                       self.ca3.T) - self.tau + self.w_sigma_mag * self.rng_.standard_normal(
            size=self.w.shape)
        # decay weights by norm
        self.w -= self.w_norm_decay * np.linalg.norm(self.w, axis=-1, keepdims=True)
        # bound weights
        self.w = np.minimum(np.maximum(self.w, 0), self.w_max)
        return ca1.T

    # def oja(self):
    #
    #     ca1 = self.winners()
    #     self.w += self.eta * (
    #                 np.matmul(np.matmul(ca1, self.eta_gain_mat), self.ca3.T) - np.matmul(np.power(ca1, 2), self.ca3.T))
    #     self.w = np.minimum(np.maximum(self.w, 0), self.w_max)
    #     return ca1

    def run_trials(self, n_trials: int = 100):
        '''
        'train' a K-Winners-Take-All model on n_trials
        :param n_trials:
        :return: ca1:
        '''
        ca1 = []
        for trial in range(n_trials):
            ca1.append(self.forward())
        return np.array(ca1)


def plot_cells(ca1, cell_inds=None, n_cols=20):
    '''

    :param ca1:
    :param cell_inds: indices of cells to plot
    :param save_figs:
    :return:
    '''

    if cell_inds is None:
        cell_inds = np.arange(ca1.shape[-1])

    n_rows = int(np.ceil(cell_inds.shape[0] / n_cols))
    fig = plt.figure(figsize=[30, 3 * n_rows])
    gs = gridspec(n_rows, n_cols)
    for cell in cell_inds:
        col = cell % n_cols
        row = int(cell / n_cols)
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(ca1[:, :, cell], cmap="Greys")

        if col == 0:
            ax.set_xlabel('pos')
            ax.set_ylabel('trial #')
        else:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.subplots_adjust(hspace=.3)
    return fig


def plot_pop_activity(ca1, trials_to_plot=None):
    '''

    :param ca1: trials x neurons x positions
    :param trials_to_plot:
    :param sort_trial:
    :return:
    '''

    if trials_to_plot is None:
        trials_to_plot = np.arange(ca1.shape[0])
    n_trials = trials_to_plot.shape[0]

    fig, ax = plt.subplots(n_trials, n_trials, figsize= [20, 20])
    sort_vecs = [np.argsort(np.argmax(ca1[trial, :, :], axis=0)) for trial in trials_to_plot]
    ca1_z = (ca1 - ca1.mean(axis=1, keepdims=True)) / (np.std(ca1, axis=1, keepdims=True) + 1E-3)
    for row, sort_vec in zip(trials_to_plot, sort_vecs):

        for col in trials_to_plot:
            ax[row, col].imshow(ca1_z[col, :, sort_vec], vmin=0, vmax=1, aspect='auto',
                                cmap='viridis')
            if col == 0:
                ax[row,col].set_xlabel('Pos.')
                ax[row,col].set_ylabel('Cells')
            else:
                ax[row,col].set_yticks([])

    fig.subplots_adjust(hspace = .5)

    return fig
