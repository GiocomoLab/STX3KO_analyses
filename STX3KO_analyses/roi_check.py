import os.path
import sys

import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec as gridspec





class RoiChecker:

    def __init__(self, statfile, opsfile, run_correct_bleedthrough=False,n_tiles = 3,
                 box_size=20):
        """

        :param statfile:
        :param opsfile:
        """
        # load data
        print("Loading data...")
        self.stat = np.load(statfile, allow_pickle=True)
        self.ops = np.load(opsfile, allow_pickle=True).all()
        self.F = np.load("F.npy", allow_pickle=True)
        self.Fneu = np.load("Fneu.npy", allow_pickle=True)
        self.F_chan2 = np.load("F_chan2.npy", allow_pickle=True)
        self.Fneu_chan2 = np.load("Fneu_chan2.npy", allow_pickle=True)
        self.spks = np.load("spks.npy", allow_pickle=True)
        print("Data loaded.")

        self.box_size = box_size
        self.circle_radius = 10
        self.n_tiles = n_tiles

        # meanImgs
        self.green = self.ops['meanImg']
        self.red = self.ops['meanImg_chan2']
        if run_correct_bleedthrough:
            self.ops['meanImg_chan2_corrected'] = correct_bleedthrough(self.ops['Ly'], self.ops['Lx'],
                                                                        self.n_tiles, np.copy(self.ops['meanImg']),
                                                                       np.copy(self.ops['meanImg_chan2']))

        self.red_corr = self.ops['meanImg_chan2_corrected']

        # merged image
        self.merge = np.zeros([self.green.shape[0], self.green.shape[1], 3])
        self.merge[:, :, 1] = self.green / np.amax(self.green)
        self.merge[:, :, 2] = self.red / np.amax(self.red)
        self.merge[:, :, 0] = self.red / np.amax(self.red)

        self.merge_mult = self.green * self.red / np.amax(self.green * self.red)
        self.merge_mult_corr = self.green * self.red_corr / np.amax(self.green * self.red_corr)



        self.fig = None
        self.axdict = None
        self.imhandles = None
        self.init_figure()

        # create pandas dataframe for results
        # check if file exists
        if os.path.isfile("roi_check.csv"):
            self.df = pd.read_csv("roi_check.csv")
            print("roi_check.csv found.")
        else:
            self.df = pd.DataFrame(columns=['roi_index', 'Cre', 'NotCre', 'Unsure', 'NotCell'], index = range(self.stat.shape[0]))


    def loop_rois(self, start_index=0):
        # fig = plt.figure(figsize=(10,10))
        # fig, axdict, imhandles = self.init_figure()
        plt.show(block=False)
        i = start_index
        while i < self.stat.shape[0]:
            i = self.check_roi(i)
            # i += 1
            if i == 'q':
                break
            i += 1
        plt.close()
        self.df.to_csv("roi_check.csv")

    def check_uncurrated(self):
        """

        :return:
        """
        # fig = plt.figure(figsize=(10, 10))
        # fig, axdict, imhandles = self.init_figure()
        plt.show(block=False)
        i = 0
        while i < self.stat.shape[0]:
            if self.df.loc[i,['Cre','NotCre','Unsure','NotCell']].isna().any():
                i = self.check_roi(i)
            # i += 1
            if i == 'q':
                break

            i+=1
        plt.close()
        self.df.to_csv("roi_check.csv")

    def init_figure(self):

        self.fig = plt.figure(figsize=(10, 10))
        gs = gridspec(5, 6)
        self.axdict = {}
        self.imhandles = {}
        self.linehandles = {}


        for row in range(3):
            for col in range(6):
                self.axdict[(row,col)] = self.fig.add_subplot(gs[row, col])
                if col >= 4:
                    self.imhandles[(row, col, 'img')] = self.axdict[(row,col)].imshow(
                        np.zeros((self.box_size*2,self.box_size*2)), cmap='magma')
                else:
                    self.imhandles[(row, col, 'img')] = self.axdict[(row, col)].imshow(
                        np.zeros((self.box_size * 2, self.box_size * 2)), cmap='Greys_r')


        for col in range(6):
            self.imhandles[(0, col, 'roi')] = self.axdict[(0, col)].imshow(np.zeros((self.box_size * 2, self.box_size * 2)),
                                                                        cmap='cool', alpha=.3)
            circle = plt.Circle((self.box_size, self.box_size), self.circle_radius, fill=False, color='blue',
                                linewidth=3)
            self.axdict[(0, col)].add_artist(circle)

            circle = plt.Circle((self.box_size, self.box_size), self.circle_radius, fill=False, color='blue',
                                linewidth=3)
            self.axdict[(1, col)].add_artist(circle)

        t = np.arange(self.F.shape[1])
        self.axdict['green_ts'] = self.fig.add_subplot(gs[3, :])
        (self.linehandles['F'],) = self.axdict['green_ts'].plot(t, t, color ='green')
        (self.linehandles['Fneu'],) = self.axdict['green_ts'].plot(t, t, color ='blue')
        (self.linehandles['spks'],) = self.axdict['green_ts'].plot(t, t, color='black')

        self.axdict['red_ts'] = self.fig.add_subplot(gs[4, :])
        (self.linehandles['F_chan2'],) = self.axdict['red_ts'].plot(t, t, color ='red')
        (self.linehandles['Fneu_chan2'],) = self.axdict['red_ts'].plot(t, t, color='blue')
        self.fig.canvas.draw()


    def plot_roi(self,roi_index):
        """

        :param roi_index:
        :param circle_radius:
        :return:
        """


        # gs = gridspec(5, 6)
        self.fig.suptitle("roi_index: " + str(roi_index))


        # pixels of roi
        com = [self.stat[roi_index]['ypix'].mean(), self.stat[roi_index]['xpix'].mean()]

        # fill roi mask
        roi_mask = np.zeros(self.green.shape) * np.nan
        roi_mask[self.stat[roi_index]['ypix'], self.stat[roi_index]['xpix']]=1

        # roi bounds
        ybounds = [int(max(0, com[0] - self.box_size)), int(min(self.ops['Ly'], com[0] + self.box_size))]
        midy = int((ybounds[1] - ybounds[0]) / 2)
        xbounds = [int(max(0, com[1] - self.box_size)), int(min(self.ops['Lx'], com[1] + self.box_size))]
        midx = int((xbounds[1] - xbounds[0]) / 2)


        # plot green channel
        # w/ roi mask
        green_roi = self.green[ybounds[0]:ybounds[1], xbounds[0]:xbounds[1]]
        self.imhandles[(0,0,'img')].set_data(green_roi)
        self.imhandles[(0,0,'img')].set_clim(vmin=np.amin(green_roi), vmax=np.amax(green_roi))
        self.imhandles[(0,0,'roi')].set_data(roi_mask[ybounds[0]:ybounds[1], xbounds[0]:xbounds[1]])

        # w/ roi circle
        self.imhandles[(1,0,'img')].set_data(green_roi)
        self.imhandles[(1,0,'img')].set_clim(vmin=np.amin(green_roi), vmax=np.amax(green_roi))

        # w/o mark
        self.imhandles[(2,0,'img')].set_data(green_roi)
        self.imhandles[(2,0,'img')].set_clim(vmin=np.amin(green_roi), vmax=np.amax(green_roi))

        # plot red channel
        red_roi = self.red[ybounds[0]:ybounds[1], xbounds[0]:xbounds[1]]
        # w/ roi mask
        self.imhandles[(0,1,'img')].set_data(red_roi)
        self.imhandles[(0,1,'img')].set_clim(vmin=np.amin(red_roi), vmax=np.amax(red_roi))
        self.imhandles[(0,1,'roi')].set_data(roi_mask[ybounds[0]:ybounds[1], xbounds[0]:xbounds[1]])

        # w/ roi circle
        self.imhandles[(1,1,'img')].set_data(red_roi)
        self.imhandles[(1,1,'img')].set_clim(vmin=np.amin(red_roi), vmax=np.amax(red_roi))

        # w/o mark
        self.imhandles[(2,1,'img')].set_data(red_roi)
        self.imhandles[(2,1,'img')].set_clim(vmin=np.amin(red_roi), vmax=np.amax(red_roi))

        # plot corrected red channel
        red_corr_roi = self.red_corr[ybounds[0]:ybounds[1], xbounds[0]:xbounds[1]]
        # w/ roi mask
        self.imhandles[(0,2,'img')].set_data(red_corr_roi)
        self.imhandles[(0,2,'img')].set_clim(vmin=np.amin(red_corr_roi), vmax=np.amax(red_corr_roi))
        self.imhandles[(0,2,'roi')].set_data(roi_mask[ybounds[0]:ybounds[1], xbounds[0]:xbounds[1]])
        # w/ roi circle
        self.imhandles[(1,2,'img')].set_data(red_corr_roi)
        self.imhandles[(1,2,'img')].set_clim(vmin=np.amin(red_corr_roi), vmax=np.amax(red_corr_roi))
        # w/o mark
        self.imhandles[(2,2,'img')].set_data(red_corr_roi)
        self.imhandles[(2,2,'img')].set_clim(vmin=np.amin(red_corr_roi), vmax=np.amax(red_corr_roi))

        # plot merge
        merge_roi = self.merge[ybounds[0]:ybounds[1], xbounds[0]:xbounds[1]]
        # w/ roi mask
        self.imhandles[(0,3,'img')].set_data(merge_roi)
        self.imhandles[(0,3,'img')].set_clim(vmin=0, vmax=1)
        self.imhandles[(0,3,'roi')].set_data(roi_mask[ybounds[0]:ybounds[1], xbounds[0]:xbounds[1]])
        # w/ roi circle
        self.imhandles[(1,3,'img')].set_data(merge_roi)
        self.imhandles[(1,3,'img')].set_clim(vmin=0, vmax=1)
        # w/o mark
        self.imhandles[(2,3,'img')].set_data(merge_roi)
        self.imhandles[(2,3,'img')].set_clim(vmin=0, vmax=1)

        # plot merge multiplied channel
        merge_roi_mult = self.merge_mult[ybounds[0]:ybounds[1], xbounds[0]:xbounds[1]]
        # w/ roi mask
        self.imhandles[(0,4,'img')].set_data(merge_roi_mult)
        self.imhandles[(0,4,'img')].set_clim(vmin=0, vmax=.5)
        self.imhandles[(0,4,'roi')].set_data(roi_mask[ybounds[0]:ybounds[1], xbounds[0]:xbounds[1]])
        # w/ roi circle
        self.imhandles[(1,4,'img')].set_data(merge_roi_mult)
        self.imhandles[(1,4,'img')].set_clim(vmin=0, vmax=.5)
        # w/o mark
        self.imhandles[(2,4,'img')].set_data(merge_roi_mult)


        # plot merge multiplied (red corrected) channel
        merge_roi_mult_corr = self.merge_mult_corr[ybounds[0]:ybounds[1], xbounds[0]:xbounds[1]]
        # w/ roi mask
        self.imhandles[(0,5,'img')].set_data(merge_roi_mult_corr)
        self.imhandles[(0,5,'img')].set_clim(vmin=0, vmax=1E-5)
        self.imhandles[(0,5,'roi')].set_data(roi_mask[ybounds[0]:ybounds[1], xbounds[0]:xbounds[1]])
        # w/ roi circle
        self.imhandles[(1,5,'img')].set_data(merge_roi_mult_corr)
        self.imhandles[(1,5,'img')].set_clim(vmin=0, vmax=1E-5)
        # w/o mark
        self.imhandles[(2,5,'img')].set_data(merge_roi_mult_corr)

        # plot green channel timeseries
        self.linehandles['F'].set_ydata(self.F[roi_index,:])
        self.linehandles['Fneu'].set_ydata(self.Fneu[roi_index,:])
        self.linehandles['spks'].set_ydata(self.spks[roi_index,:])

        # plot red channel timeseries
        self.linehandles['F_chan2'].set_ydata(self.F_chan2[roi_index,:])
        self.linehandles['Fneu_chan2'].set_ydata(self.Fneu_chan2[roi_index,:])


        self.fig.show()
        self.fig.canvas.draw()
        # self.fig.canvas.blit(self.fig.bbox)
        self.fig.canvas.flush_events()

        # fig.show()

        # return fig




    def check_roi(self, roi_index):
        """

        :param roi_index:
        :return:
        """

        # if fig is None:
        #     fig = self.plot_roi(roi_index)
        # else:
        self.plot_roi(roi_index)

        print("Cell %d of %d \n" % (roi_index, self.stat.shape[0]))
        cmd = input("1=Cre, 2=NotCre, 3=Unsure, 4=NotCell, s=skip, r=replot, b=back, c=skip to specific roi, q=quit \n")
        if cmd == '1':
            self.df.loc[roi_index, 'Cre'] = 1
            for col in self.df.columns:
                if col != 'Cre' and col != 'roi_index':
                    self.df.loc[roi_index, col] = 0
        elif cmd == '2':
            self.df.loc[roi_index, 'NotCre'] = 1
            for col in self.df.columns:
                if col != 'NotCre' and col != 'roi_index':
                    self.df.loc[roi_index, col] = 0
        elif cmd == '3':
            self.df.loc[roi_index, 'Unsure'] = 1
            for col in self.df.columns:
                if col != 'Unsure' and col != 'roi_index':
                    self.df.loc[roi_index, col] = 0

        elif cmd == '4':
            self.df.loc[roi_index, 'NotCell'] = 1
            # fig.close()
            for col in self.df.columns:
                if col != 'NotCell' and col != 'roi_index':
                    self.df.loc[roi_index, col] = 0
        elif cmd == 's':
            pass
        elif cmd == 'c':
            roi_index = int(input("roi index: "))
            roi_index = self.check_roi(roi_index)
        elif cmd == 'r':
            # fig.close()
            roi_index = self.check_roi(roi_index)

        elif cmd == 'b':
            # fig.close()
            roi_index = self.check_roi(roi_index-1)
        elif cmd == 'q':

            if input("save?") in ['1', 'y', 'yes','Y','Yes','']:
                self.df.to_csv("roi_check.csv")
            else:
                print("not saving")
            plt.close(self.fig)
            return cmd

        else:
            print("invalid input")
            roi_index = self.check_roi(roi_index)

        return roi_index


def correct_bleedthrough(Ly, Lx, nblks, mimg, mimg2):
    """

    :param Ly:
    :param Lx:
    :param nblks:
    :param mimg:
    :param mimg2:
    :return:
    """
    # subtract bleedthrough of green into red channel
    # non-rigid regression with nblks x nblks pieces
    sT = np.round((Ly + Lx) / (nblks * 2) * 0.25)
    mask = np.zeros((Ly, Lx, nblks, nblks), np.float32)
    weights = np.zeros((nblks, nblks), np.float32)
    yb = np.linspace(0, Ly, nblks + 1).astype(int)
    xb = np.linspace(0, Lx, nblks + 1).astype(int)
    for iy in range(nblks):
        for ix in range(nblks):
            ny = np.arange(yb[iy], yb[iy + 1]).astype(int)
            nx = np.arange(xb[ix], xb[ix + 1]).astype(int)
            mask[:, :, iy, ix] = quadrant_mask(Ly, Lx, ny, nx, sT)
            x = mimg[np.ix_(ny, nx)].flatten()
            x2 = mimg2[np.ix_(ny, nx)].flatten()
            # predict chan2 from chan1
            a = (x * x2).sum() / (x * x).sum()
            weights[iy, ix] = a
    mask /= mask.sum(axis=-1).sum(axis=-1)[:, :, np.newaxis, np.newaxis]
    mask *= weights
    mask *= mimg[:, :, np.newaxis, np.newaxis]
    mimg2 -= mask.sum(axis=-1).sum(axis=-1)
    mimg2 = np.maximum(0, mimg2)
    return mimg2


def quadrant_mask(Ly, Lx, ny, nx, sT):
    mask = np.zeros((Ly, Lx), np.float32)
    mask[np.ix_(ny, nx)] = 1
    mask = sp.ndimage.gaussian_filter(mask, sT)
    return mask


















if __name__ == "__main__":

    # check if arguments exist
    if len(sys.argv) < 2:
        statfile = "stat.npy"
        opsfile = "ops.npy"
    elif len(sys.argv)==2:
        statfile = sys.argv[1]
    elif len(sys.argv)==3:
        statfile = sys.argv[1]
        opsfile = sys.argv[2]
    else:
        print("Too many arguments")
        sys.exit()

    # load data
    if os.path.exists(statfile) and os.path.exists(opsfile):
        print("Beginning ROI check...")
        roi_checker = RoiChecker(statfile, opsfile)
        print("Data loaded.")
    else:
        print("File not found.")
        sys.exit()

