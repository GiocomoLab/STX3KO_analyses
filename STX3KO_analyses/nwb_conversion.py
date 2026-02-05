import datetime
import gc
import os
import time
import pathlib
import subprocess
from uuid import uuid4
import hdmf
from hdmf.backends.hdf5.h5_utils import H5DataIO

import numpy as np
import scipy
from natsort import natsorted
import pynwb
from pynwb.file import Subject


import suite2p
from pynwb import NWBHDF5IO, NWBFile
from pynwb.base import Images
from pynwb.image import GrayscaleImage
from pynwb.ophys import (
    Fluorescence,
    ImageSegmentation,
    OpticalChannel,
    RoiResponseSeries,
    TwoPhotonSeries,
)

from pynwb.behavior import BehavioralTimeSeries

import STX3KO_analyses as stx
import TwoPUtils as tpu



SCRATCH_DIR = pathlib.Path("/mnt/BigDisk/2P_scratch")
VR_DIR = pathlib.Path("/mnt/BigDisk/VRData")

OUTPATH = pathlib.Path("/mnt/BigDisk/NWB_files")

class RawDataNWB_Dense:
    
    def __init__(self, mouse, metadata, session, day, oak_pwd, scan=1, sub_notes=''):

        self.mouse = mouse
        self.session = session
        self.metadata = metadata
        self._oak_pwd = oak_pwd
        self.day = day
        self.sub_notes = sub_notes

        self.sess_dir = SCRATCH_DIR / mouse / session.get('date_str') / session.get('scene')
        self.sess_dir.mkdir(parents=True, exist_ok=True)

        self.s2p_path = self.sess_dir /  \
            f"{self.session.get('scene')}_{self.session.get('session'):03}_{self.session.get('scan'):03}" / \
            "suite2p" / "plane0"
        self.s2p_ops = None
        
        self.vr_path = VR_DIR / mouse / session.get('date_str') / f"{session.get('scene')}_{session.get('session')}.sqlite"
        self.vr_data = None
        self.sbx_mat = None
        
        self.nwb_file = None

        self.out_path = OUTPATH / mouse / f"ymaze_day{day}_scan{scan}_ophys_behav_RAW.nwb"
        self.out_path.parent.mkdir(parents=True, exist_ok=True)

    def run_rsync(self):
        remote_user = "mplitt"
        remote_host = "dtn.sherlock.stanford.edu"
        remote_base_path = "/oak/stanford/groups/giocomo/mplitt/2P_Data/STX3KO/"

        cmd = [
            "sshpass", "-p", self._oak_pwd,
            "rsync", "-rlt", "--progress", 
            f"{remote_user}@{remote_host}:{remote_base_path}{self.mouse}/{self.session.get('date_str')}/{self.session.get('scene')}",
            str(self.sess_dir.parent)
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print("Rsync completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Rsync failed with error: {e}")


    def init_nwb_file(self):
        self.nwb_file = NWBFile(
            session_description = "Raw 2P Data and VR Data",
            session_start_time = datetime.datetime.now().astimezone(),
            identifier=str(uuid4()),  # required
        )

        self.nwb_file.subject = Subject(
            subject_id = self.metadata.get('alias'),
            age = self.session.get('datetime') - self.metadata.get('date_of_birth'),
            description = f"YMaze day {self.day}." + self.sub_notes,
            species = 'Mus musculus',
            sex = self.metadata.get('sex'),
            genotype = self.session.get('genotype'),
        )

        # add subject metadata
    
    def add_binary_2Pdata(self):

        self.s2p_ops = np.load(self.s2p_path / "ops.npy", allow_pickle=True).item()

        device = self.nwb_file.create_device(
        name="Microscope",
        description="Giocomo lab Neurolabware 2P Scope",
        manufacturer="Neurolabware",
        )
        
        optical_channel0 = OpticalChannel(
                name="Green PMT",
                description="an optical channel",
                emission_lambda=500.0,
        )
        optical_channel1 = OpticalChannel(
                name="Red PMT",
                description="an optical channel",
                emission_lambda=600.0,
        )

        imaging_plane0 = self.nwb_file.create_imaging_plane(
            name="ImagingPlane_ch0",
            optical_channel=optical_channel0,
            imaging_rate=self.s2p_ops["fs"],
            description="standard",
            device=device,
            excitation_lambda=self.metadata.get('imaging_lambda'),
            indicator=self.metadata.get('functional_indicator'),
            location="CA1",
            grid_spacing=([1000/512., 1000/796.]),
            grid_spacing_unit="microns",
        )

        binary_file_ch0 = self.s2p_path / "data.bin"
        binfile_ch0 = suite2p.io.BinaryFile(self.s2p_ops['Ly'], self.s2p_ops['Lx'], str(binary_file_ch0))
        image_series_ch0 = TwoPhotonSeries(
                name="TwoPhotonSeries_ch0",
                dimension=[self.s2p_ops["Ly"], self.s2p_ops["Lx"]],
                data=H5DataIO(data =binfile_ch0.data, compression="gzip"),
                imaging_plane=imaging_plane0,
                starting_time=0.0,
                rate=self.s2p_ops["fs"],
                unit="n.a.",
            )
        self.nwb_file.add_acquisition(image_series_ch0)

        imaging_plane1 = self.nwb_file.create_imaging_plane(
                name="ImagingPlane_ch1",
                optical_channel=optical_channel1,
                imaging_rate=self.s2p_ops["fs"],
                description="standard",
                device=device,
                excitation_lambda=self.metadata.get('imaging_lambda'),
                indicator=self.metadata.get('static_indicator'),
                location="CA1",
                grid_spacing=([2.0, 2.0]),
                grid_spacing_unit="microns",
            )

        binary_file1 = self.s2p_path / "data_chan2.bin"
        binfile1 = suite2p.io.BinaryFile(self.s2p_ops['Ly'], self.s2p_ops['Lx'], str(binary_file1))
        image_series_ch1 = TwoPhotonSeries(
                name="TwoPhotonSeries_ch1",
                dimension=[self.s2p_ops["Ly"], self.s2p_ops["Lx"]],
                data=H5DataIO(data =binfile_ch0.data, compression="gzip"),
                imaging_plane=imaging_plane1,
                format="external",
                starting_time=0.0,
                rate=self.s2p_ops["fs"] ,
                unit="n.a.",
            )
        self.nwb_file.add_acquisition(image_series_ch1)

    def load_sbx_mat(self):


        self.sbx_mat = tpu.scanner_tools.sbx_utils.loadmat(str(self.sess_dir / \
                    f"{self.session.get('scene')}_{self.session.get('session'):03}_{self.session.get('scan'):03}.mat"))
        
    def get_ttl_times(self):
        sbx_mat = self.sbx_mat

        fr = sbx_mat['frame_rate'] # frame rate
        lr = fr * sbx_mat['config']['lines']/sbx_mat['fov_repeats']  # line rate

        frames = sbx_mat['frame'].astype(int)
        frame_diff = np.ediff1d(frames, to_begin=0)
        try:
            mods = np.argwhere(frame_diff < -100)[0]
            for i, mod in enumerate(mods.tolist()):
                frames[mod:] += (i + 1) * 65535
        except:
            pass
        
        frames = frames * sbx_mat['fov_repeats']
        if sbx_mat['fold_lines']>0:
            lines = np.array([l % sbx_mat['fold_lines'] for l in sbx_mat['line']])
        else:
            lines = np.array(sbx_mat['line'])

        ttl_times = frames / fr + lines / lr
        return ttl_times
        
    def load_vr_data(self):
    

        vr_df = tpu.preprocessing.load_sqlite(self.vr_path)
        ttl_times = self.get_ttl_times()
        
        self.vr_data = vr_df.iloc[-ttl_times.shape[0]:]
        self.vr_data['2P time'] = ttl_times

    
    def add_behav_timeseries(self):
        
        behav_module = self.nwb_file.create_processing_module('VR behavior', 'raw behavior timeseries')
        behav_ts_container = BehavioralTimeSeries(name = 'VR behavior')

        timestamps = self.vr_data['2P time'].to_numpy()

        behav_ts_container.create_timeseries(
            name = 'trial number',
            data = self.vr_data['trialnum'].to_numpy(),
            unit = 'arbitrary',
            description = "trial number",
            timestamps = timestamps,
        )

        behav_ts_container.create_timeseries(
            name = 'position',
            data = self.vr_data['t'].to_numpy(),
            timestamps = timestamps,
            unit = '10 cm',
            description = "Position along spline trajectory. \n \
                    Track starts at a value of 13 and ends at 43. \n \
                    Points less than 13 correspond to when the mouse is \
                        in the grey hallway prior to trial start ",
        )

        behav_ts_container.create_timeseries(
            name = 'x position',
            data = self.vr_data['posx'].to_numpy(),
            timestamps = timestamps,
            unit = 'arbitrary',
            description = "Unity units x position on 2D plane",
        )

        behav_ts_container.create_timeseries(
            name = 'y position',
            data = self.vr_data['posz'].to_numpy(),
            timestamps = timestamps,
            unit = "arbitrary",
            description = "Unity units y position on 2D plane",
        )

        behav_ts_container.create_timeseries(
            name = 'licks',
            data = self.vr_data['lick'].to_numpy(),
            timestamps = timestamps,
            unit = "arbitrary",
            description = "Boolean corresponding to whether capacitve touch sensor on lick port detects a touch. \n \
                Single licks correspond to rising edges",
            comments = "These are the raw touch values. Processing is needed to isolate single licks. \n \
                Occasionally, there are artifacts where the pin value is stuck high from excess moisture on the lickport."
        )

        behav_ts_container.create_timeseries(
            name = 'reward',
            data = self.vr_data['reward'].to_numpy(),
            timestamps = timestamps,
            unit = "arbitrary",
            description = "Boolean corresponding to reward delivery. Only rising edges are informative ",
        )

        behav_ts_container.create_timeseries(
            name = 'manual rewards',
            data = self.vr_data['manrewards'].to_numpy(),
            timestamps = timestamps,
            unit = "arbitrary",
            description = "Boolean corresponding to manual reward delivery. Only rising edges are informative ",
        )

        behav_ts_container.create_timeseries(
            name = 'trial start',
            data = self.vr_data['tstart'].to_numpy(),
            timestamps = timestamps,
            unit = "arbitrary",
            description = "Boolean corresponding to trial starts",
        )

        behav_ts_container.create_timeseries(
            name = 'trial end',
            data = self.vr_data['teleport'].to_numpy(),
            timestamps = timestamps,
            unit = "arbitrary",
            description = "Boolean corresponding to trial ends",
        )

        behav_ts_container.create_timeseries(
            name = 'left_right',
            data = self.vr_data['LR'].to_numpy(),
            timestamps = timestamps,
            unit = "arbitrary",
            description = "-1 = left trial, 1 = right trial",
        )
        
        behav_module.add(behav_ts_container)

    def build_file(self):
        self.init_nwb_file()
        self.add_binary_2Pdata()
        self.load_sbx_mat()
        self.load_vr_data()
        self.add_behav_timeseries()


    def write_nwb(self):
        with NWBHDF5IO(self.out_path, "w") as fio:
            fio.write(self.nwb_file)






    








                


