import datetime
import copy
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
    DfOverF,
)

from pynwb.behavior import BehavioralTimeSeries

import STX3KO_analyses as stx
import TwoPUtils as tpu



# SCRATCH_DIR = pathlib.Path("/mnt/BigDisk/2P_scratch")
# VR_DIR = pathlib.Path("/mnt/BigDisk/VRData")


OUTPATH = pathlib.Path("/mnt/BigDisk/NWB_files")
SESSPATH = pathlib.Path("/home/mplitt/YMazeSessPkls")
VRSESSPATH = pathlib.Path("/home/mplitt/YMaze_VR_Pkls")
SBXMATPATH = pathlib.Path("/mnt/BigDisk/2P_scratch")


class SessNWBConverter_Dense:
    
    def __init__(self, mouse, metadata, session, day, oak_pwd, scan=1, sub_notes=''):

        self.mouse = mouse
        if mouse in stx.mouse_metadata.ctrl_sessions.keys():
            self.sub_description = f"Control mouse. Viruses: {metadata.get('functional_indicator')} \
                {metadata.get('static_indicator')}"
        elif mouse in stx.mouse_metadata.cre_sessions.keys():
            self.sub_description = f"Cre mouse. Viruses: {metadata.get('functional_indicator')} \
                {metadata.get('static_indicator')}"
        else:
            raise ValueError("Mouse name must be in ctrl or cre mice metadata")
        
        self.session = session
        self.metadata = metadata
        self._oak_pwd = oak_pwd
        self.day = day
        self.sub_notes = sub_notes

        self.sess_path = SESSPATH / mouse / session.get('date_str') / f"{session.get('scene')}_{session.get('session')}.pkl" 
        self.sess = stx.session.YMazeSession.from_file(self.sess_path, novel_arm = session.get('novel_arm'))
        
        self.vr_sess_path = VRSESSPATH / mouse / session.get('date_str') / f"{session.get('scene')}_{session.get('session')}.pkl"
        self.vr_sess = stx.session.YMazeSession.from_file(self.vr_sess_path, novel_arm = session.get('novel_arm'))
        
        self.sbx_mat_path = SBXMATPATH / mouse / session.get('date_str') / \
            f"{session.get('scene')}_{session.get('session'):03}_{session.get('scan'):03}.mat"
        self.sbx_path = SBXMATPATH / mouse / session.get('date_str') / \
            f"{session.get('scene')}_{session.get('session'):03}_{session.get('scan'):03}.sbx"
        self.sbx_mat_path.parent.mkdir(exist_ok=True, parents=True)
        self.sbx_mat = None
        
        self.nwb_file = None
        self.behav_module = None
        self.ophys_module = None
        self.roi_table = None
        

        self.out_path = OUTPATH / mouse / f"ymaze_day{day}_scan{scan}_ophys_behav.nwb"
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        
    def _sbx_rsync(self):
        remote_user = "mplitt"
        remote_host = "dtn.sherlock.stanford.edu"
        remote_base_path = pathlib.Path("/oak/stanford/groups/giocomo/mplitt/2P_Data/STX3KO/")
        
        session_dir = remote_base_path / self.mouse / self.session.get("date_str") / self.session.get("scene")
        sbx_mat_filename = f"{self.session.get('scene')}_{self.session.get('session'):03}_{self.session.get('scan'):03}.mat"
        sbx_mat_path = str(session_dir / sbx_mat_filename)

        cmd = [
            "sshpass", "-p", self._oak_pwd,
            "rsync", "-rlt", "--progress", 
            f"{remote_user}@{remote_host}:{sbx_mat_path}",
            str(self.sbx_mat_path) 
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print("Rsync completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Rsync failed with error: {e}")
            
        sbx_filename = f"{self.session.get('scene')}_{self.session.get('session'):03}_{self.session.get('scan'):03}.sbx"
        sbx_path = str(session_dir / sbx_filename)

        cmd = [
            "sshpass", "-p", self._oak_pwd,
            "rsync", "-rlt", "--progress", 
            f"{remote_user}@{remote_host}:{sbx_path}",
            str(self.sbx_path) 
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print("Rsync completed successfully.")
        except subprocess.CalledProcessError as e:
            raise Exception(f"Rsync failed with error: {e}")
            
            
            
    def _load_sbx_mat(self):
        
        if self.sbx_mat_path.is_file() and self.sbx_path.is_file():
            self.sbx_mat = tpu.scanner_tools.sbx_utils.loadmat(str(self.sbx_mat_path))
        else:
            self._sbx_rsync()
            self._load_sbx_mat()
        
    def _get_ttl_times(self):
        if self.sbx_mat is None:
            self._load_sbx_mat()
            
        

        fr = self.sbx_mat['frame_rate'] # frame rate
        lr = fr * self.sbx_mat['config']['lines']/self.sbx_mat['fov_repeats']  # line rate

        frames = self.sbx_mat['frame'].astype(int)
        frame_diff = np.ediff1d(frames, to_begin=0)
        try:
            mods = np.argwhere(frame_diff < -100)[0]
            for i, mod in enumerate(mods.tolist()):
                frames[mod:] += (i + 1) * 65535
        except:
            pass
        
        frames = frames * self.sbx_mat['fov_repeats']
        if self.sbx_mat['fold_lines']>0:
            lines = np.array([l % self.sbx_mat['fold_lines'] for l in self.sbx_mat['line']])
        else:
            lines = np.array(self.sbx_mat['line'])

        ttl_times = frames / fr + lines / lr
        return ttl_times

    def init_nwb_file(self):
        self.nwb_file = NWBFile(
            session_description = "Preprocessed 2P and VR Data",
            session_start_time = datetime.datetime.now().astimezone(),
            identifier=str(uuid4()),  # required
            experimenter = ['Plitt, Mark'],
            lab="Lisa Giocomo",
            institution="Stanford University",
            experiment_description =  f"YMaze day {self.day}. Novel arm = {self.session.get('novel_arm')}." + self.sub_notes,
            related_publications='https://doi.org/10.1101/2023.11.20.567978 ',
            keywords=["two photon", "hipppocampus", "CA1", "syntaxin3"]
        )

        self.nwb_file.subject = Subject(
            subject_id = self.metadata.get('alias'),
            age = self.session.get('datetime') - self.metadata.get('date_of_birth'),
            description = self.sub_description,
            species = 'Mus musculus',
            sex = self.metadata.get('sex'),
            genotype = self.session.get('genotype'),
        )
        
        self.behavior_module = self.nwb_file.create_processing_module('behavior', 'VR behavioral timeseries')


    def add_vr_data_full_res(self):
        ts_cntnr = BehavioralTimeSeries(name = 'Full temporal resolution behavior')
        
        time_stamps = self._get_ttl_times()
        
        
        vr_timeseries = {k: v[:,-time_stamps.shape[0]:] for k,v in self.vr_sess.timeseries.items()}
        vr_data = self.vr_sess.vr_data.iloc[-time_stamps.shape[0]:]
        
        
        # vr_data info
        
        # trial num 
        ts_cntnr.create_timeseries(
            name = 'trial number',
            data = vr_data['trialnum'].to_numpy(),
            unit = 'arbitrary',
            description = 'current trial number',
            timestamps = time_stamps,
        )
        
        # t - spline position
        ts_cntnr.create_timeseries(
            name = 'position',
            data = vr_data['t'].to_numpy(),
            unit = '10 cm',
            timestamps = time_stamps,
            description = "Position along spline trajectory. \n \
                    Track starts at a value of 13 and ends at 43. \n \
                    Points less than 13 correspond to when the mouse is \
                    in the grey hallway prior to trial start ",
        )
        
        # posx 
        ts_cntnr.create_timeseries(
            name = 'x position',
            data = vr_data['posx'].to_numpy(),
            unit = 'arbitrary',
            timestamps = time_stamps,
            description = "Unity units x position on 2D plane",
            
        )
        
        # posz
        ts_cntnr.create_timeseries(
            name = 'y position',
            data = vr_data['posz'].to_numpy(),
            unit = 'arbitrary',
            timestamps = time_stamps,
            description = "Unity units y position on 2D plane",
        )
        
        
        # dz 
        ts_cntnr.create_timeseries(
            name = 'rotary encoder reading',
            data = vr_data['dz'].to_numpy(),
            unit = '10 cm',
            timestamps = time_stamps,
            description = "Scaled rotary encoder output. Raw speed of mouse. During timeouts, visual speed is 0",
        )
        
        # tstart
        ts_cntnr.create_timeseries(
            name = 'trial start',
            data = vr_data['tstart'].to_numpy(),
            unit = 'arbitrary',
            timestamps = time_stamps,
            description = "Boolean. Trial start time",
        )
        # teleport
        ts_cntnr.create_timeseries(
            name = 'trial end',
            data = vr_data['teleport'].to_numpy(),
            unit = 'arbitrary',
            timestamps = time_stamps,
            description = "Boolean. Trial end/teleport time",
        )
        
        # LR
        ts_cntnr.create_timeseries(
            name = 'left or right',
            data = vr_data['LR'].to_numpy(),
            unit = 'arbitrary',
            timestamps = time_stamps,
            description = "-1 = left trial. 1 = right trial",
        )
        
        # manrewards
        ts_cntnr.create_timeseries(
            name = 'manual rewards',
            data = vr_data['manrewards'].to_numpy(),
            unit = 'arbitrary',
            timestamps = time_stamps,
            description = "Boolean. Manually delivered reward, typically for solenoid failure or to unclog line",
        )
        
        # vr_timeseries 
        # speed
        ts_cntnr.create_timeseries(
            name = 'speed',
            data = vr_timeseries['speed'].ravel(),
            unit = '10 cm/s',
            timestamps = time_stamps,
            description = "Speed along Y maze",
        )
        
        # block
        ts_cntnr.create_timeseries(
            name = 'block',
            data = vr_timeseries['block'].ravel(),
            unit = 'arbitrary',
            timestamps = time_stamps,
            description = "current block",
        )
        
        # nonconsum_licks
        ts_cntnr.create_timeseries(
            name = 'non-consummatory licks',
            data = vr_timeseries['nonconsum_licks'].ravel(),
            unit = 'arbitrary',
            timestamps = time_stamps,
            description = "Licks outside of reward consumption. Note this may contain artifacts from periods when \n \
                there is excess liquid on the capacitive sensor",
        )
        
        ts_cntnr.create_timeseries(
            name = 'consummatory licks',
            data = vr_timeseries['consum_licks'].ravel(),
            unit = 'arbitrary',
            timestamps = time_stamps,
            description = "Licks during reward consumption",
        )
        
        # reward
        ts_cntnr.create_timeseries(
            name = 'reward',
            data = vr_timeseries['reward'].ravel(),
            unit = 'arbitrary',
            timestamps = time_stamps,
            description = "Reward delivery times.",
        )
        
        
        self.behavior_module.add(ts_cntnr)

    def add_vr_data_aligned(self):
        ts_cntnr = BehavioralTimeSeries(name = '2P-aligned behavior')
        

        
        
        vr_timeseries = self.sess.timeseries
        vr_data = self.sess.vr_data
        time_stamps = vr_data['time'].to_numpy()
        rate = self.sess.s2p_ops['fs']
        
        
        # vr_data info
        
        # trial num 
        ts_cntnr.create_timeseries(
            name = 'trial number',
            data = vr_data['trialnum'].to_numpy(),
            unit = 'arbitrary',
            description = 'current trial number',
            # timestamps = time_stamps,
            rate=rate,
            starting_time=time_stamps[0],
        )
        
        # t - spline position
        ts_cntnr.create_timeseries(
            name = 'position',
            data = vr_data['t'].to_numpy(),
            unit = '10 cm',
            # timestamps = time_stamps,
            rate=rate,
            starting_time=time_stamps[0],
            description = "Position along spline trajectory. \n \
                    Track starts at a value of 13 and ends at 43. \n \
                    Points less than 13 correspond to when the mouse is \
                    in the grey hallway prior to trial start ",
        )
        
        # posx 
        ts_cntnr.create_timeseries(
            name = 'x position',
            data = vr_data['posx'].to_numpy(),
            unit = 'arbitrary',
            # timestamps = time_stamps,
            rate=rate,
            starting_time=time_stamps[0],
            description = "Unity units x position on 2D plane",
            
        )
        
        # posz
        ts_cntnr.create_timeseries(
            name = 'y position',
            data = vr_data['posz'].to_numpy(),
            unit = 'arbitrary',
            # timestamps = time_stamps,
            rate=rate,
            starting_time=time_stamps[0],
            description = "Unity units y position on 2D plane",
        )
        
        # tstart
        ts_cntnr.create_timeseries(
            name = 'trial start',
            data = vr_data['tstart'].to_numpy(),
            unit = 'arbitrary',
            # timestamps = time_stamps,
            rate=rate,
            starting_time=time_stamps[0],
            description = "Boolean. Trial start time",
        )
        # teleport
        ts_cntnr.create_timeseries(
            name = 'trial end',
            data = vr_data['teleport'].to_numpy(),
            unit = 'arbitrary',
            # timestamps = time_stamps,
            rate=rate,
            starting_time=time_stamps[0],
            description = "Boolean. Trial end/teleport time",
        )
        
        # LR
        ts_cntnr.create_timeseries(
            name = 'left or right',
            data = vr_data['LR'].to_numpy(),
            unit = 'arbitrary',
            # timestamps = time_stamps,
            rate=rate,
            starting_time=time_stamps[0],
            description = "-1 = left trial. 1 = right trial",
        )
        
        # vr_timeseries 
        # speed
        ts_cntnr.create_timeseries(
            name = 'speed',
            data = vr_timeseries['speed'].ravel(),
            unit = '10 cm/s',
            # timestamps = time_stamps,
            rate=rate,
            starting_time=time_stamps[0],
            description = "Speed along Y maze",
        )
        
        # block
        ts_cntnr.create_timeseries(
            name = 'block',
            data = vr_timeseries['block_number'].ravel(),
            unit = 'arbitrary',
            # timestamps = time_stamps,
            rate=rate,
            starting_time=time_stamps[0],
            description = "current block",
        )
        
        # lick rate
        ts_cntnr.create_timeseries(
            name = 'licks',
            data = vr_data['lick'].to_numpy(),
            unit = 'avg number of licks',
            # timestamps = time_stamps,
            rate=rate,
            starting_time=time_stamps[0],
            description = "Average downsampled lick rate. Do not use for quantitative lick comparisons between groups",
        )
        
        # reward
        ts_cntnr.create_timeseries(
            name = 'reward',
            data = vr_timeseries['reward'].ravel(),
            unit = 'arbitrary',
            # timestamps = time_stamps,
            rate=rate,
            starting_time=time_stamps[0],
            description = "Reward delivery times.",
        )
        
        self.behavior_module.add(ts_cntnr)
        
    def init_2p_data(self):
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

        imaging_plane = self.nwb_file.create_imaging_plane(
            name="ImagingPlane",
            optical_channel=[optical_channel0, optical_channel1],
            indicator='channel 0: jGCaMP7f, channel 1: mCherry',
            imaging_rate=self.sess.s2p_ops["fs"],
            description="CA1 pyramidal cell layer",
            device=device,
            excitation_lambda=self.metadata.get('imaging_lambda'),
            location="CA1",
            grid_spacing=([1000/512., 1000/796.]),
            grid_spacing_unit="microns",
        )
        
        
        img_seg = ImageSegmentation()
        ps = img_seg.create_plane_segmentation(
            name="PlaneSegmentation",
            description="Suite2P output",
            imaging_plane=imaging_plane,
        )
        
        self.ophys_module = self.nwb_file.create_processing_module(
            name="ophys", description="2P imaging data"
        )
        self.ophys_module.add(img_seg)
        
        stat = self.sess.s2p_stats
        for n in range(len(stat)):
            pixel_mask = np.array(
                [stat[n]['ypix'], stat[n]['xpix'], stat[n]['lam']]
            )
            ps.add_roi(pixel_mask=pixel_mask.T)
            
        self.roi_table = ps.create_roi_table_region(
            region = [i for i in range(len(stat))],
            description="ROIs"
        )
        
        images = Images("Backgrounds", description='motion aligned average images')
        images.add_image(GrayscaleImage(name='meanImg_ch0', 
                                        data=self.sess.s2p_ops['meanImg'],
                                        description='average channel 0 (gcamp) image'))
        images.add_image(GrayscaleImage(name='meanImg_ch1', 
                                        data=self.sess.s2p_ops['meanImg_chan2'],
                                        description='average channel 1 (mCherry) image'))
        self.ophys_module.add(images)


    def add_cell_timeseries(self):
        if self.ophys_module is None:
            self.init_2p_data()
            
        F = self.sess.timeseries.get('F')
        roi_resp_series = RoiResponseSeries(
            name = 'fluorescence',
            data = F.T,
            rois = self.roi_table,
            unit = 'arbitrary',
            rate = self.sess.s2p_ops['fs'],
            description = 'raw fluorescence from channel 0 (gcamp)',
        )
        fl = Fluorescence(roi_response_series=roi_resp_series, name='fluorescence')
        self.ophys_module.add(fl)
        
        Fneu = self.sess.timeseries.get('Fneu')
        roi_resp_series = RoiResponseSeries(
            name = 'neuropil fluorescence',
            data = Fneu.T,
            rois = self.roi_table,
            unit = 'arbitrary',
            rate = self.sess.s2p_ops['fs'],
            description = 'raw neuropil fluorescence from channel 0 (gcamp)',
        )
        fl = Fluorescence(roi_response_series=roi_resp_series, name='neuropil')
        self.ophys_module.add(fl)
        
        dff = self.sess.timeseries.get('F_dff')
        roi_resp_series = RoiResponseSeries(
            name = 'dF',
            data = dff.T,
            rois = self.roi_table,
            unit = 'arbitrary',
            rate = self.sess.s2p_ops['fs'],
            description = 'dF/F from channel 0 (gcamp)',
        )
        fl = DfOverF(roi_response_series=roi_resp_series, name='dF')
        self.ophys_module.add(fl)
        
        
    def build_file(self):
        self.init_nwb_file()
        self.add_vr_data_full_res()
        self.add_vr_data_aligned()
        self.init_2p_data()
        self.add_cell_timeseries()
        return self
        
    def write_file(self):
        with NWBHDF5IO(self.out_path, "w") as fio:
            fio.write(self.nwb_file)
            
    def remove_sbx_data(self):
        self.sbx_mat_path.unlink(missing_ok=True)
        self.sbx_path.unlink(missing_ok=True)
        
    

class SessNWBConverter_Sparse:
    
    
    def __init__(self, mouse, metadata, session, day, oak_pwd, scan=1, sub_notes=''):

        self.mouse = mouse
        if mouse in stx.mouse_metadata.sparse_sessions.keys():
            viruses = '' 
            for fv in metadata.get('functional_indicator'):
                viruses += f"{fv}, "
            for rv in metadata.get('recombinase_viruses'):
                viruses += f"{rv}, "
            self.sub_description = f"Sparse mouse. Viruses: {viruses}"
        else:
            raise ValueError("Mouse name must be in ctrl or cre mice metadata")
        
        self.session = session
        self.metadata = metadata
        self._oak_pwd = oak_pwd
        self.day = day
        self.sub_notes = sub_notes

        self.sess_path = SESSPATH / mouse / session.get('date_str') / f"{session.get('scene')}_{session.get('session')}.pkl" 
        self.sess = stx.session.YMazeSession.from_file(self.sess_path, novel_arm = session.get('novel_arm'))
        
        self.vr_sess_path = VRSESSPATH / mouse / session.get('date_str') / f"{session.get('scene')}_{session.get('session')}.pkl"
        self.vr_sess = stx.session.YMazeSession.from_file(self.vr_sess_path, novel_arm = session.get('novel_arm'))
        
        self.sbx_mat_path = SBXMATPATH / mouse / session.get('date_str') / \
            f"{session.get('scene')}_{session.get('session'):03}_{session.get('scan'):03}.mat"
        self.sbx_path = SBXMATPATH / mouse / session.get('date_str') / \
            f"{session.get('scene')}_{session.get('session'):03}_{session.get('scan'):03}.sbx"
        self.sbx_mat_path.parent.mkdir(exist_ok=True, parents=True)
        self.sbx_mat = None
        
        self.nwb_file = None
        self.behav_module = None
        self.ophys_module = None
        self.roi_table_channel0 = None
        self.roi_table_channel1 = None
        

        self.out_path = OUTPATH / mouse / f"ymaze_day{day}_scan{scan}_ophys_behav.nwb"
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        
    def _sbx_rsync(self):
        remote_user = "mplitt"
        remote_host = "dtn.sherlock.stanford.edu"
        remote_base_path = pathlib.Path("/oak/stanford/groups/giocomo/mplitt/2P_Data/STX3KO/")
        
        session_dir = remote_base_path / self.mouse / self.session.get("date_str") / self.session.get("scene")
        sbx_mat_filename = f"{self.session.get('scene')}_{self.session.get('session'):03}_{self.session.get('scan'):03}.mat"
        sbx_mat_path = str(session_dir / sbx_mat_filename)

        cmd = [
            "sshpass", "-p", self._oak_pwd,
            "rsync", "-rlt", "--progress", 
            f"{remote_user}@{remote_host}:{sbx_mat_path}",
            str(self.sbx_mat_path) 
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print("Rsync completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Rsync failed with error: {e}")
            
        sbx_filename = f"{self.session.get('scene')}_{self.session.get('session'):03}_{self.session.get('scan'):03}.sbx"
        sbx_path = str(session_dir / sbx_filename)

        cmd = [
            "sshpass", "-p", self._oak_pwd,
            "rsync", "-rlt", "--progress", 
            f"{remote_user}@{remote_host}:{sbx_path}",
            str(self.sbx_path) 
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print("Rsync completed successfully.")
        except subprocess.CalledProcessError as e:
            raise Exception(f"Rsync failed with error: {e}")
            
            
            
    def _load_sbx_mat(self):
        
        if self.sbx_mat_path.is_file() and self.sbx_path.is_file():
            self.sbx_mat = tpu.scanner_tools.sbx_utils.loadmat(str(self.sbx_mat_path))
        else:
            self._sbx_rsync()
            self._load_sbx_mat()
        
    def _get_ttl_times(self):
        if self.sbx_mat is None:
            self._load_sbx_mat()
            
        

        fr = self.sbx_mat['frame_rate'] # frame rate
        lr = fr * self.sbx_mat['config']['lines']/self.sbx_mat['fov_repeats']  # line rate

        frames = self.sbx_mat['frame'].astype(int)
        frame_diff = np.ediff1d(frames, to_begin=0)
        try:
            mods = np.argwhere(frame_diff < -100)[0]
            for i, mod in enumerate(mods.tolist()):
                frames[mod:] += (i + 1) * 65535
        except:
            pass
        
        frames = frames * self.sbx_mat['fov_repeats']
        if self.sbx_mat['fold_lines']>0:
            lines = np.array([l % self.sbx_mat['fold_lines'] for l in self.sbx_mat['line']])
        else:
            lines = np.array(self.sbx_mat['line'])

        ttl_times = frames / fr + lines / lr
        return ttl_times

    def init_nwb_file(self):
        self.nwb_file = NWBFile(
            session_description = "Preprocessed 2P and VR Data",
            session_start_time = datetime.datetime.now().astimezone(),
            identifier=str(uuid4()),  # required
            experimenter = ['Plitt, Mark'],
            lab="Lisa Giocomo",
            institution="Stanford University",
            notes=self.metadata.notes,
            experiment_description =  f"YMaze day {self.day}. Novel arm = {self.session.get('novel_arm')}." + self.sub_notes,
            related_publications='https://doi.org/10.1101/2023.11.20.567978 ',
            keywords=["two photon", "hipppocampus", "CA1", "syntaxin3"]
        )

        self.nwb_file.subject = Subject(
            subject_id = self.metadata.get('alias'),
            age = self.session.get('datetime') - self.metadata.get('date_of_birth'),
            description = self.sub_description,
            species = 'Mus musculus',
            sex = self.metadata.get('sex'),
            genotype = self.session.get('genotype'),
        )
        
        self.behavior_module = self.nwb_file.create_processing_module('behavior', 'VR behavioral timeseries')


    def add_vr_data_full_res(self):
        ts_cntnr = BehavioralTimeSeries(name = 'Full temporal resolution behavior')
        
        time_stamps = self._get_ttl_times()
        
        
        vr_timeseries = {k: v[:,-time_stamps.shape[0]:] for k,v in self.vr_sess.timeseries.items()}
        vr_data = self.vr_sess.vr_data.iloc[-time_stamps.shape[0]:]
        
        
        # vr_data info
        
        # trial num 
        ts_cntnr.create_timeseries(
            name = 'trial number',
            data = vr_data['trialnum'].to_numpy(),
            unit = 'arbitrary',
            description = 'current trial number',
            timestamps = time_stamps,
        )
        
        # t - spline position
        ts_cntnr.create_timeseries(
            name = 'position',
            data = vr_data['t'].to_numpy(),
            unit = '10 cm',
            timestamps = time_stamps,
            description = "Position along spline trajectory. \n \
                    Track starts at a value of 13 and ends at 43. \n \
                    Points less than 13 correspond to when the mouse is \
                    in the grey hallway prior to trial start ",
            comments = "Timestamps do not correspond to 2P imaging time. \n \
                Cannot be used for alignment to 2P Data"
        )
        
        # posx 
        ts_cntnr.create_timeseries(
            name = 'x position',
            data = vr_data['posx'].to_numpy(),
            unit = 'arbitrary',
            timestamps = time_stamps,
            description = "Unity units x position on 2D plane",
            comments = "Timestamps do not correspond to 2P imaging time. \n \
                Cannot be used for alignment to 2P Data"
        )
        
        # posz
        ts_cntnr.create_timeseries(
            name = 'y position',
            data = vr_data['posz'].to_numpy(),
            unit = 'arbitrary',
            timestamps = time_stamps,
            description = "Unity units y position on 2D plane",
            comments = "Timestamps do not correspond to 2P imaging time. \n \
                Cannot be used for alignment to 2P Data"
        )
        
        
        # dz 
        ts_cntnr.create_timeseries(
            name = 'rotary encoder reading',
            data = vr_data['dz'].to_numpy(),
            unit = '10 cm',
            timestamps = time_stamps,
            description = "Scaled rotary encoder output. Raw speed of mouse. During timeouts, visual speed is 0",
            comments = "Timestamps do not correspond to 2P imaging time. \n \
                Cannot be used for alignment to 2P Data"
        )
        
        # tstart
        ts_cntnr.create_timeseries(
            name = 'trial start',
            data = vr_data['tstart'].to_numpy(),
            unit = 'arbitrary',
            timestamps = time_stamps,
            description = "Boolean. Trial start time",
            comments = "Timestamps do not correspond to 2P imaging time. \n \
                Cannot be used for alignment to 2P Data"
        )
        # teleport
        ts_cntnr.create_timeseries(
            name = 'trial end',
            data = vr_data['teleport'].to_numpy(),
            unit = 'arbitrary',
            timestamps = time_stamps,
            description = "Boolean. Trial end/teleport time",
            comments = "Timestamps do not correspond to 2P imaging time. \n \
                Cannot be used for alignment to 2P Data"
        )
        
        # LR
        ts_cntnr.create_timeseries(
            name = 'left or right',
            data = vr_data['LR'].to_numpy(),
            unit = 'arbitrary',
            timestamps = time_stamps,
            description = "-1 = left trial. 1 = right trial",
            comments = "Timestamps do not correspond to 2P imaging time. \n \
                Cannot be used for alignment to 2P Data"
        )
        
        # manrewards
        ts_cntnr.create_timeseries(
            name = 'manual rewards',
            data = vr_data['manrewards'].to_numpy(),
            unit = 'arbitrary',
            timestamps = time_stamps,
            description = "Boolean. Manually delivered reward, typically for solenoid failure or to unclog line",
            comments = "Timestamps do not correspond to 2P imaging time. \n \
                Cannot be used for alignment to 2P Data"
        )
        
        # vr_timeseries 
        # speed
        ts_cntnr.create_timeseries(
            name = 'speed',
            data = vr_timeseries['speed'].ravel(),
            unit = '10 cm/s',
            timestamps = time_stamps,
            description = "Speed along Y maze",
            comments = "Timestamps do not correspond to 2P imaging time. \n \
                Cannot be used for alignment to 2P Data"
        )
        
        # block
        ts_cntnr.create_timeseries(
            name = 'block',
            data = vr_timeseries['block'].ravel(),
            unit = 'arbitrary',
            timestamps = time_stamps,
            description = "current block",
            comments = "Timestamps do not correspond to 2P imaging time. \n \
                Cannot be used for alignment to 2P Data"
        )
        
        # nonconsum_licks
        ts_cntnr.create_timeseries(
            name = 'non-consummatory licks',
            data = vr_timeseries['nonconsum_licks'].ravel(),
            unit = 'arbitrary',
            timestamps = time_stamps,
            description = "Licks outside of reward consumption. Note this may contain artifacts from periods when \n \
                there is excess liquid on the capacitive sensor",
            comments = "Timestamps do not correspond to 2P imaging time. \n \
                Cannot be used for alignment to 2P Data"
        )
        
        ts_cntnr.create_timeseries(
            name = 'consummatory licks',
            data = vr_timeseries['consum_licks'].ravel(),
            unit = 'arbitrary',
            timestamps = time_stamps,
            description = "Licks during reward consumption",
            comments = "Timestamps do not correspond to 2P imaging time. \n \
                Cannot be used for alignment to 2P Data"
        )
        
        # reward
        ts_cntnr.create_timeseries(
            name = 'reward',
            data = vr_timeseries['reward'].ravel(),
            unit = 'arbitrary',
            timestamps = time_stamps,
            description = "Reward delivery times.",
            comments = "Timestamps do not correspond to 2P imaging time. \n \
                Cannot be used for alignment to 2P Data"
        )
        
        
        self.behavior_module.add(ts_cntnr)
        
    def _single_channel_add_vr_data(self, chan):
        ts_cntnr = BehavioralTimeSeries(name = f'2P-aligned behavior {chan}')
        
        vr_timeseries = self.sess.timeseries
        if chan == 'channel_0':
            vr_data = self.sess.vr_data_chan0
        elif chan == 'channel_1':
            vr_data = self.sess.vr_data_chan1
            
        tstarts, tstops = self.trial_starts[chan], self.trial_ends[chan]
        vr_data.loc[tstarts,'tstart']=1
        vr_data.loc[tstops, 'teleport']=1
            
        time_stamps = vr_data['time'].to_numpy()
        rate = self.sess.s2p_ops[chan]['fs']/2
        
        block = np.nan*np.zeros_like(time_stamps)
        for i, (start, stop) in enumerate(zip(tstarts[chan], tstops[chan])):
            block[start:stop] = self.sess.trial_info['block_number']
        
        # vr_data info
        
        # trial num 
        ts_cntnr.create_timeseries(
            name = 'trial number',
            data = vr_data['trialnum'].to_numpy(),
            unit = 'arbitrary',
            description = 'current trial number',
            # timestamps = time_stamps,
            rate=rate,
            starting_time=time_stamps[0],
        )
        
        # t - spline position
        ts_cntnr.create_timeseries(
            name = 'position',
            data = vr_data['t'].to_numpy(),
            unit = '10 cm',
            # timestamps = time_stamps,
            rate=rate,
            starting_time=time_stamps[0],
            description = "Position along spline trajectory. \n \
                    Track starts at a value of 13 and ends at 43. \n \
                    Points less than 13 correspond to when the mouse is \
                    in the grey hallway prior to trial start ",
        )
        
        # posx 
        ts_cntnr.create_timeseries(
            name = 'x position',
            data = vr_data['posx'].to_numpy(),
            unit = 'arbitrary',
            # timestamps = time_stamps,
            rate=rate,
            starting_time=time_stamps[0],
            description = "Unity units x position on 2D plane",
            
        )
        
        # posz
        ts_cntnr.create_timeseries(
            name = 'y position',
            data = vr_data['posz'].to_numpy(),
            unit = 'arbitrary',
            # timestamps = time_stamps,
            rate=rate,
            starting_time=time_stamps[0],
            description = "Unity units y position on 2D plane",
        )
        
        # tstart
        ts_cntnr.create_timeseries(
            name = 'trial start',
            data = vr_data['tstart'].to_numpy(),
            unit = 'arbitrary',
            # timestamps = time_stamps,
            rate=rate,
            starting_time=time_stamps[0],
            description = "Boolean. Trial start time",
        )
        # teleport
        ts_cntnr.create_timeseries(
            name = 'trial end',
            data = vr_data['teleport'].to_numpy(),
            unit = 'arbitrary',
            # timestamps = time_stamps,
            rate=rate,
            starting_time=time_stamps[0],
            description = "Boolean. Trial end/teleport time",
        )
        
        # LR
        ts_cntnr.create_timeseries(
            name = 'left or right',
            data = vr_data['LR'].to_numpy(),
            unit = 'arbitrary',
            # timestamps = time_stamps,
            rate=rate,
            starting_time=time_stamps[0],
            description = "-1 = left trial. 1 = right trial",
        )
        
        # block
        ts_cntnr.create_timeseries(
            name = 'block',
            data = block.ravel(),
            unit = 'arbitrary',
            # timestamps = time_stamps,
            rate=rate,
            starting_time=time_stamps[0],
            description = "current block",
        )
        
        self.behavior_module.add(ts_cntnr)

    def add_vr_data_aligned(self):
        for chan in ('channel_0', 'channel_1'):
            self._single_channel_add_vr_data()
        
        
        
    def init_2p_data(self):
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

        imaging_plane_channel0 = self.nwb_file.create_imaging_plane(
            name="ImagingPlaneChannel0",
            optical_channel=optical_channel0,
            indicator='channel 0: GCaMP6f',
            imaging_rate=self.sess.s2p_ops['channel_0']["fs"]/2,
            description="CA1 pyramidal cell layer",
            device=device,
            excitation_lambda=self.metadata.get('imaging_lambda')[0],
            location="CA1",
            grid_spacing=([1000/512., 1000/796.]),
            grid_spacing_unit="microns",
        )
        
        imaging_plane_channel1 = self.nwb_file.create_imaging_plane(
            name="ImagingPlaneChannel1",
            optical_channel=optical_channel1,
            indicator='channel 1: sRGECO',
            imaging_rate=self.sess.s2p_ops['channel_1']["fs"]/2,
            description="CA1 pyramidal cell layer",
            device=device,
            excitation_lambda=self.metadata.get('imaging_lambda')[1],
            location="CA1",
            grid_spacing=([1000/512., 1000/796.]),
            grid_spacing_unit="microns",
        )
        
        
        img_seg_channel0 = ImageSegmentation()
        ps_channel0 = img_seg_channel0.create_plane_segmentation(
            name="PlaneSegmentationChannel0",
            description="Suite2P output",
            imaging_plane=imaging_plane_channel0,
        )
        
        img_seg_channel1 = ImageSegmentation()
        ps_channel1 = img_seg_channel0.create_plane_segmentation(
            name="PlaneSegmentationChannel1",
            description="Suite2P output",
            imaging_plane=imaging_plane_channel1,
        )
        
        self.ophys_module = self.nwb_file.create_processing_module(
            name="ophys", description="2P imaging data"
        )
        self.ophys_module.add(img_seg_channel0)
        self.ophys_module.add(img_seg_channel1)
        
        # 
        stat = self.sess.s2p_stats['channel_0']
        for n in range(len(stat)):
            pixel_mask = np.array(
                [stat[n]['ypix'], stat[n]['xpix'], stat[n]['lam']]
            )
            ps_channel0.add_roi(pixel_mask=pixel_mask.T)
            
        self.roi_table_channel0 = ps_channel0.create_roi_table_region(
            region = list(np.arange(len(stat))),
            description="GCaMP ROIs"
        )
        
        stat = self.sess.s2p_stats['channel_1']
        for n in range(len(stat)):
            pixel_mask = np.array(
                [stat[n]['ypix'], stat[n]['xpix'], stat[n]['lam']]
            )
            ps_channel1.add_roi(pixel_mask=pixel_mask.T)
            
        self.roi_table_channel1 = ps_channel1.create_roi_table_region(
            region = list(np.arange(len(stat))),
            description="sRGECO ROIs"
        )
        
        ####
        
        
        images = Images("Backgrounds", description='motion aligned average images')
        images.add_image(GrayscaleImage(name='meanImg_ch0', 
                                        data=self.sess.s2p_ops['channel_0']['meanImg'],
                                        description='average channel 0 (gcamp) image'))
        images.add_image(GrayscaleImage(name='meanImg_ch1', 
                                        data=self.sess.s2p_ops['channel_1']['meanImg'],
                                        description='average channel 1 (sRGECO) image'))
        self.ophys_module.add(images)


    def add_cell_timeseries(self):
        if self.ophys_module is None:
            self.init_2p_data()
            
        # F_ch0, Fneu_ch0, F_dff_ch0, spks_ch0
        # F_ch1, Fneu_ch1, F_dff_ch1
            
        F_ch0 = self.sess.timeseries.get('F')
        roi_resp_series = RoiResponseSeries(
            name = 'fluorescence',
            data = F.T,
            rois = self.roi_table,
            unit = 'arbitrary',
            rate = self.sess.s2p_ops['fs'],
            description = 'raw fluorescence from channel 0 (gcamp)',
        )
        fl = Fluorescence(roi_response_series=roi_resp_series, name='fluorescence')
        self.ophys_module.add(fl)
        
        Fneu = self.sess.timeseries.get('Fneu')
        roi_resp_series = RoiResponseSeries(
            name = 'neuropil fluorescence',
            data = Fneu.T,
            rois = self.roi_table,
            unit = 'arbitrary',
            rate = self.sess.s2p_ops['fs'],
            description = 'raw neuropil fluorescence from channel 0 (gcamp)',
        )
        fl = Fluorescence(roi_response_series=roi_resp_series, name='neuropil')
        self.ophys_module.add(fl)
        
        dff = self.sess.timeseries.get('F_dff')
        roi_resp_series = RoiResponseSeries(
            name = 'dF',
            data = dff.T,
            rois = self.roi_table,
            unit = 'arbitrary',
            rate = self.sess.s2p_ops['fs'],
            description = 'dF/F from channel 0 (gcamp)',
        )
        fl = DfOverF(roi_response_series=roi_resp_series, name='dF')
        self.ophys_module.add(fl)
        
        
    def build_file(self):
        self.init_nwb_file()
        self.add_vr_data_full_res()
        self.add_vr_data_aligned()
        self.init_2p_data()
        self.add_cell_timeseries()
        return self
        
    def write_file(self):
        with NWBHDF5IO(self.out_path, "w") as fio:
            fio.write(self.nwb_file)
            
    def remove_sbx_data(self):
        self.sbx_mat_path.unlink(missing_ok=True)
        self.sbx_path.unlink(missing_ok=True)
        
    

    



# class RawDataNWB_Dense:
    
#     def __init__(self, mouse, metadata, session, day, oak_pwd, scan=1, sub_notes=''):

#         self.mouse = mouse
#         self.session = session
#         self.metadata = metadata
#         self._oak_pwd = oak_pwd
#         self.day = day
#         self.sub_notes = sub_notes

#         self.sess_dir = SCRATCH_DIR / mouse / session.get('date_str') / session.get('scene')
#         self.sess_dir.mkdir(parents=True, exist_ok=True)

#         self.s2p_path = self.sess_dir /  \
#             f"{self.session.get('scene')}_{self.session.get('session'):03}_{self.session.get('scan'):03}" / \
#             "suite2p" / "plane0"
#         self.s2p_ops = None
        
#         self.vr_path = VR_DIR / mouse / session.get('date_str') / f"{session.get('scene')}_{session.get('session')}.sqlite"
#         self.vr_data = None
#         self.sbx_mat = None
        
#         self.nwb_file = None

#         self.out_path = OUTPATH / mouse / f"ymaze_day{day}_scan{scan}_ophys_behav_RAW.nwb"
#         self.out_path.parent.mkdir(parents=True, exist_ok=True)

#     def run_rsync(self):
#         remote_user = "mplitt"
#         remote_host = "dtn.sherlock.stanford.edu"
#         remote_base_path = "/oak/stanford/groups/giocomo/mplitt/2P_Data/STX3KO/"

#         cmd = [
#             "sshpass", "-p", self._oak_pwd,
#             "rsync", "-rlt", "--progress", 
#             f"{remote_user}@{remote_host}:{remote_base_path}{self.mouse}/{self.session.get('date_str')}/{self.session.get('scene')}",
#             str(self.sess_dir.parent)
#         ]
        
#         try:
#             subprocess.run(cmd, check=True)
#             print("Rsync completed successfully.")
#         except subprocess.CalledProcessError as e:
#             print(f"Rsync failed with error: {e}")


#     def init_nwb_file(self):
#         self.nwb_file = NWBFile(
#             session_description = "Raw 2P Data and VR Data",
#             session_start_time = datetime.datetime.now().astimezone(),
#             identifier=str(uuid4()),  # required
#         )

#         self.nwb_file.subject = Subject(
#             subject_id = self.metadata.get('alias'),
#             age = self.session.get('datetime') - self.metadata.get('date_of_birth'),
#             description = f"YMaze day {self.day}." + self.sub_notes,
#             species = 'Mus musculus',
#             sex = self.metadata.get('sex'),
#             genotype = self.session.get('genotype'),
#         )

#         # add subject metadata
    
#     def add_binary_2Pdata(self):

#         self.s2p_ops = np.load(self.s2p_path / "ops.npy", allow_pickle=True).item()

#         device = self.nwb_file.create_device(
#         name="Microscope",
#         description="Giocomo lab Neurolabware 2P Scope",
#         manufacturer="Neurolabware",
#         )
        
#         optical_channel0 = OpticalChannel(
#                 name="Green PMT",
#                 description="an optical channel",
#                 emission_lambda=500.0,
#         )
#         optical_channel1 = OpticalChannel(
#                 name="Red PMT",
#                 description="an optical channel",
#                 emission_lambda=600.0,
#         )

#         imaging_plane0 = self.nwb_file.create_imaging_plane(
#             name="ImagingPlane_ch0",
#             optical_channel=optical_channel0,
#             imaging_rate=self.s2p_ops["fs"],
#             description="standard",
#             device=device,
#             excitation_lambda=self.metadata.get('imaging_lambda'),
#             indicator=self.metadata.get('functional_indicator'),
#             location="CA1",
#             grid_spacing=([1000/512., 1000/796.]),
#             grid_spacing_unit="microns",
#         )

#         binary_file_ch0 = self.s2p_path / "data.bin"
#         binfile_ch0 = suite2p.io.BinaryFile(self.s2p_ops['Ly'], self.s2p_ops['Lx'], str(binary_file_ch0))
#         image_series_ch0 = TwoPhotonSeries(
#                 name="TwoPhotonSeries_ch0",
#                 dimension=[self.s2p_ops["Ly"], self.s2p_ops["Lx"]],
#                 data=binfile_ch0.data,
#                 imaging_plane=imaging_plane0,
#                 starting_time=0.0,
#                 rate=self.s2p_ops["fs"],
#                 unit="n.a.",
#             )
#         self.nwb_file.add_acquisition(image_series_ch0)

#         imaging_plane1 = self.nwb_file.create_imaging_plane(
#                 name="ImagingPlane_ch1",
#                 optical_channel=optical_channel1,
#                 imaging_rate=self.s2p_ops["fs"],
#                 description="standard",
#                 device=device,
#                 excitation_lambda=self.metadata.get('imaging_lambda'),
#                 indicator=self.metadata.get('static_indicator'),
#                 location="CA1",
#                 grid_spacing=([2.0, 2.0]),
#                 grid_spacing_unit="microns",
#             )

#         binary_file1 = self.s2p_path / "data_chan2.bin"
#         binfile1 = suite2p.io.BinaryFile(self.s2p_ops['Ly'], self.s2p_ops['Lx'], str(binary_file1))
#         image_series_ch1 = TwoPhotonSeries(
#                 name="TwoPhotonSeries_ch1",
#                 dimension=[self.s2p_ops["Ly"], self.s2p_ops["Lx"]],
#                 data=binfile_ch0.data,
#                 imaging_plane=imaging_plane1,
#                 format="external",
#                 starting_time=0.0,
#                 rate=self.s2p_ops["fs"] ,
#                 unit="n.a.",
#             )
#         self.nwb_file.add_acquisition(image_series_ch1)

#     def load_sbx_mat(self):


#         self.sbx_mat = tpu.scanner_tools.sbx_utils.loadmat(str(self.sess_dir / \
#                     f"{self.session.get('scene')}_{self.session.get('session'):03}_{self.session.get('scan'):03}.mat"))
        
#     def get_ttl_times(self):
#         sbx_mat = self.sbx_mat

#         fr = sbx_mat['frame_rate'] # frame rate
#         lr = fr * sbx_mat['config']['lines']/sbx_mat['fov_repeats']  # line rate

#         frames = sbx_mat['frame'].astype(int)
#         frame_diff = np.ediff1d(frames, to_begin=0)
#         try:
#             mods = np.argwhere(frame_diff < -100)[0]
#             for i, mod in enumerate(mods.tolist()):
#                 frames[mod:] += (i + 1) * 65535
#         except:
#             pass
        
#         frames = frames * sbx_mat['fov_repeats']
#         if sbx_mat['fold_lines']>0:
#             lines = np.array([l % sbx_mat['fold_lines'] for l in sbx_mat['line']])
#         else:
#             lines = np.array(sbx_mat['line'])

#         ttl_times = frames / fr + lines / lr
#         return ttl_times
        
#     def load_vr_data(self):
    

#         vr_df = tpu.preprocessing.load_sqlite(self.vr_path)
#         ttl_times = self.get_ttl_times()
        
#         self.vr_data = vr_df.iloc[-ttl_times.shape[0]:]
#         self.vr_data['2P time'] = ttl_times

    
#     def add_behav_timeseries(self):
        
#         behav_module = self.nwb_file.create_processing_module('VR behavior', 'raw behavior timeseries')
#         behav_ts_container = BehavioralTimeSeries(name = 'VR behavior')

#         timestamps = self.vr_data['2P time'].to_numpy()

#         behav_ts_container.create_timeseries(
#             name = 'trial number',
#             data = self.vr_data['trialnum'].to_numpy(),
#             unit = 'arbitrary',
#             description = "trial number",
#             timestamps = timestamps,
#         )

#         behav_ts_container.create_timeseries(
#             name = 'position',
#             data = self.vr_data['t'].to_numpy(),
#             timestamps = timestamps,
#             unit = '10 cm',
#             description = "Position along spline trajectory. \n \
#                     Track starts at a value of 13 and ends at 43. \n \
#                     Points less than 13 correspond to when the mouse is \
#                         in the grey hallway prior to trial start ",
#         )

#         behav_ts_container.create_timeseries(
#             name = 'x position',
#             data = self.vr_data['posx'].to_numpy(),
#             timestamps = timestamps,
#             unit = 'arbitrary',
#             description = "Unity units x position on 2D plane",
#         )

#         behav_ts_container.create_timeseries(
#             name = 'y position',
#             data = self.vr_data['posz'].to_numpy(),
#             timestamps = timestamps,
#             unit = "arbitrary",
#             description = "Unity units y position on 2D plane",
#         )

#         behav_ts_container.create_timeseries(
#             name = 'licks',
#             data = self.vr_data['lick'].to_numpy(),
#             timestamps = timestamps,
#             unit = "arbitrary",
#             description = "Boolean corresponding to whether capacitve touch sensor on lick port detects a touch. \n \
#                 Single licks correspond to rising edges",
#             comments = "These are the raw touch values. Processing is needed to isolate single licks. \n \
#                 Occasionally, there are artifacts where the pin value is stuck high from excess moisture on the lickport."
#         )

#         behav_ts_container.create_timeseries(
#             name = 'reward',
#             data = self.vr_data['reward'].to_numpy(),
#             timestamps = timestamps,
#             unit = "arbitrary",
#             description = "Boolean corresponding to reward delivery. Only rising edges are informative ",
#         )

#         behav_ts_container.create_timeseries(
#             name = 'manual rewards',
#             data = self.vr_data['manrewards'].to_numpy(),
#             timestamps = timestamps,
#             unit = "arbitrary",
#             description = "Boolean corresponding to manual reward delivery. Only rising edges are informative ",
#         )

#         behav_ts_container.create_timeseries(
#             name = 'trial start',
#             data = self.vr_data['tstart'].to_numpy(),
#             timestamps = timestamps,
#             unit = "arbitrary",
#             description = "Boolean corresponding to trial starts",
#         )

#         behav_ts_container.create_timeseries(
#             name = 'trial end',
#             data = self.vr_data['teleport'].to_numpy(),
#             timestamps = timestamps,
#             unit = "arbitrary",
#             description = "Boolean corresponding to trial ends",
#         )

#         behav_ts_container.create_timeseries(
#             name = 'left_right',
#             data = self.vr_data['LR'].to_numpy(),
#             timestamps = timestamps,
#             unit = "arbitrary",
#             description = "-1 = left trial, 1 = right trial",
#         )
        
#         behav_module.add(behav_ts_container)

#     def build_file(self):
#         self.init_nwb_file()
#         self.add_binary_2Pdata()
#         self.load_sbx_mat()
#         self.load_vr_data()
#         self.add_behav_timeseries()


#     def write_nwb(self):
#         with NWBHDF5IO(self.out_path, "w") as fio:
#             fio.write(self.nwb_file)






    








                


