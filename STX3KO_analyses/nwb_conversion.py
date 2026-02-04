import datetime
import gc
import os
import time
from pathlib import Path

import numpy as np
import scipy
from natsort import natsorted
import pynwb
# from .. import run_s2p
# from ..detection.stats import roi_stats
# from . import utils
# from .. import run_s2p, default_ops


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

def save_nwb(save_folder):
    """convert folder with plane folders to NWB format"""

    plane_folders = natsorted([
        Path(f.path)
        for f in os.scandir(save_folder)
        if f.is_dir() and f.name[:5] == "plane"
    ])
    ops1 = [
        np.load(f.joinpath("ops.npy"), allow_pickle=True).item() for f in plane_folders
    ]
    nchannels = min([ops["nchannels"] for ops in ops1])

    
    if len(ops1) > 1:
        multiplane = True
    else:
        multiplane = False

    ops = ops1[0]
    if "date_proc" in ops:
        session_start_time = ops["date_proc"]
        if not session_start_time.tzinfo:
            session_start_time = session_start_time.astimezone()
    else:
        session_start_time = datetime.datetime.now().astimezone()

    # INITIALIZE NWB FILE
    nwbfile = NWBFile(
        session_description="suite2p_proc",
        identifier=str(ops["data_path"][0]),
        session_start_time=session_start_time,
    )
    print(nwbfile)

    device = nwbfile.create_device(
        name="Microscope",
        description="Giocomo lab Neurolabware 2P Scope",
        manufacturer="Neurolabware",
    )
    optical_channel = OpticalChannel(
        name="OpticalChannel",
        description="an optical channel",
        emission_lambda=500.0,
    )

    
    imaging_plane = nwbfile.create_imaging_plane(
        name="ImagingPlane",
        optical_channel=optical_channel,
        imaging_rate=ops["fs"],
        description="standard",
        device=device,
        excitation_lambda=920.0,
        indicator="GCaMP7f",
        location="CA1",
        grid_spacing=([2.0, 2.0, 30.0] if multiplane else [2.0, 2.0]),
        grid_spacing_unit="microns",
    )
    # link to external data
    external_data = ops["filelist"] if "filelist" in ops else [""]
    image_series = TwoPhotonSeries(
        name="TwoPhotonSeries",
        dimension=[ops["Ly"], ops["Lx"]],
        external_file=external_data,
        imaging_plane=imaging_plane,
        starting_frame=[0 for i in range(len(external_data))],
        format="external",
        starting_time=0.0,
        rate=ops["fs"] * ops["nplanes"],
    )
    nwbfile.add_acquisition(image_series)

    # processing
    img_seg = ImageSegmentation()
    ps = img_seg.create_plane_segmentation(
        name="PlaneSegmentation",
        description="suite2p output",
        imaging_plane=imaging_plane,
        reference_images=image_series,
    )
    ophys_module = nwbfile.create_processing_module(
        name="ophys", description="optical physiology processed data")
    ophys_module.add(img_seg)

    file_strs = ["F.npy", "Fneu.npy"]
    file_strs_chan2 = ["F_chan2.npy", "Fneu_chan2.npy"]
    traces, traces_chan2 = [], []
    ncells = np.zeros(len(ops1), dtype=np.int_)
    Nfr = np.array([ops["nframes"] for ops in ops1]).max()
    for iplane, ops in enumerate(ops1):
        if iplane == 0:
            iscell = np.load(os.path.join(ops["save_path"], "iscell.npy"))
            for fstr in file_strs:
                traces.append(np.load(os.path.join(ops["save_path"], fstr)))
            if nchannels > 1:
                for fstr in file_strs_chan2:
                    traces_chan2.append(
                        np.load(plane_folders[iplane].joinpath(fstr)))
            PlaneCellsIdx = iplane * np.ones(len(iscell))
        else:
            iscell = np.append(
                iscell,
                np.load(os.path.join(ops["save_path"], "iscell.npy")),
                axis=0,
            )
            for i, fstr in enumerate(file_strs):
                trace = np.load(os.path.join(ops["save_path"], fstr))
                if trace.shape[1] < Nfr:
                    fcat = np.zeros((trace.shape[0], Nfr - trace.shape[1]),
                                    "float32")
                    trace = np.concatenate((trace, fcat), axis=1)
                traces[i] = np.append(traces[i], trace, axis=0)
            if nchannels > 1:
                for i, fstr in enumerate(file_strs_chan2):
                    traces_chan2[i] = np.append(
                        traces_chan2[i],
                        np.load(plane_folders[iplane].joinpath(fstr)),
                        axis=0,
                    )
            PlaneCellsIdx = np.append(
                PlaneCellsIdx, iplane * np.ones(len(iscell) - len(PlaneCellsIdx)))

        stat = np.load(os.path.join(ops["save_path"], "stat.npy"),
                        allow_pickle=True)
        ncells[iplane] = len(stat)
        for n in range(ncells[iplane]):
            if multiplane:
                pixel_mask = np.array([
                    stat[n]["ypix"],
                    stat[n]["xpix"],
                    iplane * np.ones(stat[n]["npix"]),
                    stat[n]["lam"],
                ])
                ps.add_roi(voxel_mask=pixel_mask.T)
            else:
                pixel_mask = np.array(
                    [stat[n]["ypix"], stat[n]["xpix"], stat[n]["lam"]])
                ps.add_roi(pixel_mask=pixel_mask.T)

    ps.add_column("iscell", "two columns - iscell & probcell", iscell)

    rt_region = []
    for iplane, ops in enumerate(ops1):
        if iplane == 0:
            rt_region.append(
                ps.create_roi_table_region(
                    region=list(np.arange(0, ncells[iplane]),),
                    description=f"ROIs for plane{int(iplane)}",
                ))
        else:
            rt_region.append(
                ps.create_roi_table_region(
                    region=list(
                        np.arange(
                            np.sum(ncells[:iplane]),
                            ncells[iplane] + np.sum(ncells[:iplane]),
                        )),
                    description=f"ROIs for plane{int(iplane)}",
                ))

    # FLUORESCENCE (all are required)
    name_strs = ["Fluorescence", "Neuropil", "Deconvolved"]
    name_strs_chan2 = ["Fluorescence_chan2", "Neuropil_chan2"]

    for i, (fstr, nstr) in enumerate(zip(file_strs, name_strs)):
        for iplane, ops in enumerate(ops1):
            roi_resp_series = RoiResponseSeries(
                name=f"plane{int(iplane)}",
                data=np.transpose(traces[i][PlaneCellsIdx == iplane]),
                rois=rt_region[iplane],
                unit="lumens",
                rate=ops["fs"],
            )
            if iplane == 0:
                fl = Fluorescence(roi_response_series=roi_resp_series, name=nstr)
            else:
                fl.add_roi_response_series(roi_response_series=roi_resp_series)
        ophys_module.add(fl)

    if nchannels > 1:
        for i, (fstr, nstr) in enumerate(zip(file_strs_chan2, name_strs_chan2)):
            for iplane, ops in enumerate(ops1):
                roi_resp_series = RoiResponseSeries(
                    name=f"plane{int(iplane)}",
                    data=np.transpose(traces_chan2[i][PlaneCellsIdx == iplane]),
                    rois=rt_region[iplane],
                    unit="lumens",
                    rate=ops["fs"],
                )

                if iplane == 0:
                    fl = Fluorescence(roi_response_series=roi_resp_series,
                                        name=nstr)
                else:
                    fl.add_roi_response_series(roi_response_series=roi_resp_series)

            ophys_module.add(fl)

    # BACKGROUNDS
    # (meanImg, Vcorr and max_proj are REQUIRED)
    bg_strs = ["meanImg", "meanImg_chan2"]
    for iplane, ops in enumerate(ops1):
        images = Images("Backgrounds_%d" % iplane)
        for bstr in bg_strs:
            if bstr in ops:
                if bstr == "Vcorr" or bstr == "max_proj":
                    img = np.zeros((ops["Ly"], ops["Lx"]), np.float32)
                    img[
                        ops["yrange"][0]:ops["yrange"][-1],
                        ops["xrange"][0]:ops["xrange"][-1],
                    ] = ops[bstr]
                else:
                    img = ops[bstr]
                images.add_image(GrayscaleImage(name=bstr, data=img))

        ophys_module.add(images)

    with NWBHDF5IO(os.path.join(save_folder, "ophys.nwb"), "w") as fio:
        fio.write(nwbfile)
    

import pandas as pd
from pynwb import NWBHDF5IO
from pynwb.behavior import BehavioralTimeSeries, TimeSeries

# 1. Setup - assume 'df' is your pandas dataframe from the sqlite file
# df = pd.read_sql_query("SELECT * FROM behavior_table", sqlite_conn)
timestamp_column = 'timestamps' # Update this to your actual column name

with NWBHDF5IO('existing_data.nwb', mode='a') as io:
    nwbfile = io.read()
    
    # 2. Get or create the behavior module
    if 'behavior' not in nwbfile.processing:
        beh_module = nwbfile.create_processing_module('behavior', 'behavioral data')
    else:
        beh_module = nwbfile.processing['behavior']

    # 3. Create a container to hold the multiple time series
    beh_ts_container = BehavioralTimeSeries(name='sqlite_behavior_data')

    # 4. Iterate through columns and add to container
    for col in df.columns:
        if col == timestamp_column:
            continue
            
        # Create TimeSeries for each behavioral metric (e.g., licks, speed)
        ts = TimeSeries(
            name=col,
            data=df[col].values,
            timestamps=df[timestamp_column].values,
            unit='arbitrary', # Change to specific units if known (e.g., 'm/s')
            description=f"Behavioral metric: {col}"
        )
        beh_ts_container.add_time_series(ts)

    # 5. Add container to the module and save
    beh_module.add(beh_ts_container)
    io.write(nwbfile)
