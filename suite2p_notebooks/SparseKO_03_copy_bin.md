---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.4
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
import numpy as np
import scipy as sp
import suite2p as s2p
import TwoPUtils as tpu
import InVivoDA_analyses as da
import os

%matplotlib inline

%load_ext autoreload
%autoreload 2
```

```python
from InVivoDA_analyses.path_dict_firebird import path_dictionary as path_dict
# options: path_dict_josquin, path_dict_msosamac, etc.
path_dict
```

```python
mouse = "SparseKO_03"
basedir = os.path.join(path_dict['preprocessed_root'],mouse) #"/mnt/BigDisk/2P_scratch/GRABDA15"
sbxdir = os.path.join(path_dict['sbx_root'],mouse) 

custom_nplanes = 1 # if needed to override empty optotune params

basedir, sbxdir
```

```python

# file_list = stx3.ymaze_sess_deets.SparseKO_sessions[mouse]
# file_list = [ {'date': '22_11_2024', 'scene': 'YMaze_LNovel_LongTimeout', 'session': 1, 'scan': 19, 'exp_day': 7} ]
file_list = [ {'date': '16_11_2024', 'scene': 'YMaze_LNovel', 'session': 1, 'scan': 17, 'exp_day': 1} ]

file_list
```

```python
for fn,f in enumerate(file_list):

    # Set data input and output paths
    fullpath = os.path.join(basedir,f['date'],f['scene'],"%s_%03d_%03d" % (f['scene'], f['session'], f['scan']))
    scanpath = os.path.join(sbxdir,f['date'],f['scene'],"%s_%03d_%03d" % (f['scene'], f['session'], f['scan']))
    h5path = os.path.join(basedir,f['date'],f['scene'],"%s_%03d_%03d.h5" % (f['scene'], f['session'], f['scan']))

    # load .mat header with metadata
    scanmat, sbxfile = scanpath+'.mat', scanpath+'.sbx'
    info = tpu.scanner_tools.sbx_utils.loadmat(scanmat, sbx_version=3)
    if len(info['etl_table'])>0:
        nplanes = info['etl_table'].shape[0]
    else:
        nplanes = custom_nplanes
        print('"etl_table" was empty; hardcoding %d planes' % nplanes)

    print('nplanes=',nplanes)

    # convert .sbx file to .h5
    h5name = tpu.scanner_tools.sbx_utils.sbx2h5(scanpath,output_name=h5path, sbx_version=3)
    
    # run suite2p to run motion registration AND extract ROIs on the first
    # functional channel (channel 1, green, PMT0)
    ops_orig = tpu.s2p.set_ops(d={'data_path': [os.path.split(fullpath)[0]],
                           'save_path0': fullpath,
                           'fast_disk':[],
                           'move_bin':False,
                           'two_step_registration':True,
                           'maxregshiftNR':10,
                           'nchannels':2,
                           'tau': 0.7,
                           'functional_chan':1,
                           'nimg_init': 2000,
                           'fs':info['frame_rate'],
                           'roidetect':True,
                           'input_format':"h5",
                           'h5py_key':'data',
                           'sparse_mode':True,
                           'threshold_scaling':.6, #.6
                            'nplanes':nplanes})

    ops_orig=s2p.run_s2p(ops=ops_orig)

    # !rm {h5name} 
```

## Copy binary files so you don't have to re-register

```python
## Use the existing motion-registered binary and
# extract ROIs for functional channel 2 (red, PMT1)

ops = ops_orig.copy()
save_path_orig = ops_orig['save_path0']

# Make new save path, destination for new ops, stat, and copied binaries
ops['save_path'] = os.path.join(ops['save_path0'], 'chan2','suite2p', 'plane0')
os.makedirs(ops['save_path'], exist_ok=True)

```

```python
import shutil

ops['fast_disk'] = [] # remove this so there's no confusion with save_path

orig_reg_file_1 = os.path.join(save_path_orig,'suite2p','plane0','data.bin') # the original data.nbin
orig_reg_file_2 = os.path.join(save_path_orig,'suite2p','plane0','data_chan2.bin') # the original data_chan2.nbin

# switch meanImg <-> meanImg_chan2
mimg = np.copy(ops_orig['meanImg'])
ops['meanImg'] = np.copy(ops_orig['meanImg_chan2'])
ops['meanImg_chan2'] = mimg

# copy chan2 reg file (original data_chan2.bin) to new location as "data.bin"
ops['reg_file'] = os.path.join(ops['save_path'], 'data.bin') # create a path for the new reg_file
shutil.copyfile(orig_reg_file_2, ops['reg_file'])
print('red reg_file_chan2 copied to %s'%(ops['reg_file']))

# now copy original chan1 reg file to new "data_chan2.bin"
ops['reg_file_chan2'] = os.path.join(ops['save_path'], 'data_chan2.bin') # create a path for the new chan2 reg_file
shutil.copyfile(orig_reg_file_1, ops['reg_file_chan2'])
print('green reg_file copied to %s'%(ops['reg_file_chan2']))

```

```python
# save new ops file with paths
ops['ops_path'] = os.path.join(ops['save_path'],'ops.npy')
ops['save_path0'] = os.path.join(ops['save_path0'], 'chan2') # need it to look here + suite2p/plane0/ for binaries that we copied

np.save(ops['ops_path'], ops)
```

```python
## Optional: update ROI threshold for the second channel
ops['threshold_scaling'] = .85
```

```python
import matplotlib.pyplot as plt
```

```python
plt.figure()
plt.imshow(ops['meanImg_chan2'])

plt.figure()
plt.imshow(ops['meanImg'])
```

```python tags=[]
## Run the ROI extraction for the second channel
opsEnd=s2p.run_s2p(ops=ops)
```

### You can now open both channels in suite2p

```python
## OPTIONAL delete copied binaries from the chan2 path to save disk space
!rm {ops['reg_file']}
!rm {ops['reg_file_chan2']} 
```

```python
## Remove enormous h5 file
!rm {h5name} 
```
