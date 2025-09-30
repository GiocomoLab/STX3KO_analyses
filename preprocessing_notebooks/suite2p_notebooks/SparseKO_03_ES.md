---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.4
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python
import numpy as np
import scipy as sp
import suite2p as s2p
import TwoPUtils as tpu
import InVivoDA_analyses as da
import STX3KO_analyses as stx
import os

%matplotlib inline

%load_ext autoreload
%autoreload 2
# %reload_ext autoreload
```

```python
# from InVivoDA_analyses.path_dict_esay import path_dictionary as path_dict
from STX3KO_analyses.path_dicts.path_dict_esay import path_dictionary as path_dict

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
file_list = stx.ymaze_sess_deets.SparseKO_sessions[mouse]

file_list = [file_list[3:6]]
file_list
```

```python
for fn,f in enumerate(file_list):
    # print(f)
    f = f[fn]

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
    h5name = tpu.scanner_tools.sbx_utils.sbx2h5(scanpath,output_name=h5path, sbx_version=3) #, force_2chan=True)

    # run suite2p to run motion registration AND extract ROIs on the first
    # functional channel (channel 1, green, PMT0)
    ops = tpu.s2p.set_ops(d={'data_path': [os.path.split(fullpath)[0]],
                           'save_path0': fullpath,
                           'fast_disk':[],
                           'move_bin':True,
                           'two_step_registration':True,
                           'maxregshiftNR':10,
                           'nchannels':2,
                           'tau': 0.7,
                           'functional_chan':1,
                            'align_by_chan' : 1,
                           'nimg_init': 2000,
                           'fs':info['frame_rate'],
                           'roidetect':True,
                           'input_format':"h5",
                           'h5py_key':'data',
                           'sparse_mode':True,
                           'threshold_scaling':.8, #.6
                            'nplanes':nplanes})
    ops=s2p.run_s2p(ops=ops)
```

```python
## Use the existing motion-registered binary and
# extract ROIs for functional channel 2 (red, PMT1)
ops_orig = ops.copy()
```

```python
ops = ops_orig.copy()
save_path_orig = ops_orig['save_path0']
ops['save_path0'] = os.path.join(save_path_orig,'chan2')
ops['save_path'] = os.path.join(ops['save_path0'], 'suite2p', 'plane0')
os.makedirs(ops['save_path'], exist_ok=True)

```

```python
ops['functional_chan'] = 2
ops['align_by_chan'] = 1
# get rid of refImg so it re-runs registration in the new channel
del ops['refImg']
ops['do_registration'] = 1
```

```python
## currently this will extract and re-register a new binary from the same .h5 file (which is redundant)
opsEnd=s2p.run_s2p(ops=ops)
```

```python

!rm {h5name} 
```

## Below here was our attempt to use Carsen Stringer's method of copying the reg_files

```python
### SETUP FOR RUNNING ONLY CELL DETECTION ON SECOND CHANNEL
import shutil

# grab fast_disk variable if it is user-specified
if 'fast_disk' in ops_orig and len(ops_orig['fast_disk'])>0:
    fast_disk = ops_orig['fast_disk']
else:
    fast_disk = []
# grab save_path0
save_path_orig = ops_orig['save_path0']
# make a new folder for channel 2 save files and binary files
ops1 = []
j=0
# for ops in opsEnd:
ops = ops_orig.copy()
# make a chan2 folder inside the save path
ops['save_path0'] = os.path.join(save_path_orig,'chan2')
ops['save_path'] = os.path.join(ops['save_path0'], 'suite2p', 'plane%d'%j)
os.makedirs(ops['save_path'], exist_ok=True)
# ops['ops_path'] = os.path.join(ops['save_path'],'ops.npy')
# put binary file there or in user-specified + chan2
if len(fast_disk)==0:
    fast_disk0 = os.path.join(ops['save_path0'])
else:
    fast_disk0 = os.path.join(fast_disk,'chan2')
ops['fast_disk'] = os.path.join(fast_disk0, 'suite2p', 'plane%d'%j)
os.makedirs(ops['fast_disk'], exist_ok=True)
```

```python
ops['reg_file'] = os.path.join(ops['fast_disk'], 'data.bin') # create a path for the new reg_file
orig_reg_file = os.path.join(save_path_orig,'suite2p','plane0','data.bin') # the original data.nbin

# switch meanImg <-> meanImg_chan2
mimg = np.copy(ops_orig['meanImg'])
ops['meanImg'] = np.copy(ops_orig['meanImg_chan2'])
ops['meanImg_chan2'] = mimg
# copy chan2 reg file (doriginal data_chan2.bin) to new location
shutil.copyfile(ops['reg_file_chan2'], ops['reg_file'])
print('reg_file_chan2 copied to %s'%(ops['reg_file']))

# now copy original chan1 data to new "chan2"
ops['reg_file_chan2'] = os.path.join(ops['fast_disk'], 'data_chan2.bin') # create a path for the new chan2 reg_file
shutil.copyfile(orig_reg_file, ops['reg_file_chan2'])
print('reg_file copied to %s'%(ops['reg_file_chan2']))

# save new ops file with paths
ops['ops_path'] = os.path.join(ops['fast_disk'],'ops.npy')
np.save(ops['ops_path'], ops)
ops1.append(ops.copy())
j+=1
# save ops across planes in new location
# np.save(os.path.join(ops1[0]['save_path0'],'suite2p','ops1.npy'), ops1)
```

```python
ops['fast_disk']
```

```python
ops['save_path']
```

```python
import matplotlib.pyplot as plt
```

```python
plt.figure()
plt.imshow(ops_orig['meanImg_chan2'])

plt.figure()
plt.imshow(ops['meanImg'])
```

```python
# run pipeline on second channel 
# (will skip registration b/c binary file exists)
ops_to_run = ops1[0].copy()
ops_to_run['functional_chan'] = 1
del ops_to_run['save_path']
del ops_to_run['ops_path']
# del ops0['reg_file']
# del ops0['reg_file_chan2']

# we want to get the data on chan1 for the chan2 ROIs as well so we can measure crosstalk?

# opsEnd=run_s2p(ops=ops0,db=db)
opsEnd=s2p.run_s2p(ops=ops_to_run)
```

```python
scanpath
```

### Plotting each FOV with the "dead columns" from the bidirectional scanning cut off

```python
x = tpu.scanner_tools.sbx_utils.sbxread(scanpath, N=100, sbx_version=3)
```

```python
x[:,:,:,::2].shape
```

```python
ndeadcols = tpu.scanner_tools.sbx_utils.find_deadbands(scanpath)
ndeadcols
```

```python
np.arange(0,11)[::2], np.arange(0,11)[1::2]
```

```python
plt.figure()
data_chan1 = x[0,ndeadcols:,:,::2]
vmax = np.percentile(np.nanmean(data_chan1,axis=2).ravel(), 99)
print(vmax)
plt.imshow(np.nanmean(data_chan1,axis=2), cmap='Greys_r', vmax = vmax)
```

```python
plt.figure()
data_chan2 = x[1,ndeadcols:,:,1::2]
vmax = np.percentile(np.nanmean(data_chan2,axis=2).ravel(), 99)
print(vmax)
plt.imshow(np.nanmean(data_chan2,axis=2), cmap='Greys_r', vmax = vmax)
```

```python

```
