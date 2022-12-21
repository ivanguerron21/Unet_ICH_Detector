# This code loads the CT slices (grayscale images) of the brain-window for each subject in ct_scans folder then saves them to
# one folder (data\image).
# Their segmentation from the masks folder is saved to another folder (data\label).
# it was modified from the original script found on physionet

import os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import nibabel as nib
from matplotlib.pyplot import imsave


def window_ct (ct_scan, w_level=40, w_width=120):
    w_min = w_level - w_width / 2
    w_max = w_level + w_width / 2
    num_slices=ct_scan.shape[2]
    for s in range(num_slices):
        slice_s = ct_scan[:,:,s]
        slice_s = (slice_s - w_min)*(255/(w_max-w_min)) #or slice_s = (slice_s - (w_level-(w_width/2)))*(255/(w_width))
        slice_s[slice_s < 0]=0
        slice_s[slice_s > 255] = 255
        #slice_s=np.rot90(slice_s)
        ct_scan[:,:,s] = slice_s

    return ct_scan


def do_split():
    numSubj = 82
    new_size = (256, 256)
    window_specs=[40,120] #Brain window
    currentDir = Path(os.getcwd())
    datasetDir = str(Path(currentDir))

    # Reading labels
    hemorrhage_diagnosis_df = pd.read_csv(
        Path(currentDir, 'hemorrhage_diagnosis_raw_ct.csv'))
    hemorrhage_diagnosis_array = hemorrhage_diagnosis_df.values

    # reading images
    train_path = Path('data')
    image_path = train_path / 'image'
    label_path = train_path / 'label'
    if not train_path.exists():
        train_path.mkdir()
        image_path.mkdir()
        label_path.mkdir()

    counterI = 1
    for sNo in range(0+49, numSubj+49):
        if sNo>58 and sNo<66: #no raw data were available for these subjects
            next
        else:
            #Loading the CT scan
            ct_dir_subj = Path(datasetDir,'ct_scans', "{0:0=3d}.nii".format(sNo))
            ct_scan_nifti = nib.load(str(ct_dir_subj))
            ct_scan = ct_scan_nifti.get_data()
            ct_scan = window_ct(ct_scan, window_specs[0], window_specs[1])

            #Loading the masks
            masks_dir_subj = Path(datasetDir,'masks', "{0:0=3d}.nii".format(sNo))
            masks_nifti = nib.load(str(masks_dir_subj))
            masks = masks_nifti.get_data()
            idx = hemorrhage_diagnosis_array[:, 0] == sNo
            sliceNos = hemorrhage_diagnosis_array[idx, 1]
            NoHemorrhage = hemorrhage_diagnosis_array[idx, 7]
            if sliceNos.size!=ct_scan.shape[2]:
                print('Warning: the number of annotated slices does not equal the number of slices in NIFTI file!')

            for sliceI in range(0, sliceNos.size):
                # Saving the a given CT slice

                x = np.array(Image.fromarray(ct_scan[:,:,sliceI]).resize(new_size))
                imsave(image_path / (str(counterI) + '.png'), x)

                # Saving the segmentation for a given slice
                x = np.array(Image.fromarray(masks[:,:,sliceI]).resize(new_size))
                imsave(label_path / (str(counterI) + '.png'), x)
                counterI = counterI+1
