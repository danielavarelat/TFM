import os
import numpy as np
import h5py
import nibabel as nib
import sys

sys.path.insert(1, "/homedtic/dvarela")

import util_daniela as u


def convert_nii_rotate(fileNII, fileH5):
    """."""
    img_crop_zyx = u.read_nii(fileNII, cellpose=True)
    print(img_crop_zyx.shape)
    hf = h5py.File(fileH5, "a")
    _ = hf.create_dataset("raw", data=img_crop_zyx)
    hf.close()
    print(fileH5)


if __name__ == "__main__":
    folder = "/homedtic/dvarela/specimens/20190401_E2/membranes_decon_crops"
    for i in os.listdir(folder):
        if "nii.gz" in i:
            file_nii = os.path.join(folder, i)
            file_h5 = file_nii.replace(".nii.gz", "_ZYX.h5")
            convert_nii_rotate(file_nii, file_h5)
