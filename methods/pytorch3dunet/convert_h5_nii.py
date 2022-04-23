import os
import numpy as np
import h5py
import nibabel as nib
import sys


if __name__ == "__main__":
    # mem = "/homedtic/dvarela/pretrained/pytunet3D/mytest_mem/20190401_E2_mGFP_CardiacRegion_0.5_ZXY_predictions.h5"
    # pred_zxy = dset = np.array(h5py.File(mem, "r")["predictions"])
    # print(pred_zxy.shape)
    # save_path = "/homedtic/dvarela/dataT/20190401_E2_mGFP_CardiacRegion_0.5_ZXY_predictions_pnas.nii.gz"
    # ni_img = nib.Nifti1Image(pred_zxy, affine=np.eye(4))
    # nib.save(ni_img, save_path)

    especimens = [
        "20190119_E1",
        "20190208_E2",
        "20190401_E1",
        "20190404_E1",
        "20190401_E2",
    ]
    files = [
        os.path.join(
            "/homedtic/dvarela/RESULTS/membranes/PNAS",
            e + "_mGFP_CardiacRegion_0.5_ZYX_predictions.h5",
        )
        for e in especimens
    ]
    for file_h5 in files:
        print(file_h5)
        pred_zxy = np.array(h5py.File(file_h5, "r")["predictions"])
        print(pred_zxy.shape)
        ni_img = nib.Nifti1Image(pred_zxy, affine=np.eye(4))
        nib.save(ni_img, file_h5.replace("h5", "nii.gz"))
    files = [
        os.path.join(
            "/homedtic/dvarela/RESULTS/membranes/UNet3D",
            e + "_mGFP_CardiacRegion_0.5_ZYX_predictions.h5",
        )
        for e in especimens
    ]
    for file_h5 in files:
        print(file_h5)
        pred_zxy = np.array(h5py.File(file_h5, "r")["predictions"])
        print(pred_zxy.shape)
        ni_img = nib.Nifti1Image(pred_zxy, affine=np.eye(4))
        nib.save(ni_img, file_h5.replace("h5", "nii.gz"))
