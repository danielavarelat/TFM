import numpy as np
import sys
import os
import numpy as np
from importlib import reload
import nibabel as nib
import importlib
import json

sys.path.insert(1, "/homedtic/dvarela")
import cardiac_region
import importlib

importlib.reload(cardiac_region)
import cardiac_region as c


if __name__ == "__main__":
    f = open("/homedtic/dvarela/specimens.json")
    data = json.load(f)
    flatten_list = [
        element
        for sublist in [data[i] for i in ["stage1", "stage2", "stage3", "stage4"]]
        for element in sublist
    ]
    gasp = "/homedtic/dvarela/RESULTS/membranes/GASP_PNAS"
    folder_lines = "/homedtic/dvarela/LINES"
    mems = "/homedtic/dvarela/DECON_05/MGFP/mem"
    for sp in flatten_list:
        print(sp)
        file_gasp = os.path.join(
            gasp, f"2019{sp}_mGFP_CardiacRegion_0.5_XYZ_predictions_GASP.nii.gz"
        )

        linefile = os.path.join(folder_lines, f"line_2019{sp}.nii.gz")
        print(linefile)
        decon = os.path.join(mems, f"2019{sp}_mGFP_decon_0.5.nii.gz")
        print(f"RUNNING {file_gasp}")
        decon05 = nib.load(decon).get_fdata()
        decon05 = decon05[:, :, :, 0]
        print(decon05.shape)
        pred_mem = nib.load(file_gasp).get_fdata()
        print(pred_mem.shape)
        margs = c.crop_line(linefile, decon, escala2048=False, ma=5)
        cut = c.crop_embryo(margs, decon)
        zeros = np.zeros(decon05.shape)
        zeros[
            margs[0][0] : margs[1][0],
            margs[0][1] : margs[1][1],
            margs[0][2] : margs[1][2],
        ] = pred_mem

        new_file = os.path.join(gasp, f"2019{sp}_mGFP_XYZ_predictions_GASP.nii.gz")
        c.saveNifti(zeros, c.load3D_metadata(decon), new_file)

        print("-------------------------")
