import trimesh
import numpy as np
import sys
import importlib
import pandas as pd
import nibabel as nib
import porespy as ps

import importlib
import mcubes
import pickle
from numpy import dot
from skimage import morphology
import json

sys.path.insert(1, "/Users/dvarelat/Documents/MASTER/TFM/methods")
# sys.path.insert(1, "/homedtic/dvarela")

import cardiac_region

importlib.reload(cardiac_region)
import cardiac_region as cR


def calculate_elongation(mesh):
    cov = np.cov(mesh.vertices.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    idx = eigenvalues.argsort()[::-1]
    eigenValues = eigenvalues[idx]
    return np.sqrt(eigenValues[1] / eigenValues[0])


if __name__ == "__main__":
    # f = open("/homedtic/dvarela/specimens.json")
    f = open("/Users/dvarelat/Documents/MASTER/TFM/methods/specimens.json")

    data = json.load(f)

    FOLDERS = [
        element
        for sublist in [
            [f"{i[-1]}_2019" + e for e in data[i]]
            for i in ["stage2", "stage3", "stage4"]
        ]
        for element in sublist
    ]
    for folder in FOLDERS:
        print(folder)
        ESPECIMEN = folder.split("_")[1] + "_" + folder.split("_")[2]
        gasp_mem = f"/Users/dvarelat/Documents/MASTER/TFM/DATA/RESULTS/membranes/GASP_PNAS/{ESPECIMEN}_mGFP_XYZ_predictions_GASP.nii.gz"
        pickle_spl = f"/Users/dvarelat/Documents/MASTER/TFM/DATA/EXTRACTION/features/list_meshes/{ESPECIMEN}_SPL_lines_corr.pkl"
        DFANGLE = f"/Users/dvarelat/Documents/MASTER/TFM/DATA/EXTRACTION/features/orientation/{ESPECIMEN}_angles_spl.csv"

        dim_info = cR.load3D_metadata(gasp_mem)
        df = pd.read_csv(DFANGLE)
        with open(pickle_spl, "rb") as f:
            readMESHES = pickle.load(f)
        if len(readMESHES) == df.shape[0]:
            print(f"MISMO SIZE {len(readMESHES)}")
            pred_mem = nib.load(gasp_mem).get_fdata()
            props = ps.metrics.regionprops_3D(morphology.label(pred_mem))
            indices_props = list(df.cell_in_props.values)
            print(f"Final number of cells {len(indices_props)}")
            props_set = [props[i] for i in indices_props]
            index_vertice = []
            elongation = []
            for i, p in enumerate(props_set):
                elongation.append(calculate_elongation(readMESHES[i]))
            df["Elongation2"] = elongation
            df.to_csv(DFANGLE, index=False, header=True)
            print(DFANGLE)
        else:
            print("DIFF TAMAÑO")
        print("-------------")
