import os
import numpy as np
import nibabel as nib
import pickle
import pandas as pd
import sys
import importlib
import porespy as ps
from skimage import morphology
from skimage.segmentation import find_boundaries
import json
import ast


f = open("/Users/dvarelat/Documents/MASTER/TFM/methods/specimens.json")
data = json.load(f)

### CONDA ACTIVATE IMAGES

sys.path.insert(1, "/Users/dvarelat/Documents/MASTER/TFM/methods")
import cardiac_region

importlib.reload(cardiac_region)
import cardiac_region as c

FOLDERS = [
    element
    for sublist in [
        [f"{i[-1]}_2019" + e for e in data[i]] for i in ["stage2", "stage3", "stage4"]
    ]
    for element in sublist
]
for i, folder in enumerate(FOLDERS):
    print(folder)
    ESPECIMEN = folder.split("_")[1] + "_" + folder.split("_")[2]
    FILE = f"/Users/dvarelat/Documents/MASTER/TFM/DATA/EXTRACTION/features/{ESPECIMEN}_cell_properties_radiomics.csv"
    gasp_mem = f"/Users/dvarelat/Documents/MASTER/TFM/DATA/RESULTS/membranes/GASP_PNAS/{ESPECIMEN}_mGFP_XYZ_predictions_GASP.nii.gz"
    df = pd.read_csv(FILE)
    pred_mem = nib.load(gasp_mem).get_fdata()
    print(pred_mem.shape)
    props_mem = ps.metrics.regionprops_3D(morphology.label(pred_mem))
    
    dfnoback = df[df.lines !=0]
    dfnoback = dfnoback.reset_index(drop=True)
    dfback = df[df.lines==0]
    new_lines = {}
    d = 45
    for i in dfback.cell_in_props:
        c = props_mem[i].centroid
        distances = dfnoback["centroids"].apply(lambda x: np.linalg.norm(np.array(ast.literal_eval(x))-np.array(c)))
        index = np.argsort(distances)[0]
        #print(f"Distance --> {distances[index]}")
        if distances[index] > d:
            new_lines[i] = 0
        else:
            new_lines[i] = dfnoback.loc[index, ["lines"]].values[0]
    df["improved_lines"] = df.apply(lambda x: x["lines"] if x["lines"] !=0 else new_lines[x["cell_in_props"]], axis=1)
    df.to_csv(FILE, index=False, header=True)
    print("---------------------")
        
    # pred_mem = nib.load(gasp_mem).get_fdata()
    # print(pred_mem.shape)
    # props_mem = ps.metrics.regionprops_3D(morphology.label(pred_mem))
    # df["improved_lines"] = df["lines"]
    # list_neigs = []
    # print(df.shape)
    # for r in range(df.shape[0]):
    #     mycent = ast.literal_eval(df.centroids[r])
    #     dist_tome = [calc_distance(mycent, ast.literal_eval(i)) for i in df.centroids]
    #     top_n_idx = np.argsort(dist_tome)[1 : N + 1]
    #     neigs = list(df.loc[top_n_idx].cell_in_props)
    #     list_neigs.append(neigs)
    #     if df.lines[r] == 0:
    #         neig_lines = list(df.loc[top_n_idx].lines)
    #         df.at[r, "improved_lines"] = max(set(neig_lines), key=neig_lines.count)
    # df["neighborhood"] = list_neigs
    # df.to_csv(FILE, index=False, header=True)
    # print("---------------------")
    