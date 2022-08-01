#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from skimage import morphology
import nibabel as nib
import json
import porespy as ps
from collections import Counter
import pandas as pd
import numpy as np
import sys
import importlib
import SimpleITK as sitk
from radiomics import shape

sys.path.insert(1, "/Users/dvarelat/Documents/MASTER/TFM/methods")
#sys.path.insert(1, "/homedtic/dvarela")
import cardiac_region as cR

f = open("/Users/dvarelat/Documents/MASTER/TFM/methods/specimens.json")
# f = open("/homedtic/dvarela/specimens.json")
data = json.load(f)


if __name__ == "__main__":
    FOLDERS = [
        element
        for sublist in [
            [f"{i[-1]}_2019" + e for e in data[i]]
            for i in ["stage1", "stage2", "stage3", "stage4"]
        ]
        for element in sublist
    ]
    for folder in FOLDERS:
        print(folder)
        #### INPUT!!! 
        ESPECIMEN = folder.split("_")[1] + "_" + folder.split("_")[2]
        FILE = f"/Users/dvarelat/Documents/MASTER/TFM/DATA/EXTRACTION/features/{ESPECIMEN}_cell_properties_radiomics.csv"
        linefile = (
            f"/Users/dvarelat/Documents/MASTER/TFM/DATA/LINES/line_{ESPECIMEN}.nii.gz"
        )
        gasp_mem = f"/Users/dvarelat/Documents/MASTER/TFM/DATA/RESULTS/membranes/GASP_PNAS/{ESPECIMEN}_mGFP_XYZ_predictions_GASP.nii.gz"
        mem = f"/Users/dvarelat/Documents/MASTER/TFM/DATA/CNIC/paraDaniela/mem/{ESPECIMEN}_mGFP_decon_0.5.nii.gz"
        
        #### -------------------------------------- 
        pred_mem = nib.load(gasp_mem).get_fdata()
        MEM = nib.load(mem).get_fdata()
        MEM = MEM[:, :, :, 0]
        lines = nib.load(linefile).get_fdata()
        print(pred_mem.shape)
        print(lines.shape)
        props_mem = ps.metrics.regionprops_3D(morphology.label(pred_mem))
        centroids_mem = [[round(i) for i in p["centroid"]] for p in props_mem]
        original_labels_centroids = [pred_mem[c[0], c[1], c[2]] for c in centroids_mem]
        print("Extracting features...")
        most_communs = []
        for p in props_mem:
            b = Counter(lines[p.slices].flatten())
            if len(list(b)) == 1:  ## si es solo 1, ese es
                m = list(b)[0]
            else:  # si hay varios. coger mayor diff cero
                d = {key: val for key, val in dict(b).items() if key != 0}
                m = max(d, key=d.get)
            most_communs.append(m)
        l = []
        for p in props_mem:
            try:
                l.append(p.axis_minor_length)
            except:
                l.append(0)
        df = pd.DataFrame(
            {
                "cell_in_props": range(len(props_mem)),
                "volumes": [p.volume for p in props_mem],
                "sphericities": [p.sphericity for p in props_mem],
                "original_labels": original_labels_centroids,
                "centroids": centroids_mem,
                "lines": most_communs,
                "axis_major_length": [p.axis_major_length for p in props_mem],
                "axis_minor_length": l,
                "solidity": [p.solidity for p in props_mem],
            }
        )
        df = df[df.original_labels != 0]
        med = np.median(df.volumes)
        print(f"Filtros {0.2 * med} - {10 * med}")
        df = df[df.volumes > 0.2 * med]
        df = df[df.volumes < 10 * med]
        
        ######### ADD RADIOMICS
        props_mem = ps.metrics.regionprops_3D(morphology.label(pred_mem))
        print(f"Props {len(props_mem)}")
        spacing = [
                float(cR.load3D_metadata(mem)["x_res"]),
                float(cR.load3D_metadata(mem)["y_res"]),
                float(cR.load3D_metadata(mem)["z_res"]),
            ]
        results = []
        for cell in df.cell_in_props:
            img = np.swapaxes(np.swapaxes(MEM[props_mem[cell].slice], 0, 2), 1, 2)
            m = np.swapaxes(np.swapaxes(props_mem[cell].mask, 0, 2), 1, 2)
            sitk_img = sitk.GetImageFromArray(img)
            sitk_img = sitk.JoinSeries(sitk_img)[:, :, :, 0]
            sitk_img.SetSpacing(spacing)
            sitk_mask = sitk.GetImageFromArray(m.astype("uint16"))
            sitk_mask = sitk.JoinSeries(sitk_mask)[:, :, :, 0]
            sitk_mask.SetSpacing(spacing)
            shapeFeatures = shape.RadiomicsShape(
                sitk_img,
                sitk_mask,
            )
            shapeFeatures.enableAllFeatures()
            result = shapeFeatures.execute()
            results.append(result)
        df = pd.concat([df, pd.DataFrame(results)], axis=1)
        print(df.shape)
        df.to_csv(FILE, index=False, header=True)

        
        
        
        