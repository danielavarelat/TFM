#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import sys
import os
import numpy as np
from importlib import reload
import nibabel as nib
import pandas as pd
import importlib
from skimage import morphology
import porespy as ps
from collections import Counter
from math import sqrt


def rgbA(value, only_one_color="rgb", minimum=0, maximum=255):
    # if minimum != -1 or maximum != -1:
    # minimum, maximum = float(min(value)), float(max(value))
    zeros = np.zeros(shape=value.shape, dtype=value.dtype)
    if only_one_color == "rgb":
        ratio = 2 * (value - minimum) / (maximum - minimum)
        b = np.maximum(zeros, 255 * (1 - ratio))
        r = np.maximum(zeros, 255 * (ratio - 1))
        b = np.asarray(b, dtype="uint8")
        r = np.asarray(r, dtype="uint8")
        g = 255 - b - r
        rgb0 = np.array([r, g, b, zeros + 255]).transpose()
        return rgb0.astype("uint8")


def feature_colorRGB(df, variable):
    variable255 = df[variable] - df[variable].min()
    variable255 = 255 * (variable255 / variable255.max())
    variableRGB = rgbA(variable255)
    return [list(i) for i in variableRGB]


def eccentricity_3D(prop):
    l1, l2, l3 = prop.inertia_tensor_eigvals
    if l1 == 0:
        return 0
    return sqrt(1 - l2 / l1)


# ESPECIMEN = "20190404_E2"


if __name__ == "__main__":
    ## save
    especimens = [
        # "20190401_E2",
        "20190504_E1",
        "20190404_E2",
        "20190520_E4",
        "20190516_E3",
        "20190806_E3",
        "20190520_E2",
        "20190401_E3",
        "20190517_E1",
        "20190520_E1",
        "20190401_E1",
    ]
    for e in especimens:
        print(e)
        FILE = f"/Users/dvarelat/Documents/MASTER/TFM/DATA/EXTRACTION/{e}/cell_properties.csv"
        cellpose_nu = f"/Users/dvarelat/Documents/MASTER/TFM/DATA/RESULTS/nuclei/{e}_MASK_EQ_XYZ_decon.nii.gz"
        gasp_mem = f"/Users/dvarelat/Documents/MASTER/TFM/DATA/RESULTS/membranes/GASP_PNAS/{e}_mGFP_XYZ_predictions_GASP.nii.gz"
        linefile = (
            f"/Users/dvarelat/Documents/MASTER/TFM/DATA/LINES/line_{e}.nii.gz"
        )

        ## LEER ARCHIVOS
        pred_mem = nib.load(gasp_mem).get_fdata()
        print(pred_mem.shape)

        pred_nu = nib.load(cellpose_nu).get_fdata()
        print(pred_nu.shape)

        lines = nib.load(linefile).get_fdata()
        print(lines.shape)

        ## mask on nuclei
        mask_mem = np.where(pred_mem != 0, True, False)
        mask_on_nuclei = mask_mem * pred_nu

        ## props membrane
        img_mem = morphology.label(pred_mem)
        props_mem = ps.metrics.regionprops_3D(img_mem)
        centroids_mem = [[round(i) for i in p["centroid"]] for p in props_mem]
        original_labels_centroids = [pred_mem[c[0], c[1], c[2]] for c in centroids_mem]

        ### LINES EXTRACTION - TOUCHING
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
                # "eccentricities": [p.eccentricity for p in props_mem],
                "axis_major_length": [p.axis_major_length for p in props_mem],
                "axis_minor_length": l
                # "axis_minor_length": [p.axis_minor_length for p in props_mem],
            }
        )

        df = df[df.original_labels != 0]
        df = df[df.volumes < 1.5 * np.median(df.volumes)]
        df = df[df.volumes > 0.2 * np.median(df.volumes)]

        print(df.head())
        ### centroid approach for nuclei

        img_nu = morphology.label(mask_on_nuclei)
        props_nu = ps.metrics.regionprops_3D(img_nu)
        centroids_nu = [[round(i) for i in p["centroid"]] for p in props_nu]
        NU_original_labels_centroids = [pred_nu[c[0], c[1], c[2]] for c in centroids_nu]
        labels_centroid_nu_in_mem = [pred_mem[c[0], c[1], c[2]] for c in centroids_nu]
        dict_nuclei_membrane_centroids = dict(
            zip(labels_centroid_nu_in_mem, NU_original_labels_centroids)
        )
        dict_memlabel_cellnumber_props_nuclei = dict(
            zip(labels_centroid_nu_in_mem, range(len(centroids_nu)))
        )

        df["nuclei_label_cent"] = [
            dict_nuclei_membrane_centroids[label]
            if label in labels_centroid_nu_in_mem
            else -1
            for label in df.original_labels
        ]
        df["nuclei_cell_in_props"] = [
            dict_memlabel_cellnumber_props_nuclei[label]
            if label in labels_centroid_nu_in_mem
            else -1
            for label in df.original_labels
        ]
        df.to_csv(FILE, index=False, header=True)
        print(FILE)
        print("-------------")
