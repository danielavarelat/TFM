#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 20:29:23 2020

@author: iesteban
"""

import trimesh
import scipy
import numpy as np
import matplotlib.cm as cm
import nibabel as nib
import gdist
import sys
import pandas as pd
import importlib

# import array2PLY
from matplotlib.colors import Normalize
import cardiac_region as c

especimens = ["20190520_E2"]
final_number_faces = 180000
d = 40
nombre = "_"


def ch2num(CHANNEL):
    if CHANNEL == "r":
        return 0
    if CHANNEL == "g":
        return 1
    if CHANNEL == "b":
        return 2


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

    elif only_one_color == "r" or only_one_color == "g" or only_one_color == "b":
        # Ratio para saltar de color en color
        ratio = (value - minimum) / (maximum - minimum)

        # Primero todo zeros
        rgb0 = np.array([zeros, zeros, zeros, zeros]).transpose()

        # Calculo rango de colores [0 255]
        color = np.maximum(zeros, 255 * (ratio))

        # coloco color en la columda correspondiente
        rgb0[:, ch2num(only_one_color)] = color

        return rgb0.astype("uint8")

    elif only_one_color == "grey":
        # Ratio para saltar de color en color
        ratio = (value - minimum) / (maximum - minimum)

        # Primero todo zeros
        rgb0 = np.array([zeros, zeros, zeros, zeros]).transpose()

        # Calculo rango de colores [0 255]
        color = np.maximum(zeros, 255 * (ratio))

        # El gris se caracteriza por tener todos los niveles de color igual
        rgb0[:, 0], rgb0[:, 1], rgb0[:, 2] = color, color, color

        return rgb0.astype("uint8")


# https://stackoverflow.com/questions/15140072/how-to-map-number-to-color-using-matplotlibs-colormap
def colorines(lista, colormap=cm.jet, mn=None, mx=None):
    if type(colormap).__name__ == "LinearSegmentedColormap":
        if mn == None:
            norm = Normalize(vmin=min(lista), vmax=max(lista))
        else:
            norm = Normalize(vmin=mn, vmax=mx)
        colores = colormap(norm(lista))
    else:
        colores = rgbA(lista, colormap)

    return colores


if __name__ == "__main__":
    for ESPECIMEN in ["20190520_E2"]:
        image_path = f"/Users/dvarelat/Documents/MASTER/TFM/DATA/RESULTS/membranes/GASP_PNAS/{ESPECIMEN}_mGFP_XYZ_predictions_GASP.nii.gz"
        DFFILE = f"/Users/dvarelat/Documents/MASTER/TFM/DATA/EXTRACTION/{ESPECIMEN}/cell_properties.csv"
        file_mesh = f"/homedtic/dvarela/midS/{ESPECIMEN}_myo_midS.ply"
        file_out = (
            f"/Users/dvarelat/Documents/MASTER/TFM/DATA/EXTRACTION/{ESPECIMEN}/{nombre}"
        )
        # df_clean = pd.read_csv(DFFILE)
        # print(df_clean.shape)
        # dict_label_cell = dict(zip(df_clean.original_labels, df_clean[variable]))
        # proxy = nib.load(image_path)
        # img = np.asarray(proxy.get_fdata(), dtype=proxy.get_data_dtype())
        # img = img.astype("uint16")
        # dim_info = c.load3D_metadata(image_path)
        mesh = trimesh.load_mesh(file_mesh, process=False)
        mesh = mesh.simplify_quadratic_decimation(int(final_number_faces))
        print(f"Numero de vertices {len(mesh.vertices)}")
        matrix = gdist.local_gdist_matrix(mesh.vertices, mesh.faces.astype(np.int32), d)
        matrix_ij = scipy.sparse.find(matrix)
        matrix_i, matrix_j, distances = matrix_ij[1], matrix_ij[0], matrix_ij[2]
        print(len(matrix_j))
        print(len(np.unique(matrix_j)))
        # j_coord = np.asarray(mesh.vertices[matrix_j])
        # j_pix = np.floor(
        #     j_coord / [dim_info["x_res"], dim_info["y_res"], dim_info["z_res"]]
        # ).astype("uint16")
        # j_VARIABLE_labels = img[j_pix[:, 0], j_pix[:, 1], j_pix[:, 2]]

        # dict_feature = {
        #     k: dict_label_cell[k] if k in list(df_clean.original_labels) else 0
        #     for k in set(j_VARIABLE_labels)
        # }

        # tight_i = [np.where(matrix_i == i)[0][0] for i in np.unique(matrix_i)]
        # withme_i_indices = np.split(matrix_j, tight_i)[1:]
        # thisfar_i = np.split(distances, tight_i)[1:]
        # withme_i_coord = np.split(j_coord, tight_i)[1:]
        # withme_i_pix = np.split(j_pix, tight_i)[1:]

        # j_VARIABLE_propiedad = [dict_feature[i] for i in j_VARIABLE_labels]
        # withme_i_VARIABLE = np.split(j_VARIABLE_propiedad, tight_i)[1:]
        # VARIABLE_per_i = np.asarray(
        #     [np.mean(p) for p in withme_i_VARIABLE]
        # )  # VALOR MEDIO

        # all_colors = np.zeros(shape=(mesh.vertices.shape[0], 4), dtype="float64")

        # colores = array2PLY.colorines(VARIABLE_per_i, cm.jet)

        # all_colors[np.unique(matrix_i)] = colores

        # mesh.visual.vertex_colors = all_colors
        # mesh.export(file_out)
        # print(file_out)
        # print("---------------")
        # break
