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
import array2PLY


sys.path.insert(1, "/Users/dvarelat/Documents/MASTER/TFM/methods")
import cardiac_region

importlib.reload(cardiac_region)
import cardiac_region as c


especimens = [
    "20190504_E1",
    "20190404_E2",
    "20190520_E4",
    "20190516_E3",
    "20190806_E3",
    # "20190520_E2",
    "20190401_E3",
    "20190517_E1",
    "20190520_E1",
    "20190401_E1",
]

FOLDERS = [
    "1_20190504_E1",
    "7_20190404_E2",
    "10_20190520_E4",
    "3_20190516_E3",
    "4_20190806_E3",
    # "2_20190520_E2",
    "5_20190401_E3",
    "8_20190517_E1",
    "9_20190520_E1",
    "6_20190401_E1",
]

ESPECIMEN = especimens[0]
d = 40
final_number_faces = 18000
# variable = "sphericities"

# ENTRADAS
# image_path = f"/Users/dvarelat/Documents/MASTER/TFM/DATA/RESULTS/membranes/GASP_PNAS/{ESPECIMEN}_mGFP_XYZ_predictions_GASP.nii.gz"
# DFFILE = f"/Users/dvarelat/Documents/MASTER/TFM/DATA/EXTRACTION/{ESPECIMEN}/cell_properties.csv"
# file_mesh = f"/Users/dvarelat/Documents/MASTER/TFM/midS/{ESPECIMEN}_myo_midS.ply"
# file_out = f"/Users/dvarelat/Documents/MASTER/TFM/DATA/EXTRACTION/{ESPECIMEN}/{nombre}"

# ERROR --> 20190520_E2
variables = [
    "Sphericity",
    "MeshVolume",
    "Elongation",
    "Flatness",
    # "improved_lines",
]
if __name__ == "__main__":
    for i, ESPECIMEN in enumerate(especimens):
        print(ESPECIMEN)
        try:
            image_path = f"/Users/dvarelat/Documents/MASTER/TFM/DATA/RESULTS/membranes/GASP_PNAS/{ESPECIMEN}_mGFP_XYZ_predictions_GASP.nii.gz"
            DFFILE = f"/Users/dvarelat/Documents/MASTER/TFM/DATA/EXTRACTION/{FOLDERS[i]}/cell_properties_radiomics.csv"
            file_mesh = (
                f"/Users/dvarelat/Documents/MASTER/TFM/midS/{ESPECIMEN}_myo_midS.ply"
            )

            df_clean = pd.read_csv(DFFILE)
            print(df_clean.shape)

            proxy = nib.load(image_path)
            img = np.asarray(proxy.get_fdata(), dtype=proxy.get_data_dtype())
            img = img.astype("uint16")
            print(img.shape)
            dim_info = c.load3D_metadata(image_path)
            mesh = trimesh.load_mesh(file_mesh, process=False)
            mesh = mesh.simplify_quadratic_decimation(int(final_number_faces))
            print(f"Numero de vertices {len(mesh.vertices)}")
            matrix = gdist.local_gdist_matrix(
                mesh.vertices, mesh.faces.astype(np.int32), d
            )
            matrix_ij = scipy.sparse.find(matrix)
            matrix_i, matrix_j, distances = matrix_ij[1], matrix_ij[0], matrix_ij[2]
            j_coord = np.asarray(mesh.vertices[matrix_j])
            j_pix = np.floor(
                j_coord / [dim_info["x_res"], dim_info["y_res"], dim_info["z_res"]]
            ).astype("uint16")
            ##labels segun las posiciones de los vertices
            j_VARIABLE_labels = img[j_pix[:, 0], j_pix[:, 1], j_pix[:, 2]]
            unique_labels = set(j_VARIABLE_labels)
            print(f"Unique labels {len(unique_labels)}")
            print(len(unique_labels.intersection(set(df_clean.original_labels))))
            tight_i = [np.where(matrix_i == i)[0][0] for i in np.unique(matrix_i)]
            for v in variables:
                nombre = ESPECIMEN + "_" + v + "_" + str(d) + ".ply"
                print(f"Creating {nombre}")
                file_out = f"/Users/dvarelat/Documents/MASTER/TFM/DATA/EXTRACTION/{FOLDERS[i]}/{nombre}"
                dict_label_cell = dict(zip(df_clean.original_labels, df_clean[v]))
                # recorrrer los labels según las posiciones de los vertices para encontrar el valor de la propiedad
                dict_feature = {
                    k: dict_label_cell[k] if k in list(df_clean.original_labels) else 0
                    for k in unique_labels
                }
                j_VARIABLE_propiedad = [dict_feature[i] for i in j_VARIABLE_labels]
                withme_i_VARIABLE = np.split(j_VARIABLE_propiedad, tight_i)[1:]
                VARIABLE_per_i = np.asarray(
                    [np.mean(p) for p in withme_i_VARIABLE]
                )  # VALOR MEDIO
                all_colors = np.zeros(
                    shape=(mesh.vertices.shape[0], 4), dtype="float64"
                )
                colores = array2PLY.colorines(VARIABLE_per_i, cm.jet)
                all_colors[np.unique(matrix_i)] = colores
                mesh.visual.vertex_colors = all_colors
                mesh.export(file_out)
                print(file_out)
            print("---------------")
        except:
            print("NOOOOOOO")
            pass
