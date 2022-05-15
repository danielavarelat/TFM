#!/usr/bin/env python3
# -*- coding: utf-8 -*-


### SCRIPT TO BE RUN ON PORESPY CONDA ENV

import numpy as np
import scipy
import skimage
import porespy as ps
import mcubes
import trimesh
import csv
from skimage import morphology
from skimage.morphology import disk
import nibabel as nib
import pandas as pd
import time
import ast
import pickle


def merge(parts, concatenate_colors=True):
    """
    Esta funcion recibe -parts- que es una lista cuyos elementos son meshes de trimesh
    Lo que hace es juntar todas las meshes de la lista y expresarlas en una misma mesh"""
    # Concateno los vertices
    vertices = np.concatenate(([p.vertices for p in parts]), axis=0)

    # En cada componente de parts, el indice de las faces empieza desde 0
    # Con offset, se calcula el indice que se debe sumar a las caras de cada parte
    # cuando ests se unan en el mismo objeto trimesh
    offset = [0]
    for i, p in enumerate(parts):
        offset.append(int(p.faces.max() + 1))
    offset = np.cumsum(offset)

    # Concateno las caras añadiendo el offset
    faces = np.concatenate(([p.faces + offset[i] for i, p in enumerate(parts)]), axis=0)

    # Creo nueva mesh
    mesh2send = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    # Concateno los colores
    if concatenate_colors:
        colors = np.concatenate(
            ([p.visual.vertex_colors for i, p in enumerate(parts)]), axis=0
        )
        mesh2send.visual.vertex_colors = colors

    return mesh2send


def Median3D_Array(NumpyArray, disk_size):
    # disk_size: el tamaño de los objetos que el median filter elimina
    # objetos aislados de menor o igual tamaño se los ventila
    x_height, y_height, z_depth = (
        NumpyArray.shape[0],
        NumpyArray.shape[1],
        NumpyArray.shape[2],
    )

    # Hay arrays que se guardan como una cuarta dimension de valor =1 que no se porque lo hace...
    # Con esto lo quito
    if len(NumpyArray.shape) == 4:
        NumpyArray = NumpyArray[:, :, :, 0]

    return scipy.ndimage.median_filter(NumpyArray, size=disk_size)

    # Bluring over the {xy} plane
    for i in range(0, z_depth):
        NumpyArray[:, :, i] = skimage.filters.median(
            NumpyArray[:, :, i], disk(disk_size)
        )
    # Bluring over the {xz} plane
    for i in range(0, y_height):
        NumpyArray[:, i, :] = skimage.filters.median(
            NumpyArray[:, i, :], disk(disk_size)
        )
    # Bluring over the {yz} plane
    for i in range(0, x_height):
        NumpyArray[i, :, :] = skimage.filters.median(
            NumpyArray[i, :, :], disk(disk_size)
        )

    return NumpyArray


# ENTRADAS:
ESPECIMEN = "20190401_E1"

FILE = f"/Users/dvarelat/Documents/MASTER/TFM/DATA/EXTRACTION/{ESPECIMEN}/cell_properties.csv"
gasp = f"/Users/dvarelat/Documents/MASTER/TFM/DATA/RESULTS/membranes/GASP_PNAS/{ESPECIMEN}_mGFP_CardiacRegion_0.5_XYZ_predictions_GASP.nii.gz"

file_out = (
    f"/Users/dvarelat/Documents/MASTER/TFM/DATA/EXTRACTION/{ESPECIMEN}/{ESPECIMEN}.pkl"
)
if __name__ == "__main__":
    df_clean = pd.read_csv(FILE)
    print(f"Features shape {df_clean.shape}")
    print(df_clean.head())
    pred_mem = nib.load(gasp).get_fdata()
    print(f"Segmentation shape {pred_mem.shape}")
    img = morphology.label(pred_mem)
    props = ps.metrics.regionprops_3D(img)
    dict_label_cell = dict(zip(df_clean.original_labels, df_clean.cell_in_props))
    # dict_label_rgb_vol = dict(zip(df_clean.original_labels, df_clean.volumeRGB))
    # dict_label_rgb_esp = dict(zip(df_clean.original_labels, df_clean.sphericityRGB))
    disk_size = 3
    meshes = []
    ### loop over cells
    runtime = time.time()
    bad = []
    for i, cell_id in enumerate(list(df_clean.original_labels)):  # [::100]
        print(i)
        print(cell_id, "/", df_clean.shape[0])
        prop = props[dict_label_cell[cell_id]]
        coords = prop.mask * 1
        add = 10
        assert add % 2 == 0
        aux = np.zeros(shape=tuple(np.asarray(coords.shape) + add), dtype="uint8")
        aux[
            add // 2 : -add // 2, add // 2 : -add // 2, add // 2 : -add // 2
        ] = coords.copy()

        coords = aux.copy()
        del aux
        coords = Median3D_Array(coords.copy(), disk_size)
        vert, trian = mcubes.marching_cubes(mcubes.smooth(coords), 0)
        if len(vert) > 0 and len(trian) > 0:
            vert -= np.asarray([add // 2, add // 2, add // 2])
            vert += np.array([prop.slices[i].start for i in [0, 1, 2]])
            m_cell = trimesh.Trimesh(vert, trian, process=False)
            trimesh.smoothing.filter_taubin(m_cell, lamb=0.5, nu=-0.5, iterations=20)
            m_cell = m_cell.simplify_quadratic_decimation(int(100))  ## BAJARLE
            # m_cell.visual.vertex_colors = ast.literal_eval(dict_label_rgb_vol[cell_id])
            meshes.append(m_cell)
        else:
            print("NO")
            bad.append(cell_id)
    print(bad)
    runtime = time.time() - runtime
    print(f"Meshing took {runtime:.2f} s")
    # all_meshes = merge(meshes)
    with open(file_out, "wb") as f:
        pickle.dump(meshes, f)

    # with open(file_out, "w", encoding="utf-8") as f:
    #     all_meshes.export(f, file_type="obj")
