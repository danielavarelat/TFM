import trimesh
import scipy
import numpy as np
import matplotlib.cm as cm
import nibabel as nib
import gdist
import sys
import pandas as pd
import importlib
import os
import ast
import json
import math
import pickle


sys.path.insert(1, "/Users/dvarelat/Documents/MASTER/TFM/methods/extraction")
import pickle_to_3d as p

sys.path.insert(1, "/Users/dvarelat/Documents/MASTER/TFM/methods")
import cardiac_region

importlib.reload(cardiac_region)
import cardiac_region as c


def multiDimenDist(point1, point2):
    # find the difference between the two points, its really the same as below
    deltaVals = [
        point2[dimension] - point1[dimension] for dimension in range(len(point1))
    ]
    runningSquared = 0
    # because the pythagarom theorm works for any dimension we can just use that
    for coOrd in deltaVals:
        runningSquared += coOrd**2
    return runningSquared ** (1 / 2)


def findVec(point1, point2, unitSphere=False):
    # setting unitSphere to True will make the vector scaled down to a sphere with a radius one, instead of it's orginal length
    finalVector = [0 for coOrd in point1]
    for dimension, coOrd in enumerate(point1):
        # finding total differnce for that co-ordinate(x,y,z...)
        deltaCoOrd = point2[dimension] - coOrd
        # adding total difference
        finalVector[dimension] = deltaCoOrd
    if unitSphere:
        totalDist = multiDimenDist(point1, point2)
        unitVector = []
        for dimen in finalVector:
            unitVector.append(dimen / totalDist)
        return unitVector
    else:
        return finalVector


def dotproduct(v1, v2):
    return sum((a * b) for a, b in zip(v1, v2))


def length(v):
    return math.sqrt(dotproduct(v, v))


def angle(v1, v2):
    return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))


FOLDERS = [
    "1_20190504_E1",
    "7_20190404_E2",
    "10_20190520_E4",
    "3_20190516_E3",
    "4_20190806_E3",
    "2_20190520_E2",
    "5_20190401_E3",
    "8_20190517_E1",
    "9_20190520_E1",
    "6_20190401_E1",
]


for folder in FOLDERS:
    print(folder)
    ESPECIMEN = folder.split("_")[1] + "_" + folder.split("_")[2]
    file_mesh = f"/Users/dvarelat/Documents/MASTER/TFM/midS/{ESPECIMEN}_myo_midS.ply"
    midplane = (
        f"/Users/dvarelat/Documents/MASTER/TFM/midplanes/{ESPECIMEN}_midplane.ply"
    )
    folder_surface = (
        f"/Users/dvarelat/Documents/MASTER/TFM/DATA/EXTRACTION/SURFACE/{ESPECIMEN}/myo"
    )
    image_path = f"/Users/dvarelat/Documents/MASTER/TFM/DATA/RESULTS/membranes/GASP_PNAS/{ESPECIMEN}_mGFP_XYZ_predictions_GASP.nii.gz"
    DFFILE = f"/Users/dvarelat/Documents/MASTER/TFM/DATA/EXTRACTION/{folder}/{ESPECIMEN}_cell_properties_radiomics.csv"
    land_json = (
        f"/Users/dvarelat/Documents/MASTER/TFM/landmarks/{ESPECIMEN}_key_points.json"
    )

    mesh_mid = trimesh.load_mesh(midplane, process=False)
    print(mesh_mid.vertices.shape)
    dim_info = c.load3D_metadata(image_path)
    vertices_location = np.floor(
        mesh_mid.vertices / [dim_info["x_res"], dim_info["y_res"], dim_info["z_res"]]
    ).astype("uint16")
    vertices_location = vertices_location[vertices_location[:, 2] < dim_info["z_size"]]
    vertices_location = vertices_location[vertices_location[:, 0] < dim_info["x_size"]]
    vertices_location = vertices_location[vertices_location[:, 1] < dim_info["y_size"]]
    print(vertices_location.shape)

    n_points = 1
    points, faces = mesh_mid.sample(n_points, return_index=True)
    normal = mesh_mid.face_normals[faces]

    df = pd.read_csv(DFFILE)
    df["angle"] = df["centroids"].apply(
        lambda x: angle(
            normal[0, :], findVec(list(vertices_location[0]), ast.literal_eval(x))
        )
    )
    df["degrees"] = df["angle"].apply(lambda x: x * (180.0 / math.pi))

    with open(land_json) as json_file:
        data = json.load(json_file)
    p1 = np.round(data["p1"])
    vector_l1_plane = findVec(list(vertices_location[0]), p1)

    if angle(normal[0, :], vector_l1_plane) * (180.0 / math.pi) > 90:
        print("Normal pa la RIGHT")
        df["RL"] = df["degrees"].apply(lambda x: 1 if x > 90 else 0)
    else:
        print("Normal pa la LEFT")
        df["RL"] = df["degrees"].apply(lambda x: 0 if x > 90 else 1)
    print(df.groupby(["RL"]).count()["cell_in_props"])
    df.to_csv(DFFILE, index=False, header=True)
    
    ##COLOR MESH
    
    # OUT_myo = f"/Users/dvarelat/Documents/MASTER/TFM/DATA/EXTRACTION/cells_RL/{ESPECIMEN}_RL.ply"
    # FILE = f"/Users/dvarelat/Documents/MASTER/TFM/DATA/EXTRACTION/{folder}/myo/{ESPECIMEN}_MYOmesh.pkl"

    # DF_MYO = df[df.lines == 1]
    # with open(FILE, "rb") as f:
    #     readMESHES = pickle.load(f)
    # featRGB = p.feature_colorRGB(DF_MYO, "RL")
    # dict_label_rgb = dict(zip(DF_MYO.original_labels, featRGB))
    # for j, cell_id in enumerate(list(DF_MYO.original_labels)):
    #     readMESHES[j].visual.vertex_colors = dict_label_rgb[cell_id]
    # all_readMESHES = p.merge(readMESHES)
    # print(all_readMESHES)
    # all_readMESHES.export(OUT_myo)
    
    print("----------------------")
