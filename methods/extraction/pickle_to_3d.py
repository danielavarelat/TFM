from lib2to3.pgen2.pgen import DFAState
import pickle
import numpy as np
import pandas as pd
import trimesh
import json

# CONDA ACTIVATE PROESPY


def feature_colorRGB(df, variable):
    variable255 = df[variable] - df[variable].min()
    variable255 = 255 * (variable255 / variable255.max())
    variableRGB = rgbA(variable255)
    return [list(i) for i in variableRGB]


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


def merge(parts, concatenate_colors=True):
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


if __name__ == "__main__":

    f = open("/Users/dvarelat/Documents/MASTER/TFM/methods/specimens.json")
    data = json.load(f)

    FOLDERS = [
        element
        for sublist in [
            [f"{i[-1]}_2019" + e for e in data[i]]
            for i in ["stage1", "stage2", "stage3", "stage4"]
        ]
        for element in sublist
    ]

    variables = [
        # "lines",
        # "improved_lines",
        # "sphericities",
        # "eccentricity3d",
        # "volumes",
        # "solidity",
        "Elongation",
        "Sphericity",
        "MeshVolume",
        # "Flatness",
        # "division"
        # "abs_angle"
    ]
    for folder in FOLDERS:
        ESPECIMEN = folder.split("_")[1] + "_" + folder.split("_")[2]
        print(folder)

        FILE = f"/Users/dvarelat/Documents/MASTER/TFM/DATA/EXTRACTION/features/list_meshes/{ESPECIMEN}_SPL_lines_corr.pkl"
        DFFILE = f"/Users/dvarelat/Documents/MASTER/TFM/DATA/EXTRACTION/features/{ESPECIMEN}_cell_properties_radiomics.csv"

        with open(FILE, "rb") as f:
            readMESHES = pickle.load(f)
        list_bads = []
        print(len(readMESHES))
        df = pd.read_csv(DFFILE)
        print(df.shape)
        df = df[df.spl == 1]
        print(f"Features SPLACHN {df.shape}")
        if len(readMESHES) == df.shape[0]:
            for variable in variables:
                print(variable)
                # OUT_myo = f"/Users/dvarelat/Documents/MASTER/TFM/DATA/EXTRACTION/features/orientation/SPLcells_{ESPECIMEN}_{variable}.ply"
                OUT_spl = f"/Users/dvarelat/Documents/MASTER/TFM/DATA/EXTRACTION/features/meshes/SPLcells_{ESPECIMEN}_{variable}.ply"
                # df_angles = pd.read_csv(OUTFILE)
                featRGB = feature_colorRGB(df, variable)
                dict_label_rgb = dict(zip(df.original_labels, featRGB))
                labels = list(df.original_labels)
                for j, color in enumerate(featRGB):
                    readMESHES[j].visual.vertex_colors = color
                all_readMESHES = merge(readMESHES)
                print(all_readMESHES)
                all_readMESHES.export(OUT_spl)
        else:
            print("no same size")
        print("----------------------")
