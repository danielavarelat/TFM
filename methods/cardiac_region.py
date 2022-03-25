import nibabel as nib
import numpy as np


def load3D_metadata(path):
    # Dictionary with information from the image
    dim_info = {}

    # Load image, but not physically in the RAM
    proxy = nib.load(path)

    # Metadata
    dim_info["x_size"] = proxy.header["dim"][1]
    dim_info["y_size"] = proxy.header["dim"][2]
    dim_info["z_size"] = proxy.header["dim"][3]

    # [u. distancia/pixel]
    dim_info["x_res"] = np.round(proxy.header["pixdim"][1], 6)
    dim_info["y_res"] = np.round(proxy.header["pixdim"][2], 6)
    dim_info["z_res"] = np.round(proxy.header["pixdim"][3], 6)

    return dim_info


def crop_line(lineXYZ_path, fileORIGINALRES, escala2048=True):
    # fileORIGINALRES --> puede ser decon pero no 05
    dim_info_line = load3D_metadata(lineXYZ_path)
    proxy_img_line = nib.load(lineXYZ_path)  # File with all lines
    line = proxy_img_line.get_data()
    print(f"LINE SHAPE = {line.shape}")
    todas_las_coordenadas_de_la_linea = np.where(line > 0)
    margenes_zona_cardiaca = (
        np.min(todas_las_coordenadas_de_la_linea, axis=1),
        np.max(todas_las_coordenadas_de_la_linea, axis=1),
    )
    if escala2048:
        print("SCALING --> 2048")
        dim_info = load3D_metadata(fileORIGINALRES)
        print(dim_info)
        margenes_zona_cardiaca[0][0] *= dim_info_line["x_res"] / dim_info["x_res"]
        margenes_zona_cardiaca[0][1] *= dim_info_line["y_res"] / dim_info["y_res"]
        margenes_zona_cardiaca[0][2] *= dim_info_line["z_res"] / dim_info["z_res"]

        margenes_zona_cardiaca[1][0] *= dim_info_line["x_res"] / dim_info["x_res"]
        margenes_zona_cardiaca[1][1] *= dim_info_line["y_res"] / dim_info["y_res"]
        margenes_zona_cardiaca[1][2] *= dim_info_line["z_res"] / dim_info["z_res"]
        ma = 20  # margen ampliacion
        for i in range(2):
            margenes_zona_cardiaca[0][i] = (
                margenes_zona_cardiaca[0][i] - ma
                if margenes_zona_cardiaca[0][i] - ma > 0
                else 0
            )
            margenes_zona_cardiaca[1][i] = (
                margenes_zona_cardiaca[1][i] + ma
                if margenes_zona_cardiaca[1][i] + ma < 2048
                else 2048
            )
            margenes_zona_cardiaca[0][2] = (
                margenes_zona_cardiaca[0][2] - ma
                if margenes_zona_cardiaca[0][2] - ma > 0
                else 0
            )
            margenes_zona_cardiaca[1][2] = (
                margenes_zona_cardiaca[1][2] + ma
                if margenes_zona_cardiaca[1][2] + ma < 566 * 2
                else 566 * 2
            )
    else:
        ma = 10  # margen ampliacion
        for i in range(2):
            margenes_zona_cardiaca[0][i] = (
                margenes_zona_cardiaca[0][i] - ma
                if margenes_zona_cardiaca[0][i] - ma > 0
                else 0
            )
            margenes_zona_cardiaca[1][i] = (
                margenes_zona_cardiaca[1][i] + ma
                if margenes_zona_cardiaca[1][i] + ma < 1024
                else 1024
            )
            margenes_zona_cardiaca[0][2] = (
                margenes_zona_cardiaca[0][2] - ma
                if margenes_zona_cardiaca[0][2] - ma > 0
                else 0
            )
            margenes_zona_cardiaca[1][2] = (
                margenes_zona_cardiaca[1][2] + ma
                if margenes_zona_cardiaca[1][2] + ma < 566
                else 566
            )
    return margenes_zona_cardiaca


def crop_embryo(margenesXYZ, deconxyz_path):
    decon = nib.load(deconxyz_path).get_fdata()
    crop = decon[
        margenesXYZ[0][0] : margenesXYZ[1][0],
        margenesXYZ[0][1] : margenesXYZ[1][1],
        margenesXYZ[0][2] : margenesXYZ[1][2],
    ]
    print(f"CROP SHAPE = {crop.shape}")
    return crop
