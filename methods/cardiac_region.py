import nibabel as nib
import numpy as np
import sys
import os

sys.path.insert(1, "/homedtic/dvarela")
import util_daniela as u


def load3D_metadata(path):
    """Path to file XYZ oriented."""
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


def crop_line(lineXYZ_path, fileORIGINALRES, escala2048=False, ma=5):
    # fileORIGINALRES --> puede ser decon pero no 05
    print(f"MARGIN : {ma}")
    dim_info_line = load3D_metadata(lineXYZ_path)
    print("Line information")
    print(dim_info_line)
    proxy_img_line = nib.load(lineXYZ_path)  # File with all lines
    line = proxy_img_line.get_data()
    print(f"LINE SHAPE = {line.shape}")
    todas_las_coordenadas_de_la_linea = np.where(line > 0)
    margenes_zona_cardiaca = (
        np.min(todas_las_coordenadas_de_la_linea, axis=1),
        np.max(todas_las_coordenadas_de_la_linea, axis=1),
    )
    if escala2048:
        print("SCALING WITH DECON ORIGINAL --> 2048")
        dim_info = load3D_metadata(fileORIGINALRES)
        print(dim_info)
        margenes_zona_cardiaca[0][0] *= dim_info_line["x_res"] / dim_info["x_res"]
        margenes_zona_cardiaca[0][1] *= dim_info_line["y_res"] / dim_info["y_res"]
        margenes_zona_cardiaca[0][2] *= dim_info_line["z_res"] / dim_info["z_res"]

        margenes_zona_cardiaca[1][0] *= dim_info_line["x_res"] / dim_info["x_res"]
        margenes_zona_cardiaca[1][1] *= dim_info_line["y_res"] / dim_info["y_res"]
        margenes_zona_cardiaca[1][2] *= dim_info_line["z_res"] / dim_info["z_res"]
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
                if margenes_zona_cardiaca[1][2] + ma < dim_info["z_size"] * 2
                else dim_info["z_size"] * 2
            )
    else:
        print("SCALING WITH DECON 05 --> 1024")
        dim_info = load3D_metadata(fileORIGINALRES)
        print(dim_info)
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
                if margenes_zona_cardiaca[1][2] + ma < dim_info["z_size"]
                else dim_info["z_size"]
            )
    return margenes_zona_cardiaca


def crop_embryo(margenesXYZ, deconxyz_path):
    decon = nib.load(deconxyz_path).get_fdata()
    cropXYZ = decon[
        margenesXYZ[0][0] : margenesXYZ[1][0],
        margenesXYZ[0][1] : margenesXYZ[1][1],
        margenesXYZ[0][2] : margenesXYZ[1][2],
    ]
    print(f"CROP SHAPE = {cropXYZ.shape}")
    return cropXYZ


if __name__ == "__main__":
    folder_lines = "/homedtic/dvarela/LINES"
    # folder_decon05 = "/homedtic/dvarela/DECON_05/MGFP"
    folder_decon05 = "/homedtic/dvarela/DECON_05/DAPI"
    especimens = [
        "20190119_E1",
        "20190208_E2",
        "20190401_E1",
        "20190404_E1",
    ]
    for e in especimens:
        print(e)
        line = os.path.join(folder_lines, "line_" + e + ".nii.gz")
        # deconxyz_path = os.path.join(folder_decon05, e + "_mGFP_decon_0.5.nii.gz")
        deconxyz_path = os.path.join(folder_decon05, e + "_DAPI_decon_0.5.nii.gz")
        cardiac_region_folder = os.path.join("/homedtic/dvarela/CardiacRegion", e)
        margenesXYZ = crop_line(line, deconxyz_path, escala2048=False, ma=5)
        # deconxyz_path_nuclei = "/homedtic/dvarela/dataT/20190401_E2_DAPI_decon_0.5.nii.gz"
        crop_n = crop_embryo(margenesXYZ, deconxyz_path)
        u.save_nii(
            crop_n,
            os.path.join(cardiac_region_folder, e + "_DAPI_CardiacRegion_0.5.nii.gz"),
        )
        print("-------")
        # u.save_nii(
        #     crop_m, "/homedtic/dvarela/dataT/20190401_E2_mGFP_CardiacRegion_0.5.nii.gz"
        # )
