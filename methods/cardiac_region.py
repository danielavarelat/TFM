import nibabel as nib
import numpy as np
import sys
import os
import json

sys.path.insert(1, "/homedtic/dvarela")
import util_daniela as u


def saveNifti(array_3D, array_dim_info, save_path, vox_units="um"):
    units_dict = {"um": 3, "mm": 2, "m": 1, "pixel": 0}
    tipo = array_3D.dtype.name
    if "uint" in tipo:
        bit_depth = 2 ** int(array_3D.dtype.name.split("uint")[1]) - 1
    elif "uint" in tipo:
        bit_depth = 255
    img = nib.Nifti1Image(array_3D, np.eye(4))
    # Se especifican unidades, dimensiones y tamano de pixel
    Dimensions = np.asarray(
        [
            1.0,
            array_dim_info["x_res"],
            array_dim_info["y_res"],
            array_dim_info["z_res"],
            1.0,
            0.0,
            0.0,
            0.0,
        ],
        dtype="float32",
    )
    img.header["pixdim"] = Dimensions
    img.header["xyzt_units"] = units_dict[vox_units]

    if "uint" in tipo:
        img.header["cal_max"] = bit_depth
    # Se guarda
    nib.save(img, save_path)


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

    f = open("/homedtic/dvarela/specimens.json")
    data = json.load(f)

    folder_lines = "/homedtic/dvarela/LINES"
    folder_decon05_mem = "/homedtic/dvarela/DECON_05/MGFP/mem"
    # folder_decon05_nu = "/homedtic/dvarela/DECON_05/DAPI/nu"
    especimens = [
        element
        for sublist in [[f"2019{e}" for e in data[i]] for i in ["stage6"]]
        for element in sublist
    ]

    # especimens = [i.split("_m")[0] for i in os.listdir(folder_decon05_mem)]

    for i, e in enumerate(especimens):
        print(f"Specimen {i} --> {e}")
        linefile = os.path.join(folder_lines, "line_" + e + ".nii.gz")
        deconxyz_path_mGFP = os.path.join(
            folder_decon05_mem, e + "_mGFP_decon_0.5.nii.gz"
        )
        # deconxyz_path_DAPI = os.path.join(
        #     folder_decon05_nu, e + "_DAPI_decon_0.5.nii.gz"
        # )

        margenesXYZ = crop_line(linefile, deconxyz_path_mGFP, escala2048=False, ma=5)

        # crop_n = crop_embryo(margenesXYZ, deconxyz_path_DAPI)
        crop_m = crop_embryo(margenesXYZ, deconxyz_path_mGFP)
        crop_l = crop_embryo(margenesXYZ, linefile)

        # lines_cc_folder = "/homedtic/dvarela/LINES/CC"
        # cardiac_region_folder_nu = "/homedtic/dvarela/CardiacRegion/all/nu"
        cardiac_region_folder_mem = "/homedtic/dvarela/CardiacRegion/all/mem"
        # if not os.path.isdir(cardiac_region_folder):
        #     try:
        #         os.mkdir(cardiac_region_folder)
        #     except OSError:
        #         print("Creation of the directory %s failed" % cardiac_region_folder)

        # u.save_nii(
        #     crop_n,
        #     os.path.join(
        #         cardiac_region_folder_nu, e + "_DAPI_CardiacRegion_0.5.nii.gz"
        #     ),
        # )
        u.save_nii(
            crop_m,
            os.path.join(
                cardiac_region_folder_mem, e + "_mGFP_CardiacRegion_0.5.nii.gz"
            ),
        )
        # u.save_nii(
        #     crop_l,
        #     os.path.join(lines_cc_folder, f"line_{e}_CC.nii.gz"),
        # )
        print("-------------------")
