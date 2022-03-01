import time, os, sys
import skimage.io
from cellpose import models, core, utils
import nibabel as nib
from cellpose.io import logger_setup

# sys.path.insert(1, "/Users/dvarelat/Documents/MASTER/TFM/methods")
sys.path.insert(1, "/homedtic/dvarela")

import util_daniela as u

USE_GPU = core.use_gpu()
print(">>> GPU activated? %d" % USE_GPU)


logger_setup()
SPECIMEN = "20190401_E2"
SPECIMEN_FOLDER = f"/homedtic/dvarela/specimens/{SPECIMEN}"


# OUT_FOLDER = "/homedtic/dvarela/pretrained/cellpose"
OUT_FOLDER = os.path.join(SPECIMEN_FOLDER, "cellpose_nuclei")
print("HERE")
print(OUT_FOLDER)

# IN_FOLDER = "/Users/dvarelat/Documents/MASTER/TFM/notebooks/testdata"

IN_FOLDER = os.path.join(SPECIMEN_FOLDER, "nuclei_decon_crops")


def run_3d_one(
    model, array_zyx, outfile, channels=[0, 0], diameter=25
):  # pylint: disable=dangerous-default-value
    """."""
    print("Predicting maks, flows, style...")
    masks, _, _, _ = model.eval(
        array_zyx, channels=channels, diameter=diameter, do_3D=USE_GPU
    )
    u.save_nii(masks, outfile)


def run_3d(
    model, list_3d_arrays, outs, channels=[0, 0], type_ch="nuclei", diameter=25
):  # pylint: disable=dangerous-default-value
    """."""

    print("Predicting ")
    print(list_3d_arrays[0].shape)
    channels = [channels]
    for i, a in enumerate(list_3d_arrays):
        mask, _, _, _ = model.eval(
            a,
            diameter=diameter,
            flow_threshold=None,
            channels=channels,
            do_3D=True,
        )
        u.save_nii(mask, outs[i])


if __name__ == "__main__":
    # files_IN = [
    #     "20190401_E2_DAPI_x1116to1352_y315to551_z337to539_decon.nii.gz",
    #     "20190401_E2_DAPI_x1116to1352_y315to551_z337to539_orig.nii.gz",
    # ]
    # l = [os.path.join(IN_FOLDER, i) for i in files_IN]
    # files_OUT = [
    #     os.path.join(OUT_FOLDER, i).replace(".nii.gz", "_MASK.nii.gz") for i in files_IN
    # ]
    # run_3d(u.read_multiple_nii(l), files_OUT)
    # decon_path = os.path.join(
    #     "/homedtic/dvarela/dataT", "20190401_E2_DAPI_decon_0.5.nii.gz"
    # )
    # full_img = u.read_nii(decon_path)
    # print(full_img.shape)
    # full_img = full_img[:, :, :, 0]
    # print(full_img.shape)
    # run_3d(
    #     [full_img],
    #     [
    #         os.path.join(OUT_FOLDER, "20190401_E2_DAPI_decon_0.5.nii.gz").replace(
    #             ".nii.gz", "_MASK.nii.gz"
    #         )
    #     ],
    # )

    ### TODO UN SPECIMEN
    modelCellpose = models.Cellpose(gpu=USE_GPU, model_type="nuclei")
    files_IN = [os.path.join(IN_FOLDER, i) for i in os.listdir(IN_FOLDER)]
    # revisar que todo sea de la extension
    files_IN = [i for i in files_IN if "nii.gz" in i]
    basenames = [os.path.basename(i) for i in files_IN]
    print(f"Getting inference from {len(files_IN)} files NII.GZ ")
    print(OUT_FOLDER)
    files_OUT = [
        os.path.join(OUT_FOLDER, i).replace(".nii.gz", "_MASKZYX.nii.gz")
        for i in basenames
    ]
    print(files_OUT[0:10])
    for i, crop in enumerate(files_IN):
        run_3d_one(modelCellpose, u.read_nii(crop, cellpose=True), files_OUT[i])
