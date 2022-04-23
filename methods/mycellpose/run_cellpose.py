import time, os, sys
import skimage.io

from cellpose import models, core
from importlib import reload

reload(models)
import nibabel as nib
from cellpose.io import logger_setup
import skimage.io

# sys.path.insert(1, "/Users/dvarelat/Documents/MASTER/TFM/methods")
sys.path.insert(1, "/homedtic/dvarela")

import util_daniela as u

USE_GPU = core.use_gpu()
print(">>> GPU activated? %d" % USE_GPU)


logger_setup()


def run_3d_one(
    model, array_zyx, outfile, channels=[0, 0], diameter=25
):  # pylint: disable=dangerous-default-value
    """."""
    print("Predicting maks, flows, style...")
    masks, _, _, _ = model.eval(
        array_zyx,
        channels=channels,
        diameter=diameter,
        normalize=True,
        do_3D=True,
        # stitch_threshold=0.01,
    )
    print(masks.shape)
    u.save_nii(masks, outfile)


# def run_3d(
#     model, list_3d_arrays, outs, channels=[0, 2], diameter=None
# ):  # pylint: disable=dangerous-default-value
#     """."""

#     print("Predicting ")
#     print(list_3d_arrays[0].shape)
#     channels = [channels]
#     for i, a in enumerate(list_3d_arrays):
#         mask, _, _, _ = model.eval(
#             a,
#             diameter=diameter,
#             flow_threshold=None,
#             channels=channels,
#             do_3D=True,
#             omni=False,
#             normalize=False,
#         )
#         u.save_nii(mask, outs[i])


if __name__ == "__main__":
    model_nuclei = models.Cellpose(gpu=USE_GPU, model_type="nuclei")
    OUT_FOLDER = "/homedtic/dvarela/pretrained/cellpose"

    region_path = "/homedtic/dvarela/CardiacRegion/20190401_E2/20190401_E2_DAPI_CardiacRegion_0.5_NORM_XYZ.nii.gz"

    file_OUT = os.path.join(OUT_FOLDER, os.path.basename(region_path)).replace(
        "XYZ.nii.gz", "_MASK_ZXY.nii.gz"
    )
    array = u.read_nii_XYZ(region_path, get_ZXY=True)
    print(array.shape)
    run_3d_one(model_nuclei, array, file_OUT)
    ### TODO UN SPECIMEN
    # modelCellpose = models.Cellpose(gpu=USE_GPU, model_type="nuclei")
    # model_nuclei = models.Cellpose(gpu=USE_GPU, model_type="nuclei")

    # files_IN = [os.path.join(IN_FOLDER, i) for i in os.listdir(IN_FOLDER)]
    # # revisar que todo sea de la extension
    # files_IN = [i for i in files_IN if "nii.gz" in i]
    # basenames = [os.path.basename(i) for i in files_IN]
    # print(f"Getting inference from {len(files_IN)} files NII.GZ ")
    # print(OUT_FOLDER)
    # files_OUT = [
    #     os.path.join(OUT_FOLDER, i).replace(".nii.gz", "_MASKZYX.nii.gz")
    #     for i in basenames
    # ]
    # print(files_OUT[0:10])
    # for i, crop in enumerate(files_IN):
    #     run_3d_one(model_nuclei, u.read_nii(crop, cellpose=True), files_OUT[i])
