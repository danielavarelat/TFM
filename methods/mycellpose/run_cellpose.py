import time, os, sys
import skimage.io

from cellpose import models, core
from importlib import reload

reload(models)
import nibabel as nib
import skimage.io
from skimage import exposure


# sys.path.insert(1, "/Users/dvarelat/Documents/MASTER/TFM/methods")
sys.path.insert(1, "/homedtic/dvarela")

import util_daniela as u

USE_GPU = core.use_gpu()
print(">>> GPU activated? %d" % USE_GPU)


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
    # files = [
    #     "/homedtic/dvarela/CardiacRegion/20190401_E2/20190401_E2_DAPI_CardiacRegion_0.5_equalized_XYZ.nii.gz",
    #     "/homedtic/dvarela/CardiacRegion/20190401_E2/20190401_E2_DAPI_CardiacRegion_0.5_stretched_XYZ.nii.gz",
    # ]
    # filesout = [
    #     "/homedtic/dvarela/pretrained/cellpose/20190401_E2_DAPI_CardiacRegion_0.5_equalized_MASK_ZXY.nii.gz",
    #     "/homedtic/dvarela/pretrained/cellpose/20190401_E2_DAPI_CardiacRegion_0.5_stretched_MASK_ZXY.nii.gz",
    # ]

    especimens = [
        # "20190504_E1",
        # "20190404_E2",
        # "20190520_E4",
        # "20190516_E3",
        # "20190806_E3",
        # "20190520_E2",
        # "20190401_E3",
        # "20190517_E1",
        # "20190520_E1",
        "20190401_E1",
    ]
    files = [
        os.path.join(
            os.path.join("/homedtic/dvarela/CardiacRegion", e),
            f"{e}_DAPI_CardiacRegion_0.5.nii.gz",
        )
        for e in especimens
    ]
    filesout = [
        os.path.join("/homedtic/dvarela/RESULTS/nuclei", f"{e}_MASK_EQ_ZXY.nii.gz")
        for e in especimens
    ]

    for i, file_nii in enumerate(files):
        arrayzxy = u.read_nii_XYZ(file_nii, get_ZXY=True)
        print(arrayzxy.shape)
        equalized = exposure.equalize_hist(arrayzxy)
        print("Equalized")
        run_3d_one(model_nuclei, equalized, filesout[i])
