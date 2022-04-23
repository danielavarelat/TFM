import stardist
from stardist.models import StarDist3D
import nibabel as nib
import cv2

print("hERE")

model = StarDist3D(None, name="3D_Iso_Basic", basedir="model_stardist")

# crop_0_n = "/homedtic/dvarela/dataT/20190401_E2_DAPI_x1116to1352_y315to551_z337to539_decon.nii.gz"
# crop_0_n = nib.load(crop_0_n).get_fdata()
# zyx_crop_0_n = np.swapaxes(crop_0_n, 0, 2)
# print(zyx_crop_0_n.shape)

# image_norm = cv2.normalize(
#     zyx_crop_0_n, None, alpha=0, beta=200, norm_type=cv2.NORM_MINMAX
# )
# # labels, details = model.predict_instances(image_norm)

# print(model)
# print(type(model))

# labels, details = model.predict_instances(
#     image_norm,
#     axes=None,
#     normalizer=None,
#     sparse=True,
#     prob_thresh=None,
#     nms_thresh=None,
#     scale=None,
#     n_tiles=None,
#     show_tile_progress=True,
#     verbose=True,
#     return_labels=True,
#     predict_kwargs=None,
#     nms_kwargs=None,
#     overlap_label=None,
#     return_predict=False,
# )
# ni_img = nib.Nifti1Image(np.swapaxes(labels, 0, 2), affine=np.eye(4))
# nib.save(ni_img, "crop0_xyz_labels_stardist.nii.gz")
