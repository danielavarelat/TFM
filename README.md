# TFM SEGMENTATION


## DATA DESCRIPTION

RED CHANNEL (Tomato): 
  - Mesoderm cytoplasm 
  - Ecto and endoderm membranes

GREEN CHANNEL (mGFP)
  - Mesorderm membranes

DAPI
  - All nuclei 


**RESOLUTION**
Volumes of ~1mm in the X and Y dimension and down to a depth of 677µm.

High-resolution images:  XY pixel resolution varying from 0.38µm to 0.49µm and a z step varying from 0.49µm to 2.0µm, depending on the stage.

Raw images: 202

**Voxel spacing**
X res = Y res = 0.854
Z res = 0.99



## CONTEXT

Cardiac development starts with cardiac **MESODERM**. 


## PIPELINE

INPUTS: 
	- DATA/DECON_05/DAPI/20190119_E1_DAPI_decon_0.5.nii.gz
	- DATA/DECON_05/MGFP/20190401_E3_mGFP_decon_0.5.nii.gz
	- DATA/LINES/line_20190401_E2.nii.gz
	

	1. Crop cardiac region from DAPI and mGFP using lines
		methods/cardiac_region.py
		methods/crop_region.sh

		Input: 
		20190119_E1_DAPI_decon_0.5.nii.gz
		20190119_E1_mGFP_decon_0.5.nii.gz
		Output: 
		20190119_E1_DAPI_CardiacRegion_0.5.nii.gz
		20190119_E1_mGFP_CardiacRegion_0.5.nii.gz

	2. Process membranes
		a. Convert nii XYZ to h5 ZXY 
			methods/pytorch3dunet/convert_nii_h5.py 
			methods/pytorch3dunet/conversion.sh
			
			Input: 
			20190209_E2_mGFP_CardiacRegion_0.5.nii.gz
			Output: 
			20190209_E2_mGFP_CardiacRegion_0.5_ZXY.h5
			
		b. Run semantic segmentation, pytorch 3d unet using .yml 
			methods/pytorch3dunet/my_test_config_mem.yml
			methods/pytorch3dunet/pytorch3dunet_mem.sh
			
			Input: 
			20190404_E1_mGFP_CardiacRegion_0.5_ZXY.h5
			models/best_checkpoint_PNAS_plantseg.pytorch
			Output: 
			20190404_E1_mGFP_CardiacRegion_0.5_ZYX_predictions.h5
			
		c. Run instance segmentation: gasp 
			methods/postpro/run_gasp.py 
			methods/postpro/GASP.sh
			Input:
			20190404_E1_mGFP_CardiacRegion_0.5_ZXY_predictions.h5
			Output:
			20190404_E1_mGFP_CardiacRegion_0.5_XYZ_predictions_GASP.nii.gz
			
		d. Padding: add the rest of the original image at the borders to complete the original size again 
			methods/padding.py 
			methods/apadding.sh
			Input:
			20190404_E1_mGFP_CardiacRegion_0.5_XYZ_predictions_GASP.nii.gz
			Output:
			20190404_E1_mGFP_XYZ_predictions_GASP.nii.gz
