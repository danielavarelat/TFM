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
			
			
			
			
# FEATURES EXTRACTION

## 1. Features + radiomics

First features are extracted from the segmentation mask, the rest come from pyradiomics package.

[df_features_radiomics.py](https://github.com/danielavarelat/TFM/blob/master/methods/extraction/df_features_radiomics.py)

This script does not include the nuclei information, that is in: [df_features.py](https://github.com/danielavarelat/TFM/blob/master/methods/extraction/df_features.py)

	INPUT
		linefile = DATA/LINES/line_{ESPECIMEN}.nii.gz
		gasp_mem = DATA/RESULTS/membranes/GASP_PNAS/{ESPECIMEN}_mGFP_XYZ_predictions_GASP.nii.gz
		mem = DATA/DECON_05/{ESPECIMEN}_mGFP_decon_0.5.nii.gz
	
	OUTPUT
		DFFILE = {ESPECIMEN}_cell_properties_radiomics.csv
		
		Columns: 'cell_in_props', 'volumes', 'sphericities', 'original_labels',
		       'centroids', 'lines', 'axis_major_length', 'axis_minor_length',
		       'solidity','Elongation','Flatness', 'LeastAxisLength', 'MajorAxisLength','Maximum2DDiameterColumn', 
		       'Maximum2DDiameterRow','Maximum2DDiameterSlice', 'Maximum3DDiameter', 'MeshVolume',
		       'MinorAxisLength', 'Sphericity', 'SurfaceArea', 'SurfaceVolumeRatio', 'VoxelVolume'
		       
		       
## 2. Classify (Myo and Spl)

Although in column "lines" the different tissues are categorized, new column "myo" and "spl" are added as a more sophisticated classification based on lines but also in the 3D volumetric mesh. 

[myo_spl.py](https://github.com/danielavarelat/TFM/blob/master/methods/extraction/myo_spl.py)

	INPUT
		DFFILE = DATA/EXTRACTION/features/{ESPECIMEN}_cell_properties_radiomics.csv
		linefile = DATA/LINES/line_{ESPECIMEN}.nii.gz
		gasp_mem = DATA/RESULTS/membranes/GASP_PNAS/{ESPECIMEN}_mGFP_XYZ_predictions_GASP.nii.gz
		mesh_myo = lines_ply_myo/line_{ESPECIMEN}_myo_10000.ply
        	mesh_spl = lines_ply_spl/line_{ESPECIMEN}_spl_10000.ply

	OUTPUT
		DFFILE = DATA/EXTRACTION/features/{ESPECIMEN}_cell_properties_radiomics.csv
	    
	   
## 3. Create meshes from every cell

Based on "myo" and "spl" columns, separately. 

	INPUT (Example for splanchnic)
	
		DFFILE = DATA/EXTRACTION/features/{ESPECIMEN}_cell_properties_radiomics.csv
		gasp_mem = DATA/RESULTS/membranes/GASP_PNAS/{ESPECIMEN}_mGFP_XYZ_predictions_GASP.nii.gz
        	line_mesh = lines_ply_spl/line_{ESPECIMEN}_spl_10000.ply
		bad_json = DATA/EXTRACTION/features/list_meshes/pickles_spl_{ESPECIMEN}.json
	
	OUTPUT
		file_out = DATA/EXTRACTION/features/list_meshes/{ESPECIMEN}_SPL_lines_corr.pkl

Results are stored as a list of meshes in a pickle. Additionally, the index of cells that were not able to create meshes are stored in a json. 

	### 3.1 Remove bad cells (not having mesh)
	
	Cells indices that are in the json file are marked as 0 instead of 1 from the "spl" column (or "myo"). 
	This is done to keep using those columns as guide for the list of meshes and further applications. 
	
## 4. Calculate orientation and other features from single meshes
	
This is not included in any script. Code:
	
      f = open(".../DATA/EXTRACTION/features/list_meshes/pickles_spl.json")
      d_spl = json.load(f)
      d_spl
      {'20190806_E5': [], '20190516_E3': [10831], '20190523_E1': [], '20190806_E4': [], '20190806_E6': []}
      for k,v in d_spl.items(): 
      if v:
	print(k)
	DFFILE = f"DATA/EXTRACTION/features/{k}_cell_properties_radiomics.csv"
	df = pd.read_csv(DFFILE)
	df["spl"] = df.apply(lambda x: 0 if x["cell_in_props"] in v else x["spl"], axis=1)
	print(df[df.spl == 1].shape)
	df.to_csv(DFFILE, index=False, header=True)






