#!/bin/bash

# download
gdown --id 1-eTfgc3degGAN1cyp6YdZS21EDpRwO3z --output S3DIS_3cm_hdf5_Rendered.zip
gdown --id 1MPVlsgqwtOteQyPaxcmK2FKHv_ykFvQy --output S3DIS-Aligned-Raw.zip

# unzip
unzip S3DIS_3cm_hdf5_Rendered.zip
unzip S3DIS-Aligned-Raw.zip

# delete .zip file
rm S3DIS_3cm_hdf5_Rendered.zip
rm S3DIS-Aligned-Raw.zip