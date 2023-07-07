#!/bin/bash

# Define the variable to be checked
dataset="$1"
gpuid="$2"

# Perform the "switch" logic
case "$dataset" in
    "s3dis")
        echo "train network for stanford3D scene segmentation"
        CUDA_VISIBLE_DEVICES="$gpuid" python train/train_s3dis_render.py --batch_size=16 --data_dir=./data/S3DIS_3cm_hdf5_Rendered
        ;;
    "scannet")
        echo "train network for scannet scene segmentation"
        CUDA_VISIBLE_DEVICES="$gpuid" python train/train_scannet_render.py --batch_size=12 --data_dir=./data/ScanNet_2cm_hdf5_Rendered
        ;;
    *)
        echo "Invalid option: UNKNOWN dataset!"
        # Add code for handling invalid options
        ;;
esac

