#!/bin/bash

# Define the variable to be checked
dataset="$1"
gpuid="$2"
trained_model="$3"

# Perform the "switch" logic
case "$dataset" in
    "s3dis")
        echo "test segmentation for stanford3D scenes"
        CUDA_VISIBLE_DEVICES="$gpuid" python eval/evaluate_s3dis_full.py --data_dir=./data/S3DIS_3cm_hdf5_Rendered \
        --model_path="$trained_model"
        ;;
    "scannet")
        echo "test segmentation for scannet scenes"
        CUDA_VISIBLE_DEVICES="$gpuid" python eval/evaluate_scannet_full.py --data_dir=./data/ScanNet_2cm_hdf5_Rendered \
        --model_path="$trained_model"
        ;;
    *)
        echo "Invalid option: UNKNOWN dataset!"
        # Add code for handling invalid options
        ;;
esac

