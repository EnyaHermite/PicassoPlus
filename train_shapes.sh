#!/bin/bash

# Define the variable to be checked
dataset="$1"
gpuid="$2"

# Perform the "switch" logic
case "$dataset" in
    "human")
        echo "train network for human body segmentation"
        CUDA_VISIBLE_DEVICES="$gpuid" python train/train_human.py --data_dir=./data/human_seg
        ;;
    "coseg_aliens")
        echo "train network for coseg_aliens segmentation"
        CUDA_VISIBLE_DEVICES="$gpuid" python train/train_coseg.py --data_dir=./data/coseg_aliens
        ;;
    "coseg_vases")
        echo "train network for coseg_vases segmentation"
        CUDA_VISIBLE_DEVICES="$gpuid" python train/train_coseg.py --data_dir=./data/coseg_vases
        ;;
    "coseg_chairs")
        echo "train network for coseg_chairs segmentation"
        CUDA_VISIBLE_DEVICES="$gpuid" python train/train_coseg.py --data_dir=./data/coseg_chairs
        ;;
    "cubes")
        echo "train network for cubes classification"
        CUDA_VISIBLE_DEVICES="$gpuid" python train/train_cubes.py --data_dir=./data/cubes
        ;;
    "shrec")
        echo "train network for shrec classification"
        CUDA_VISIBLE_DEVICES="$gpuid" python train/train_shrec.py --data_dir=./data/shrec_16
        ;;
    "faust")
        echo "train network for faust correspondence matching"
        CUDA_VISIBLE_DEVICES="$gpuid" python train/train_faust_match.py --data_dir=./data/MPI-Faust/registrations
        ;;
    "shapenet")
        echo  "train network for shapenet classification"
        CUDA_VISIBLE_DEVICES="$gpuid" python train/train_shapenetcore.py --batch_size=64 --data_dir=./data/ShapeNetCore.v2
        ;;
    *)
        echo "Invalid option: UNKNOWN dataset!"
        # Add code for handling invalid options
        ;;
esac
