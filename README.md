<p align="center" style="font-size: 50px; font-weight: bold;">
Mesh Convolution With Continuous Filters for 3-D Surface Parsing
</p>

![alt text](https://github.com/EnyaHermite/PicassoPlus/blob/main/image/teaser.png)

### Introduction
Geometric feature learning for 3-D surfaces is critical for many applications in computer graphics and 3-D vision. 
However, deep learning currently lags in hierarchical modeling of 3-D surfaces due to the lack of required operations 
and/or their efficient implementations. [This journal work](https://arxiv.org/abs/2112.01801) is a sigificant 
extension of [our original work](https://arxiv.org/abs/2103.15076) 
presented in CVPR 2021. Together, we provide PicassoPlus for deep learning over heterogeneous 3D meshes. 
We propose a series of modular operations for effective geometric 
feature learning from 3-D triangle meshes. These operations include novel mesh convolutions, efficient mesh decimation, 
and associated mesh (un)poolings. Our mesh convolutions exploit spherical harmonics as orthonormal bases to create 
continuous convolutional filters. The mesh decimation module is GPU-accelerated and able 
to process batched meshes on-the-fly, while the (un)pooling operations compute features for upsampled/downsampled meshes. 
Leveraging the modular operations of PicassoPlus, we further contribute a neural network, PicassoNet++, for 3-D surface parsing. It achieves highly competitive performance for shape analysis and scene segmentation on prominent 3-D benchmarks. 

*Note: PicassoPlus has been moved to Pytorch. We no longer provide tensorflow support.* 


### Citation
If you find our work useful in your research, please consider citing:

```
@article{lei2023mesh,
  title={Mesh Convolution With Continuous Filters for 3-D Surface Parsing},
  author={Lei, Huan and Akhtar, Naveed and Shah, Mubarak and Mian, Ajmal},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2023},
  publisher={IEEE}
}
```
```
@inproceedings{lei2021picasso,
  title={Picasso: A CUDA-based Library for Deep Learning over 3D Meshes},
  author={Lei, Huan and Akhtar, Naveed and Mian, Ajmal},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={13854--13864},
  year={2021}
} 
```
Please also cite the original dataset if you use their data or our reprocessed data, and follow their original terms of use.



### Pytorch Environment 
#### 1.  Installation of Picasso
- Install [Pytorch](https://pytorch.org/get-started/locally/) and the other dependences in requirements.txt. For compatibility, we recommend install [torch-scatter](https://github.com/rusty1s/pytorch_scatter) from source. Then install picasso:
  ```
  $ git clone https://github.com/EnyaHermite/PicassoPlus
  $ bash install_picasso.sh
  ```
- The code has been test on GPU 3090, 4090 for multiple versions of python/pytorch/cuda/cudnn/ubuntu. **Note: We assumed that the GPU supports a block of 1024 threads.** 
  

#### 2. Data Preparation
- Download the shared data of MeshCNN using download_extract_shapes.py, and S3DIS dataset using download_extract_scenes.sh. 
  ```
  $ cd data
  $ python download_extract_shapes.py
  $ python download_extract_scenes.sh   
  $ cd ../preprocess
  $ python process_coseg_label.py
  $ python process_cubes_shrec_label.py
  $ python process_human_label.py
  ```
- The MPI-Fasut dataset can be downloaded manually from [here](https://faust-leaderboard.is.tuebingen.mpg.de/). After extracting its compressed file, run the following command from /preprocess folder to prepare its labels:
  
  `$ python prepare_faust_label.py`


- Sorry, we cannot provide the scannet data. The remeshed ShapeNetCore data might be shared to the public after communication with the original organizers.
  
#### 3. Usage of PicassoNet++
  - *Shape Analysis*
   
    * The training script takes $(dataset name) and $(gpu id) as input arguments. The supported dataset names are ['shrec', 'cubes', 'human', 'coseg_aliens', 'coseg_chairs', 'coseg_vases']. For example, to train the coseg_vases dataset on gpu 0, run the following command in the root folder:
    
      `$ bash train_shapes.sh coseg_vases 0`


- *Scene Segmentation*    
  * train  
    `$ bash train_scenes.sh s3dis 0`  
  * test   
    `$ bash eval_scenes.sh s3dis 0 /path/to/trained_model`

#### 4. How to design arbitrary mesh neural networks in PicassoPlus?



