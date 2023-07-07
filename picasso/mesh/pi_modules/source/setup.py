from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

current_file_path = os.path.abspath(__file__)


setup(
    name='mesh_conv3d',
    ext_modules=[
        CUDAExtension('mesh_conv3d', [
            f'{current_file_path}/mesh_conv3d.cpp',
            f'{current_file_path}/mesh_conv3d_gpu.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

setup(
    name='mesh_decimation',
    ext_modules=[
        CUDAExtension('mesh_decimation', [
            f'{current_file_path}/decimate.cpp',
            f'{current_file_path}/decimate_gpu.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

setup(
    name='mesh_pool3d',
    ext_modules=[
        CUDAExtension('mesh_pool3d', [
            f'{current_file_path}/mesh_pool3d.cpp',
            f'{current_file_path}/mesh_pool3d_gpu.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

setup(
    name='mesh_unpool3d',
    ext_modules=[
        CUDAExtension('mesh_unpool3d', [
            f'{current_file_path}/mesh_unpool3d.cpp',
            f'{current_file_path}/mesh_unpool3d_gpu.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

