from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

current_file_path = os.path.abspath(__file__)

setup(
    name='point_buildkernel',
    ext_modules=[
        CUDAExtension('point_buildkernel', [
            f'{current_file_path}/buildkernel.cpp',
            f'{current_file_path}/buildkernel_gpu.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

setup(
    name='point_conv3d',
    ext_modules=[
        CUDAExtension('point_conv3d', [
            f'{current_file_path}/pcloud_conv3d.cpp',
            f'{current_file_path}/pcloud_conv3d_gpu.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

setup(
    name='point_nnquery',
    ext_modules=[
        CUDAExtension('point_nnquery', [
            f'{current_file_path}/nnquery.cpp',
            f'{current_file_path}/nnquery_gpu.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

setup(
    name='point_pool3d',
    ext_modules=[
        CUDAExtension('point_pool3d', [
            f'{current_file_path}/pcloud_pool3d.cpp',
            f'{current_file_path}/pcloud_pool3d_gpu.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

setup(
    name='point_sample',
    ext_modules=[
        CUDAExtension('point_sample', [
            f'{current_file_path}/sample.cpp',
            f'{current_file_path}/sample_gpu.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

setup(
    name='point_unpool3d',
    ext_modules=[
        CUDAExtension('point_unpool3d', [
            f'{current_file_path}/pcloud_unpool3d.cpp',
            f'{current_file_path}/pcloud_unpool3d_gpu.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })