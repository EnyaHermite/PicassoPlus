from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

mesh_path = 'picasso/mesh/pi_modules/source'
point_path = 'picasso/point/pi_modules/source'

setup(
    name='mesh_conv3d',
    ext_modules=[
        CUDAExtension('mesh_conv3d', [
            f'{mesh_path}/mesh_conv3d.cpp',
            f'{mesh_path}/mesh_conv3d_gpu.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

setup(
    name='mesh_decimation',
    ext_modules=[
        CUDAExtension('mesh_decimation', [
            f'{mesh_path}/decimate.cpp',
            f'{mesh_path}/decimate_gpu.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

setup(
    name='mesh_pool3d',
    ext_modules=[
        CUDAExtension('mesh_pool3d', [
            f'{mesh_path}/mesh_pool3d.cpp',
            f'{mesh_path}/mesh_pool3d_gpu.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

setup(
    name='mesh_unpool3d',
    ext_modules=[
        CUDAExtension('mesh_unpool3d', [
            f'{mesh_path}/mesh_unpool3d.cpp',
            f'{mesh_path}/mesh_unpool3d_gpu.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })



setup(
    name='point_buildkernel',
    ext_modules=[
        CUDAExtension('point_buildkernel', [
            f'{point_path}/buildkernel.cpp',
            f'{point_path}/buildkernel_gpu.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

setup(
    name='point_conv3d',
    ext_modules=[
        CUDAExtension('point_conv3d', [
            f'{point_path}/pcloud_conv3d.cpp',
            f'{point_path}/pcloud_conv3d_gpu.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

setup(
    name='point_nnquery',
    ext_modules=[
        CUDAExtension('point_nnquery', [
            f'{point_path}/nnquery.cpp',
            f'{point_path}/nnquery_gpu.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

setup(
    name='point_pool3d',
    ext_modules=[
        CUDAExtension('point_pool3d', [
            f'{point_path}/pcloud_pool3d.cpp',
            f'{point_path}/pcloud_pool3d_gpu.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

setup(
    name='point_sample',
    ext_modules=[
        CUDAExtension('point_sample', [
            f'{point_path}/sample.cpp',
            f'{point_path}/sample_gpu.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

setup(
    name='point_unpool3d',
    ext_modules=[
        CUDAExtension('point_unpool3d', [
            f'{point_path}/pcloud_unpool3d.cpp',
            f'{point_path}/pcloud_unpool3d_gpu.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })