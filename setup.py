import os.path as osp
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = osp.dirname(osp.abspath(__file__))

setup(
    name='leapvo',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension('cuda_corr',
            sources=['main/backend/altcorr/correlation.cpp', 'main/backend/altcorr/correlation_kernel.cu'],
            extra_compile_args={
                'cxx':  ['-O3'], 
                'nvcc': ['-O3'],
            }),
        CUDAExtension('lietorch_backends', 
            include_dirs=[
                osp.join(ROOT, 'main/backend/lietorch/include'), 
                osp.join(ROOT, 'thirdparty/eigen-3.4.0')],
            sources=[
                'main/backend/lietorch/src/lietorch.cpp', 
                'main/backend/lietorch/src/lietorch_gpu.cu',
                'main/backend/lietorch/src/lietorch_cpu.cpp'],
            extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3'],}),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })