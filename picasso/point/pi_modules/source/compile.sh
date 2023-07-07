#!/bin/bash

#python setup.py clean --all install
TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9+PTX" python setup.py install
# SM_75 for RTX2080, SM_86 for RTX3090


