#!/bin/bash

# Activer l'environnement GCC Toolset 14
scl enable gcc-toolset-14 -- bash -c 'export CUDA_PATH=/usr/local/cuda-12.1; exec bash'
