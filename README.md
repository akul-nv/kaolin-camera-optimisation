# Kaolin Camera Optimisation using Differential Rendering



## Getting started

This guide helps you set up a conda environment with Kaolin and NVDiffrast dependencies.

## Prerequisites
- CUDA-compatible GPU 
- CUDA toolkit
- Compatible GPU drivers
- Anaconda/Miniconda installed

## Installation Steps

1. Create and activate a new conda environment with Python 3.10:
```bash
conda create -n kaolin python=3.10
conda activate kaolin
```

2. Install PyTorch:
```bash
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
```

3. Install Kaolin:
```bash
pip install kaolin==0.16.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html
```

4. Install NVDiffrast
```bash
git clone https://github.com/NVlabs/nvdiffrast
cd nvdiffrast
pip install .
```

## Usage
```bash
python optimize.py
```

