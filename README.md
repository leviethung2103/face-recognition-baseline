# Face Recognition Baseline

This repo is a starting point for using Face Recognition Library

## Dependencies

* Python>=3.6
* CUDA=10.0
* MXNet

### Installation 

Make sure `conda` is installed. 

**Note that**: If you are using GPU, you need to change the version of cuda  in `requirements.txt` file. 

*`mxnet-cu101`* means the package is built with CUDA/cuDNN and the CUDA version is 10.1.

```bash
# create environment 
conda create --name face_recog_test python=3.6
conda activate face_recog_test

# install dependencies for cpu 
pip install -r requirements-cpu.txt
# or install dependencies for gpu 
pip install -r requirements-gpu.txt

# install face recognition library
pip install --upgrade greenlab-library
```



## Getting Started

**Steps:**

1. Change the image path in and parameters in `configs/server_api.yaml` file. 
2. Run `main.py` file
3. Check the results at `logs/outputs/test.png` 
