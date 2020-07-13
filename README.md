# Face Recognition Baseline

This repo is a starting point for using Face Recognition Library

## Dependencies

* python>=3.6
* Cuda=10.0
* Mxnet-gpu

### Installation 

Make sure `conda` is installed. 

```bash
# create environment 
conda create --name face_recog_test python=3.6
conda activate face_recog_test
# install dependencies
pip install -r requirements.txt
# install face recognition library
pip install --upgrade green-face-recognition
```

## Getting Started
Change the file path in `main.py` file.  

At the first time, the system will try to download the pre-trained models. It may causes an error. Just simply run again. 

```
python main.py 
```

Check the results at `logs/outputs/test.png` 