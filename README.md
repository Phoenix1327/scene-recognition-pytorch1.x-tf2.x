# Scene-Recognition-Service-PyTorch-TF2.0

Scene recognition models based on pytorch and TF2.0.

## Install

```bash
conda env create -f environment.yml python=3.7

conda activate scene_pytorch_tf

export PYTHONPATH=$PYTHONPATH:$(pwd)
```

## Model Zoo (Pretrained Models)

Please refer [[Model Zoo]](model_zoo.md)

## Train

We download the data from http://places2.csail.mit.edu/download.html

### PyTorch == 1.x

#### 1. Download the data

These images are 256x256 images, in a more friendly directory structure that in train and val split the images are organized such as train/reception/00003724.jpg and val/raft/000050000.jpg

```python
sh download_data_pytorch.sh
```

#### 2. Train the model with multiple GPUs

```bash
python tools/train.py
```

#### 3. Remove the .module

```python
python scripts/remove_pytorch_module.py
```

#### 4. Test a model

```python
python tools/test.py
```

#### 5. Convert a model to TorchScript

```python
python scripts/convert_torchscript.py
```

### Tensorflow == 2.x

#### 1. Download the data. (Refer: https://github.com/tensorflow/datasets)

```bash
python -m tensorflow_datasets.scripts.download_and_prepare --datasets=places365_small
```

#### - [ ] 2. Train the model with multiple GPUs

#### - [ ] 3. Test models
