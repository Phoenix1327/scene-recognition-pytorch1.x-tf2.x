# Scene-Recognition-Service-PyTorch-TF2.0

Scene recognition models based on pytorch and TF2.0

## Install

```bash
conda env create -f environment.yml python=3.7

conda activate scene_pytorch_tf

export PYTHONPATH=$PYTHONPATH:$(pwd)
```

## Model Zoo (Pretrained Models)

Please refer [[Model Zoo]](#model_zoo)

## Train

### PyTorch == 1.x

1. Download the data

```bash
sh download_data_pytorch.sh
```

```bash
python tools/train.py
```

### Tensorflow == 2.x

1. Download the data. (Refer: https://github.com/tensorflow/datasets)

```bash
python -m tensorflow_datasets.scripts.download_and_prepare --datasets=places365_small
```

- [ ] to do

## Test

## Deploy

We first tranform the models from research to production. Then deploy it as a service.
