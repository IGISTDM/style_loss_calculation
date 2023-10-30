# loss_cal

This repo calculates vgg loss and clip score loss of sets of images. Loss function written by [cseslowpoke](https://github.com/cseslowpoke)

## env

```bash
conda create -f environment.yml
```

if you encounter issues with clip, delete "- clip==1.0" in environment.yml, after succcess installing and activating the environment zecon use the following code to install it manually.

```bash
pip install git+https://github.com/openai/CLIP.git
```

## usage

1. Use main.py to generate loss data (it takes too long to get the loss data of so many images, so graph generation is splitted)
2. Then use graph.py to generate mean and standard deviation graph

```bash
python main.py
python graph.py

```
