# YOLO26 Retail Product Detection

This project demonstrates how to train and deploy a YOLO26 model for retail shelf object detection using the SKU-110K dataset.

## Features

- Train YOLO26 on custom datasets
- Run inference on shelf images
- Count detected products
- Interactive Gradio web app

## Installation

```bash
git clone https://github.com/smukherjee1do/yolo26-retail-detection
cd yolo26-retail-detection
pip install -r requirements.txt
```

## Train Model

```bash
python src/train.py
```

## Run Inference

```bash
python src/inference.py
```

## Launch Web App

```bash
python app/gradio_app.py
```

## Dataset

[SKU-110K Retail Shelf Dataset](https://github.com/eg4000/SKU110K_CVPR19)
