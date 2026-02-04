# Acoustic-Signal-Classification
This project Aims to train a foundational CNN to classify acoustic signals generated/propagated in oceans and water bodies, to identify the type of vessel. This can work as a benchmark for other improvement efforts.

## Data
The data has been taken from DeepShip repository: (https://github.com/irfankamboh/DeepShip). It contains real world underwater recordings and metadata of 265 different ships belong
to four classes. The recordings have natural background noise, marine mammal sounds as well as Anthropogenic sounds.

A more extensive dataset is available on google drive locations provided here: (https://github.com/irfankamboh/DeepShip/issues/1). 

### Preprocessing
The data has been labeled and appended into a materdata. Deleted duplicate ID rows from tanker-metafile.

## Cuda Setup

To enable GPU acceleration, you must install the CUDA-enabled version of PyTorch.
Run the following command in your virtual environment:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

To verify the installation:
```bash
python Test/check_gpu.py
```



