# AudioEnhancer
The codebase for a model that improve the quality of an audio.

## Installation
1. Clone the repository
2. Install the required packages
```bash
pip install -r requirements.txt
```

## Download the dataset

To download the dataset, run the following command:
```bash
python -m scripts.download_dataset.py --audio_dir PATH --quantity 1
```

## Build the dataset
To build the dataset, run the following command:
```bash
python -m scripts.build_dataset.py --audio_dir PATH --dataset_dir PATH --codec dac encodec soundstream opus
```

## Train the model
To train the model, run the following command:
``` 
python -m scripts.train.py 
```
