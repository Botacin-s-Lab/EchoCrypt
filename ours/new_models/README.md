# VT Model Training Directory
This directory contains scripts and configurations for training Vision Transformer (VT) models, specifically focusing on training models for phone and zoom datasets. The directory includes data preprocessing, model training, and evaluation code for both phone and zoom-based audio datasets.

## Files in this Directory
### 1. `models_phone.py`
A Python script that handles the training of Vision Transformer models for phone datasets. It supports multiple model architectures (ViT, Swin, Swinv2, DeiT, Beit, and CLIP). The script performs data preprocessing, splits the dataset into training, validation, and test sets, and trains a model using specified configurations.

### 2. `models_zoom.py`
Similar to `models_phone.py`, this script is responsible for training Vision Transformer models for the Zoom dataset. It also supports the same range of model architectures and performs similar preprocessing, dataset splitting, and training tasks as `models_phone.py`.

### 3. `phone_same_trans.py`
A script for preprocessing and augmenting the phone dataset before training. It includes time-shifting and spectral augmentation techniques for audio data. The dataset is transformed into a mel-spectrogram format, suitable for training the models defined in `models_phone.py`.

### 4. `zoom_same_trans.py`
A script for preprocessing and augmenting the Zoom dataset before training. Similar to `phone_same_trans.py`, it includes time-shifting and spectral augmentation techniques, and transforms the dataset into a mel-spectrogram format for training.

### 5. `run_all.bat` (Windows batch file)
A batch file that automates the training process for the VT models. It runs the necessary scripts to preprocess data, train models, and evaluate their performance on both the phone and Zoom datasets.

### 6. `run_all_st.bat` (Windows batch file)
A second batch file for running a specialized version of the training process, tailored for specific requirements or modifications in the training pipeline.

## Setup and Workflow
### Step 1: Preprocessing and Dataset Transformation
For both the **phone** and **zoom** datasets, you need to preprocess and augment the data. The scripts `phone_same_trans.py` and `zoom_same_trans.py` provide transformations like time-shifting and spectral augmentation to improve the quality of the dataset before training. These scripts convert the raw audio into mel-spectrograms.

- **Run the preprocessing script**: 
  - For phone dataset: `python phone_same_trans.py`
  - For zoom dataset: `python zoom_same_trans.py`

### Step 2: Model Training
The training is handled by `models_phone.py` and `models_zoom.py`. You can select a specific model architecture (ViT, Swin, Swinv2, DeiT, Beit, or CLIP) by passing it as an argument when running the scripts.

- **To train a model**:
  - For the phone dataset: `python models_phone.py --model vit --seed st_seed0`
  - For the zoom dataset: `python models_zoom.py --model swin --seed st_seed1`

These scripts will:
1. Load the dataset.
2. Split it into training, validation, and test sets.
3. Train the selected model using the training data.
4. Save the best model during training based on validation accuracy.

### Step 3: Evaluation
After training the models, you can evaluate their performance by running the evaluation sections within the training scripts. These scripts will generate classification reports and confusion matrices for further analysis.

### Step 4: Running Everything Automatically
To automate the entire process, you can use the batch files `run_all.bat` or `run_all_st.bat`, depending on your needs. These batch files will sequentially execute the necessary steps for both datasets, including preprocessing, training, and evaluation.

### Directory Structure
```
/home/ali/EchoCrypt/ours/models_training
├── README.md        # this file
├── models_phone.py  # phone dataset model training
├── models_zoom.py   # zoom dataset model training
├── phone_same_trans.py  # preprocessing and transformation for phone dataset
├── zoom_same_trans.py   # preprocessing and transformation for zoom dataset
├── run_all.bat      # Windows batch file to run all steps for phone and zoom models
└── run_all_st.bat   # Windows batch file for specialized training
```

## Model Selection
You can choose from several transformer architectures:

- **ViT (Vision Transformer)**
- **Swin Transformer**
- **SwinV2 Transformer**
- **DeiT (Distilled Vision Transformer)**
- **Beit (Bidirectional Encoder Representations from Transformers)**
- **CLIP (Contrastive Language-Image Pretraining)**

Use the `--model` argument to specify which model to train.
