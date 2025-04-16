# Finetuning LLMs
This directory contains the necessary scripts and notebooks for preparing the dataset, training, and testing LoRA models for finetuning on our dataset.

## Overview
The process starts by formatting dataset in a CSV file (see `ours/zoom/results/` and `ours/phone/results/`) into a dataset Arrow file. This Arrow file is then used for training and testing LoRA models. The following instructions guide you through the steps for preparing data, training, and running inference with the model.

## Files in this Directory
This directory contains the following files:

### 1. `README.md`
This file, which provides an overview of the directory structure and usage instructions.

### 2. `format_convos.py`
A Python script used to format conversations in a CSV file into an Arrow file. This Arrow file is required for both training and testing the model.

### 3. `phone.ipynb`
A Jupyter Notebook for finetuning the model on the phone conversation dataset.

### 4. `zoom.ipynb`
A Jupyter Notebook for finetuning the model on the zoom conversation dataset.

## Setup and Workflow
### Step 1: Format Conversations
Before training, you need to format your conversation data. To do this:

1. Open the `format_convos.py` script.
2. **Line 6**: Replace with the input CSV file containing the your data. The CSV should have two columns: 
   - **True Sentence**: The ground truth sentence.
   - **Predicted Sentence**: The sentence with typos.
3. **Line 80**: Specify the output folder and name for the resulting Arrow file.

Running this script will generate a folder containing the formatted data.

### Step 2: Preparing the Data
Place the generated Arrow files in the same directory as the Jupyter notebooks (`phone.ipynb` or `zoom.ipynb`). These notebooks will load the Arrow files for further processing and training.

### Step 3: Training the Model
To train the model, choose one of the following options:

- **Training from Scratch (LoRA)**: 
  In the notebook, find the section labeled **Training from scratch (LoRA)** and run the blocks step by step.

- **Training from Pretrained LoRA**: 
  If you're using a pretrained model, navigate to the section labeled **Training from pretrained LoRA**. Here, you can either:
  - Load the pretrained model from online weights by running the block **Load pretrained from uploaded weights**.
  - Alternatively, load a locally stored pretrained model if you have newer fine-tuned weights.

### Step 4: Data Processing
Once the model is ready, process the data by running the appropriate blocks in the notebook. The print statements will help you verify that the dataset is correctly loaded and processed. 

### Step 5: Model Training
Run the blocks for training, following the steps one by one. The training process will fine-tune the LoRA model on your dataset.

### Step 6: Save the Model
After training, save the model. The saved model is stored in a folder. You can upload this folder to Hugging Face for easy access and sharing. We will use the fine-tuned model in `ours/phone/` and `ours/zoom/` for inference.

## Directory Structure

```
/home/ali/EchoCrypt/ours/finetune
├── README.md        # this file
├── format_convos.py  # prepares the data for finetuning
├── main.ipynb      # finetuning codes
```
