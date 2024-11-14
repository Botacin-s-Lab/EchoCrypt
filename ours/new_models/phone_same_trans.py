import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import librosa
import os
import random
import seaborn as sns
import pandas as pd
import pickle
import math
import torchaudio
import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt 
import torchvision.transforms.functional as F
import argparse
import seaborn as sns
import torch

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision import transforms
from transformers import ViTForImageClassification, SwinForImageClassification, Swinv2ForImageClassification, DeiTForImageClassification, BeitForImageClassification
from transformers import CLIPVisionModel, CLIPConfig
from transformers import AutoImageProcessor
from collections import defaultdict
from PIL import Image
from torchaudio.transforms import TimeMasking, FrequencyMasking
from torchvision.ops import SqueezeExcitation
from torchinfo import summary
from torchsummary import summary
from torchvision.ops import SqueezeExcitation
from tqdm import tqdm
from torchvision import datasets, transforms
from collections import defaultdict
from torchvision.transforms import Compose, ToTensor
from torch.utils.data import DataLoader, Dataset
from scipy.io import wavfile
from functools import reduce
from sklearn import metrics 
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

pd.set_option('future.no_silent_downcasting', True)

class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.n_samples = 0
        self.dataset = []
        self.labels = set()  # To track unique labels
        self.load_audio_files(self.data_dir)

    def load_audio_files(self, path: str):
        for dirname, _, filenames in os.walk(path):
            for filename in filenames:
                file_path = os.path.join(dirname, filename)
                
                # label = dirname.split('/')[-1]  # on MAC 
                label = os.path.basename(dirname)   # on Windows
       
                # my implementation start
                if '0' <= label <= '9':
                    label_index = ord(label) - ord('0')
                    # print(label_index)
                elif 'a' <= label <= 'z':
                    label_index = ord(label) - ord('a') + 10
                    # print(label_index)
                else:
                    raise ValueError(f"Unexpected label: {label}")
                    break
                label_tensor = torch.tensor(label_index)
                # my implementation done
                
                # Add the label to the set of unique labels
                self.labels.add(label_tensor.item())
                
                # Load audio
                waveform, sample_rate = torchaudio.load(file_path)
                if self.transform is not None:
                    waveform_transformed = self.transform(waveform)
                
                if waveform_transformed.shape[2] != 224:
                    print("Wrong shape:", waveform_transformed.shape)
                    continue
                
                self.n_samples += 1
                self.dataset.append((waveform, label_tensor))

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        waveform, label = self.dataset[idx]
        return waveform, label

    def num_classes(self):
        return len(self.labels)  # Return the number of unique labels
    
sample_rate = 44100
to_mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate, n_mels=224, hop_length=85, n_fft=2048, win_length=1024)
mel_spectrogram_to_numpy = lambda spectrogram: spectrogram.log2()[0,:,:].numpy()
transforms_ = Compose([to_mel_spectrogram, mel_spectrogram_to_numpy, ToTensor(), transforms.Lambda(lambda x: x.expand(3, -1, -1)),])
dataset = AudioDataset('../../new_dataset_phone', transforms_)
print("number of classes:", dataset.num_classes())

# Assume your dataset has a 'targets' attribute or you can extract labels from it
targets = [data[1] for data in dataset]  # Assuming dataset returns (data, label) pairs

# Split the dataset indices with stratification
train_indices, tmp_indices = train_test_split(
    range(len(dataset)), 
    test_size=0.3,  # 30% of the data goes to val+test
    stratify=targets
)

val_indices, test_indices = train_test_split(
    tmp_indices, 
    test_size=0.33,  # 33% of the 30% goes to the test set, i.e., 10% of the original dataset
    stratify=[targets[i] for i in tmp_indices]
)

# Create subsets of the dataset based on the indices
init_train_set = torch.utils.data.Subset(dataset, train_indices)
init_val_set = torch.utils.data.Subset(dataset, val_indices)
init_test_set = torch.utils.data.Subset(dataset, test_indices)

# Print the sizes for verification
print("Sample rate:", sample_rate)
print(f"Train set size: {len(init_train_set)}, Validation set size: {len(init_val_set)}, Test set size: {len(init_test_set)}")

class TrainingDataset(Dataset):
    def __init__(self, base_dataset, transformations):
        super(TrainingDataset, self).__init__()
        self.base = base_dataset
        self.transformations = transformations

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        waveform, label = self.base[idx]
        return self.transformations(waveform), label
    
class TimeShifting():
    def __call__(self, samples):
        samples = samples.numpy()        
        shift = int(samples.shape[1] * 0.3)
        random_shift = random.randint(0, shift)
        data_roll = np.zeros_like(samples)
        data_roll[0] = np.roll(samples[0], random_shift)
        data_roll[1] = np.roll(samples[1], random_shift)
        return torch.tensor(data_roll)

# Define the dataset path
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    choices=['vit', 'swin', 'swinv2', 'deit', 'beit', 'clip'],
    required=True,
    help="Choose the model: 'vit', 'swin', 'swinv2', 'deit', 'beit', or 'clip'"
)
parser.add_argument(
    "--seed",
    type=str,
    required=True,
    help="e.g., --seed=st_seed0"
)
args = parser.parse_args()
selected_model = args.model
seed_num = args.seed

# define models
if selected_model == 'swin':
    model_name = 'microsoft/swin-tiny-patch4-window7-224'
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = SwinForImageClassification.from_pretrained(
        model_name,
        num_labels=36,
        ignore_mismatched_sizes=True  
    )
elif selected_model == 'swinv2':
    model_name = 'microsoft/swinv2-tiny-patch4-window16-256'
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = Swinv2ForImageClassification.from_pretrained(
        model_name,
        num_labels=36,
        ignore_mismatched_sizes=True  
    )
elif selected_model == 'vit':
    model_name = 'google/vit-base-patch16-224-in21k'
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(
        model_name,
        num_labels=36,
        ignore_mismatched_sizes=True,
    )
elif selected_model == 'deit':
    model_name = 'facebook/deit-base-distilled-patch16-224'
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = DeiTForImageClassification.from_pretrained(
        model_name,
        num_labels=36,
        ignore_mismatched_sizes=True
    )
elif selected_model == 'beit':
    model_name = 'microsoft/beit-base-patch16-224-pt22k-ft22k'  
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = BeitForImageClassification.from_pretrained(
        model_name,
        num_labels=36,
        ignore_mismatched_sizes=True  # Add this if necessary
    )
elif selected_model == 'clip':
    selected_model = 'clip'
    model_name = 'openai/clip-vit-base-patch32'
    processor = AutoImageProcessor.from_pretrained(model_name)
    vision_model = CLIPVisionModel.from_pretrained(model_name)
    num_labels = 36
    config = CLIPConfig.from_pretrained(model_name)
    config.num_labels = num_labels
    
    class CLIPVisionForCustomImageClassification(torch.nn.Module):
        def __init__(self, vision_model, num_labels):
            super().__init__()
            self.vision_model = vision_model
            self.classifier = torch.nn.Linear(vision_model.config.hidden_size, num_labels)
        
        def forward(self, pixel_values):
            outputs = self.vision_model(pixel_values=pixel_values)
            pooled_output = outputs.pooler_output  # [batch_size, hidden_size]
            logits = self.classifier(pooled_output)
            return logits
    model = CLIPVisionForCustomImageClassification(vision_model, num_labels)
else:
    raise ValueError('[ERROR] Select Your Model')


# model parameter count
if selected_model in ('vit', 'swin', 'deit', 'beit', 'swinv2'):
    param_count = model.num_parameters()
    print(f"Total number of parameters for {selected_model} is {param_count}")
elif selected_model == 'clip':
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")


aug_transforms_st = Compose([
    TimeShifting(),
    to_mel_spectrogram, mel_spectrogram_to_numpy, 
    ToTensor(),
    transforms.Lambda(lambda x: x.expand(3, -1, -1)),
    FrequencyMasking(7),
    TimeMasking(7),
    FrequencyMasking(7),
    TimeMasking(7),
    transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
])
transforms_st = Compose([to_mel_spectrogram, mel_spectrogram_to_numpy, 
                         ToTensor(), 
                         transforms.Lambda(lambda x: x.expand(3, -1, -1)),
                         transforms.Normalize(mean=processor.image_mean, std=processor.image_std)])

train_set = TrainingDataset(init_train_set, aug_transforms_st)
val_set = TrainingDataset(init_val_set, transforms_st)
test_set = TrainingDataset(init_test_set, transforms_st)

train_dataloader = torch.utils.data.DataLoader(
    train_set,
    batch_size=16,
    shuffle=True
)

val_dataloader = torch.utils.data.DataLoader(
    val_set,
    batch_size=16,
    shuffle=True
)
test_dataloader = torch.utils.data.DataLoader(
    test_set,
    batch_size=16,
    shuffle=True
)


# Move the model to the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm  # For progress bars

# Set up the optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
criterion = torch.nn.CrossEntropyLoss()

# Learning rate scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.1, verbose=True)

# Early stopping parameters
best_val_accuracy = 0.0
patience = 25  # Number of epochs to wait before early stopping
epochs_no_improve = 0

# Training loop
num_epochs = 1000

def get_outputs(model, inputs):
    if selected_model == 'clip':
        outputs = model(pixel_values=inputs)
    else:
        outputs = model(inputs)
    # Now outputs is defined; proceed to check its type
    if isinstance(outputs, torch.Tensor):
        return outputs
    elif hasattr(outputs, 'logits'):
        return outputs.logits
    else:
        raise ValueError("Model output format not recognized.")
    
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    # Training
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    
    for batch in tqdm(train_dataloader, desc="Training", leave=False):
        inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = get_outputs(model, inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    train_loss = train_loss / total
    train_accuracy = correct / total
    
    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Validation", leave=False):
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = get_outputs(model, inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = val_loss / total
    val_accuracy = correct / total
    
    print(f'Epoch {epoch+1}/{num_epochs}, '
          f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, '
          f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')
    
    # Step the scheduler
    scheduler.step(val_accuracy)
    
    # Check for improvement
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        epochs_no_improve = 0
        # Save the best model
        torch.save(model.state_dict(), f'{seed_num}/best_model_phone_st_{selected_model}.pth')
        print("Validation accuracy improved, model saved.")
    else:
        epochs_no_improve += 1
        print(f"No improvement in validation accuracy for {epochs_no_improve} epochs.")
    
    # Early stopping
    if epochs_no_improve >= patience:
        print("Early stopping triggered.")
        break

    # Testing the model and generating a classification report
from sklearn.metrics import classification_report, confusion_matrix

# Load the best model 
model.load_state_dict(torch.load(f'{seed_num}/best_model_phone_st_{selected_model}.pth'))
class_labels = [str(i) for i in range(10)] + [chr(i) for i in range(ord('a'), ord('z') + 1)]

# Collect all predictions and labels
all_preds = []
all_labels = []

model.eval()
with torch.no_grad():
    for batch in test_dataloader:
        inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = get_outputs(model, inputs)
        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Classification Report
with open(f"{seed_num}/classification_report_phone_st_{selected_model}.txt", "w") as file:
    file.write(classification_report(all_labels, all_preds, digits=4))