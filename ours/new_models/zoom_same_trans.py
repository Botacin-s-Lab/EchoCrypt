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
from torchvision.transforms import RandomCrop
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

keys_s = '0123456789abcdefghijklmnopqrstuvwxyz'
labels = list(keys_s)
keys = [k + '.wav' for k in labels]
data_dict = {'Key':[], 'File':[]}

def isolator(signal, sample_rate, size, scan, before, after, threshold, show=False):
    strokes = []
    # -- signal'
    if show:
        plt.figure(figsize=(7, 2))
        librosa.display.waveshow(signal, sr=sample_rate)
    fft = librosa.stft(signal, n_fft=size, hop_length=scan)
    energy = np.abs(np.sum(fft, axis=0)).astype(float)
    # norm = np.linalg.norm(energy)
    # energy = energy/norm
    # -- energy'
    if show:
        plt.figure(figsize=(7, 2))
        librosa.display.waveshow(energy)
    threshed = energy > threshold
    # -- peaks'
    if show:
        plt.figure(figsize=(7, 2))
        librosa.display.waveshow(threshed.astype(float))
    peaks = np.where(threshed == True)[0]
    peak_count = len(peaks)
    prev_end = sample_rate*0.1*(-1)
    # '-- isolating keystrokes'
    for i in range(peak_count):
        this_peak = peaks[i]
        timestamp = (this_peak*scan) + size//2
        if timestamp > prev_end + (0.1*sample_rate):
            keystroke = signal[timestamp-before:timestamp+after]
            strokes.append(torch.tensor(keystroke)[None, :])
            if show:
                plt.figure(figsize=(7, 2))
                librosa.display.waveshow(keystroke, sr=sample_rate)
            prev_end = timestamp+after
    return strokes

def convert_to_df(AUDIO_FILE):
    for i, File in enumerate(keys):
        loc = AUDIO_FILE + File
        samples, sample_rate = librosa.load(loc, sr=None)
        #samples = samples[round(1*sample_rate):]
        strokes = []
        prom = 0.06
        step = 0.005
        while not len(strokes) == 25:
            strokes = isolator(samples[1*sample_rate:], sample_rate, 48, 24, 2400, 12000, prom, False)
            if len(strokes) < 25:
                prom -= step
            if len(strokes) > 25:
                prom += step
            if prom <= 0:
                print('-- not possible for: ',File)
                break
            step = step*0.99
        label = [labels[i]]*len(strokes)
        data_dict['Key'] += label
        data_dict['File'] += strokes

    df = pd.DataFrame(data_dict)
    mapper = {}
    counter = 0
    for l in df['Key']:
        if not l in mapper:
            mapper[l] = counter
            counter += 1
    df.replace({'Key': mapper}, inplace=True)
    return df, sample_rate 


data_frame, sr = convert_to_df("../../dataset/Zoom/")

print(data_frame.head())
print(data_frame.info())


train_set, tmp_set = train_test_split(data_frame, test_size=0.3, stratify=data_frame['Key'])
val_set, test_set = train_test_split(tmp_set, test_size=0.33, stratify=tmp_set['Key'])

print("Sample rate:", sr)
print(len(train_set), len(val_set), len(test_set))

class MyDataset(Dataset):
    def __init__(self, file_name, transform = None, aug = None):
        df = file_name
        self.transform = transform
        self.aug = aug
        self.labels = df['Key']
        self.values = df['File']
        
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index):
        label = self.labels.iloc[index]
        value = self.values.iloc[index]
        waveform = self.values.iloc[index]
        label = self.labels.iloc[index]
        if self.transform:
            waveform = waveform.numpy()
            waveform = waveform.flatten()
            waveform = self.transform(waveform)
        if self.aug:
            waveform = self.aug(waveform)
        return waveform, label

class ToMelSpectrogram:
    def __call__(self, samples):
#         samples = np.array(samples)
        spec = librosa.feature.melspectrogram(y = samples, sr = sr, n_mels=224, n_fft=2048, win_length=1024, hop_length=64)
        return librosa.power_to_db(spec)

class TimeShifting():
    def __call__(self, samples):
        samples = samples.numpy()
        samples = samples.flatten()
        
        shift = int(len(samples) * 0.4)
        random_shift =random.randint(0, shift)
        data_roll = np.roll(samples, random_shift)
        return data_roll
    
class SpecAugment():
    def __call__(self, samples):
        num_mask = 2
        freq_masking_max_percentage=0.10
        time_masking_max_percentage=0.10
        spec = samples.copy()
        mean_value = spec.mean()
        for i in range(num_mask):
            all_frames_num, all_freqs_num = spec.shape[1], spec.shape[1] 
            freq_percentage = random.uniform(0.0, freq_masking_max_percentage)

            num_freqs_to_mask = int(freq_percentage * all_freqs_num)
            f0 = np.random.uniform(low=0.0, high=all_freqs_num - num_freqs_to_mask)
            f0 = int(f0)
            spec[:, f0:f0 + num_freqs_to_mask] = mean_value

            time_percentage = random.uniform(0.0, time_masking_max_percentage)

            num_frames_to_mask = int(time_percentage * all_frames_num)
            t0 = np.random.uniform(low=0.0, high=all_frames_num - num_frames_to_mask)
            t0 = int(t0)
            spec[t0:t0 + num_frames_to_mask, :] = mean_value
        return spec

# Define the dataset path
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    choices=['vit', 'swin', 'swinv2', 'deit', 'beit', 'clip'],
    required=True,
    help="Choose the model: 'vit', 'swin', 'swinv2', 'deit', 'beit', or 'clip'"
)
args = parser.parse_args()
selected_model = args.model

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


aug_st = Compose([
    TimeShifting(),
    ToMelSpectrogram(),
    SpecAugment(),
    ToTensor(),
    transforms.Lambda(lambda x: x.expand(3, -1, -1)),
    RandomCrop(224),
    transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
    ])

transform_st = Compose([
    ToMelSpectrogram(),
    ToTensor(),
    transforms.Lambda(lambda x: x.expand(3, -1, -1)),
    RandomCrop(224),
    transforms.Normalize(mean=processor.image_mean, std=processor.image_std)])
        
train_set = MyDataset(train_set, aug = aug_st)
val_set = MyDataset(val_set, transform = transform_st)
test_set = MyDataset(test_set, transform = transform_st)

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
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
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
        torch.save(model.state_dict(), f'best_model_zoom_st_{selected_model}.pth')
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
model.load_state_dict(torch.load(f'best_model_zoom_st_{selected_model}.pth'))
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
with open(f"classification_report_zoom_st_{selected_model}.txt", "w") as file:
    file.write(classification_report(all_labels, all_preds, digits=4))