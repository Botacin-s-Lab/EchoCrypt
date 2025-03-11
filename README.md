# EchoCrypt
TODO: Fix this file

## Experiments and Results

### Reference
Optimizer: Adam.
The following are all test accuracies.
Zoom Model: 98%
Phone Model: 97%
Zoom Model on Phone Dataset: 3%
Phone Model on Zoom Dataset: 2%
Both Model: 97%
Both Models on Phone Dataset: 99%
Both Models on Zoom Dataset: 98%

### Baselines

5 models
we test each raw model 0% 30%
fine-tune
we test fine-tuned model (how much improvement) 


craft sentences --> use CoAtNet 24m --> 70%
1000 senteces. --> randomly concate keystrokes --> sentence wav
sentence wav --> isolator --> char wav --> CoAtNet --> recover char. e.g. good morning (goad morning)  diff 1

craft sentences --> use CoAtNet + LLM --> 90%


### Code Snippets
Converting the zoom dataset into new_dataset_zoom:
```python
import torchaudio

for index, row in data_frame.iterrows():
    key = row['Key']
    label = keys_s[key]
    audio = row['File']
    
    target_dir = f'../new_dataset_zoom/{label}/'
    os.makedirs(target_dir, exist_ok=True)
    
    wav_files = [f for f in os.listdir(target_dir) if f.endswith('.wav')]
    num_wav_files = len(wav_files)
    
    file_name = f"{num_wav_files}.wav"
    file_path = os.path.join(target_dir, file_name)
    assert not os.path.exists(file_path), f"File {file_path} already exists"
    
    torchaudio.save(file_path, audio, sr)
```

Converting the dataset into img_dataset:
```python
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import ConcatDataset

# Concatenate the three datasets
combined_dataset = ConcatDataset([train_set_no_aug, val_set, test_set])
for item in combined_dataset:
    tensor, label_index = item

    if 0 <= label_index <= 9:
        label = chr(label_index + ord('0'))
    elif 10 <= label_index <= 35:
        label = chr(label_index - 10 + ord('a'))
    else:
        raise ValueError(f"Unexpected label index: {label_index}")

    target_dir = f"../img_dataset_phone/{label}/"
    os.makedirs(target_dir, exist_ok=True)
    jpg_files = [f for f in os.listdir(target_dir) if f.endswith('.jpg')]
    num_jpg_files = len(jpg_files)
    file_name = f"{num_jpg_files}.jpg"
    file_path = os.path.join(target_dir, file_name)
    assert not os.path.exists(file_path)
    
    # Convert to PIL Image
    # First, ensure values are in range [0, 1] or [0, 255]
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())  # Normalize to [0, 1]
    tensor = tensor * 255  # Scale to [0, 255]
    
    # Convert from [1, 64, 64] to [64, 64]
    tensor = tensor.squeeze(0)
    
    # Convert to PIL Image
    transform = transforms.ToPILImage()
    image = transform(tensor.byte())

    # Save the image as a JPG file
    image.save(file_path)
```

### Number of Parameters 
- ViT (huggingface google/vit-base-patch16-224-in21k): 85,826,340 (86M)
- BeiT (huggingface microsoft/beit-base-patch16-224-pt22k-ft22k): 85,789,668 (86M)
- DeiT (huggingface facebook/deit-base-distilled-patch16-224): 85,827,876 (86M)
- Swin (huggingface microsoft/swin-tiny-patch4-window7-224): 27,547,038 (28M)
- SwinV2 (huggingface microsoft/swinv2-tiny-patch4-window16-256): 27,605,838 (28M)
- CLIP (huggingface openai/clip-vit-base-patch32): 87,483,684 (87M)
- CoAtNet: 24,033,296 (24M)