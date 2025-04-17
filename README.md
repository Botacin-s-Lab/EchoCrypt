# Making Acoustic Side-Channel Attacks on Noisy Keyboards Viable with LLM-Assisted Spectrograms "Typo" Correction
This repo is the implementation of a research project aimed at enhancing Acoustic Side-Channel Attacks (ASCAs) using a novel combination of Vision Transformers (VTs) and Large Language Models (LLMs). This project addresses key challenges in ASCA by improving robustness against environmental noise and correcting errors in predictions via LLMs, making ASCAs more practical in real-world scenarios.

The project leverages the latest advancements in transformer architectures (such as CoAtNet, ViT, and Swin) to process noisy acoustic signals and improve keystroke classification. Additionally, it uses LLMs to perform error correction in noisy environments, enhancing the feasibility of ASCAs for sensitive data extraction from keyboard inputs.

## Key Features:
- **Visual Transformers for Keystroke Classification**: Utilizes the power of transformer-based models like CoAtNet, Swin, and ViT for classifying keystrokes from audio spectrograms.
- **LLM-based Error Correction**: Integrates Large Language Models such as GPT-4o and Llama for detecting and correcting typos in noisy keystroke data.
- **Fine-tuning with LoRA**: Efficient fine-tuning of smaller models (via Low-Rank Adaptation) for practical ASCA attacks with lower computational overhead.
- **Robustness to Noisy Conditions**: Implements noise augmentation techniques to evaluate and enhance model performance under realistic noisy environments.

## Project Structure

```
├── README.md                   # This file
├── ours                        # Main project directory
│   ├── finetune                # Scripts for fine-tuning models
│   ├── phone                   # Files for the phone dataset and it's results
│   ├── zoom                    # Files for the zoom dataset and it's results
│   └── sentences               # Datasets used for finetuning and testing
├── reference1                  # Reference implementation of CoAtNet and other models
├── reference2                  # Auxiliary scripts for isolating keypresses and additional research
├── requirements.txt            # Required Python dependencies
├── spectrogram_clean.pdf       # Clean spectrogram examples
├── spectrogram_noisy.pdf       # Noisy spectrogram examples
```

## Installation and Setup
To set up this project, you need Python and several dependencies. Please follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/seyyedaliayati/EchoCrypt.git
   ```

2. Install the required Python packages:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. [Optional] Download the necessary pre-trained models for both the phone and zoom datasets from HuggingFace if you want to re-produce the results:
   - [Zoom Model](https://huggingface.co/seyyedaliayati/zoom_model)
   - [Phone Model](https://huggingface.co/seyyedaliayati/phone_model)

## Citation

If you use this project and find it helpful in your research, please cite the following paper:
```bibtex
@article{ayati2025making,
  title={Making Acoustic Side-Channel Attacks on Noisy Keyboards Viable with LLM-Assisted Spectrograms' "Typo" Correction},
  author={Ayati, Seyyed Ali and Park, Jin Hyun and Cai, Yichen and Botacin, Marcus},
  journal={arXiv preprint arXiv:2504.11622},
  year={2025},
  url={https://arxiv.org/abs/2504.11622}
}
```

## Dataset
The project evaluates models on two main datasets from Harrison et al. (2023) (https://ieeexplore.ieee.org/abstract/document/10190721):
- **Phone Dataset**: Audio data recorded from phone microphones, representing typical keystroke sounds.
- **Zoom Dataset**: Keystroke sounds captured via Zoom audio calls, simulating remote typing scenarios.

Additionally, the **EnglishTense dataset** is used for sentence classification tasks during fine-tuning and error correction evaluations.

## Models
The project evaluates various transformer models for keystroke classification, including:

- **CoAtNet**: Hybrid CNN and attention-based model for extracting features from mel-spectrograms.
- **Vision Transformers (ViT)**: Transformer-based model that treats spectrograms as image-like data.
- **Swin Transformer**: Hierarchical vision transformer for more efficient self-attention.
- **CLIP**: Multimodal transformer used for both image and language processing tasks.
- **Llama and GPT-4o**: Language models for typo correction and contextual error handling in noisy environments.

You can have access to the fine-tuned models on HuggingFace:
- [Zoom Model](https://huggingface.co/seyyedaliayati/zoom_model)
- [Phone Model](https://huggingface.co/seyyedaliayati/phone_model)

## Experiment Results
The project's results demonstrate significant improvements in ASCA accuracy:

- **Phone Model**: 97% accuracy in ideal conditions.
- **Zoom Model**: 98% accuracy in ideal conditions.
- **Performance in Noisy Environments**: When noise is introduced, the models' accuracy drops, but integrating LLMs like GPT-4o helps correct mispredictions, achieving an accuracy of up to 90%.

Key findings include:
- **ViT and Swin Transformers** surpass CNN-based models in classifying keystrokes in spectrograms.
- **LLMs** significantly improve the reliability of ASCAs, particularly when integrated with the keystroke classification pipeline.
- **Fine-tuned LLMs** (using LoRA) offer near-identical performance to larger models like GPT-4o while being more computationally efficient.

## Code Snippets

### Converting Zoom Dataset into New Dataset Format
```python
import torchaudio
import os

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

### Converting Dataset to Image Format for Phone Dataset
```python
import torch
from PIL import Image
from torch.utils.data import ConcatDataset

combined_dataset = ConcatDataset([train_set_no_aug, val_set, test_set])
for item in combined_dataset:
    tensor, label_index = item
    label = chr(label_index + ord('0')) if label_index <= 9 else chr(label_index - 10 + ord('a'))
    
    target_dir = f"../img_dataset_phone/{label}/"
    os.makedirs(target_dir, exist_ok=True)
    jpg_files = [f for f in os.listdir(target_dir) if f.endswith('.jpg')]
    num_jpg_files = len(jpg_files)
    file_name = f"{num_jpg_files}.jpg"
    file_path = os.path.join(target_dir, file_name)
    
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())  # Normalize to [0, 1]
    tensor = tensor * 255  # Scale to [0, 255]
    tensor = tensor.squeeze(0)
    
    image = transforms.ToPILImage()(tensor.byte())
    image.save(file_path)
```

## Contact and Support
For any questions or support regarding this project, please feel free to open an issue on the GitHub repository or contact the authors directly.

## Future Work
The future directions for this research include:
- Expanding the dataset to include more diverse keyboard types, typing behaviors, and environmental noises.
- Enhancing real-time error correction with fine-tuned lightweight LLMs on edge devices.
- Exploring multimodal ASCA techniques beyond audio, including visual side-channel attacks.
