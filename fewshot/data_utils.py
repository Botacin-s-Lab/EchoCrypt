"""
Few-shot data utilities for ASCA keystroke classification.

Supports both:
  - Image-based models (Swin, ViT, etc.) with spectrogram images
  - Audio-based models (AST) with raw .wav files
"""

import os
import random
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from transformers import AutoImageProcessor, ASTFeatureExtractor
from collections import defaultdict
from tqdm import tqdm


# Model name mapping
MODEL_CONFIGS = {
    "swin": "microsoft/swin-tiny-patch4-window7-224",
    "vit": "google/vit-base-patch16-224-in21k",
    "deit": "facebook/deit-base-distilled-patch16-224",
    "beit": "microsoft/beit-base-patch16-224-pt22k-ft22k",
    "clip": "openai/clip-vit-base-patch32",
    "ast": "MIT/ast-finetuned-audioset-10-10-0.4593",
}

# Models that use raw audio instead of spectrogram images
AUDIO_MODELS = {"ast"}


def is_audio_model(model_name):
    return model_name in AUDIO_MODELS


# ---------------------------------------------------------------------------
# Audio dataset (for AST)
# ---------------------------------------------------------------------------

class AudioFolder:
    """
    ImageFolder-like dataset for .wav audio files.
    Expects directory structure: root/class_name/*.wav
    """

    def __init__(self, root):
        self.root = root
        self.classes = sorted([
            d for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))
        ])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.samples = []

        for cls in self.classes:
            cls_dir = os.path.join(root, cls)
            for fname in sorted(os.listdir(cls_dir)):
                if fname.endswith('.wav'):
                    self.samples.append(
                        (os.path.join(cls_dir, fname), self.class_to_idx[cls])
                    )


class AudioSubset(Dataset):
    """Subset of AudioFolder with AST feature extraction and caching."""

    def __init__(self, dataset, indices, feature_extractor):
        self.dataset = dataset
        self.indices = indices
        self.feature_extractor = feature_extractor
        # AST expects 16000 Hz
        self.target_sr = feature_extractor.sampling_rate
        # Cache: preprocess all samples once upfront
        self._cache = {}
        self._preprocess_all()

    def _preprocess_all(self):
        """Load, resample, and extract features for all samples once."""
        for i in tqdm(range(len(self.indices)), desc=f"Caching {len(self.indices)} audio samples"):
            path, label = self.dataset.samples[self.indices[i]]
            waveform, sr = torchaudio.load(path)

            # Always resample to AST's expected rate (16000 Hz)
            if sr != self.target_sr:
                waveform = torchaudio.transforms.Resample(sr, self.target_sr)(waveform)

            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0)
            else:
                waveform = waveform.squeeze(0)

            # Process with AST feature extractor
            inputs = self.feature_extractor(
                waveform.numpy(),
                sampling_rate=self.target_sr,
                return_tensors="pt"
            )
            input_values = inputs.input_values.squeeze(0)
            self._cache[i] = (input_values, label)
        print(f"  Caching done.")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self._cache[idx]


# ---------------------------------------------------------------------------
# Image dataset (for Swin, ViT, etc.)
# ---------------------------------------------------------------------------

class TransformSubset(Dataset):
    """Subset of ImageFolder with a custom transform applied."""

    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        path, label = self.dataset.samples[self.indices[idx]]
        from PIL import Image
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def get_class_indices(dataset):
    """Group dataset indices by class label."""
    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset.samples):
        class_indices[label].append(idx)
    return class_indices


def few_shot_split(dataset, k_shot, n_val=5, n_test=5, seed=42):
    """
    Split dataset into k-shot support, validation, and test sets.

    Per class (25 samples total):
      - test: n_test samples (fixed)
      - val: n_val samples (fixed)
      - support: k_shot samples from remaining
    """
    rng = random.Random(seed)
    class_indices = get_class_indices(dataset)

    support_indices = []
    val_indices = []
    test_indices = []

    for cls in sorted(class_indices.keys()):
        indices = class_indices[cls]
        shuffled = indices.copy()
        rng.shuffle(shuffled)

        test_indices.extend(shuffled[:n_test])
        val_indices.extend(shuffled[n_test:n_test + n_val])

        remaining = shuffled[n_test + n_val:]
        if k_shot > len(remaining):
            raise ValueError(
                f"k_shot={k_shot} exceeds available samples "
                f"({len(remaining)}) for class {cls}"
            )
        support_indices.extend(remaining[:k_shot])

    return support_indices, val_indices, test_indices


def get_transforms(model_name, is_train=False):
    """Get image transforms for vision models."""
    processor = AutoImageProcessor.from_pretrained(model_name)

    if is_train:
        tfm = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
            transforms.RandomErasing(p=0.5, scale=(0.05, 0.2), ratio=(0.01, 5)),
        ])
    else:
        tfm = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
        ])

    return tfm, processor


def load_dataset_and_splits(dataset_dir, model_name, k_shot, seed=42):
    """
    Load dataset and create k-shot splits.

    For image models: dataset_dir should point to spectrogram images (img_dataset_phone)
    For audio models: dataset_dir should point to raw audio (new_dataset_phone)

    Returns:
        support_set, val_set, test_set, num_classes
    """
    if is_audio_model(model_name):
        # AST: load raw audio
        raw_dataset = AudioFolder(root=dataset_dir)
        num_classes = len(raw_dataset.classes)

        support_idx, val_idx, test_idx = few_shot_split(
            raw_dataset, k_shot=k_shot, seed=seed
        )

        hf_name = MODEL_CONFIGS[model_name]
        feature_extractor = ASTFeatureExtractor.from_pretrained(hf_name)

        support_set = AudioSubset(raw_dataset, support_idx, feature_extractor)
        val_set = AudioSubset(raw_dataset, val_idx, feature_extractor)
        test_set = AudioSubset(raw_dataset, test_idx, feature_extractor)
    else:
        # Vision models: load spectrogram images
        hf_name = MODEL_CONFIGS.get(model_name, model_name)
        train_tfm, _ = get_transforms(hf_name, is_train=True)
        test_tfm, _ = get_transforms(hf_name, is_train=False)

        raw_dataset = ImageFolder(root=dataset_dir)
        num_classes = len(raw_dataset.classes)

        support_idx, val_idx, test_idx = few_shot_split(
            raw_dataset, k_shot=k_shot, seed=seed
        )

        support_set = TransformSubset(raw_dataset, support_idx, transform=train_tfm)
        val_set = TransformSubset(raw_dataset, val_idx, transform=test_tfm)
        test_set = TransformSubset(raw_dataset, test_idx, transform=test_tfm)

    print(f"Dataset: {dataset_dir}")
    print(f"  Model: {model_name}, Classes: {num_classes}, k_shot: {k_shot}")
    print(f"  Support: {len(support_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

    return support_set, val_set, test_set, num_classes


def get_class_labels():
    """Return class label mapping (0-9, a-z)."""
    labels = [str(i) for i in range(10)] + [chr(i) for i in range(ord('a'), ord('z') + 1)]
    return {i: l for i, l in enumerate(labels)}
