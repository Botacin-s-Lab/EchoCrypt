"""
Baseline: Standard fine-tuning with k-shot samples.

Usage:
    # Image models (Swin, ViT, etc.)
    python baseline.py --dataset_dir ../img_dataset_phone --model swin --k_shot 5

    # Audio model (AST)
    python baseline.py --dataset_dir ../new_dataset_phone --model ast --k_shot 5
"""

import os
import sys
import argparse
import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import (
    ViTForImageClassification,
    SwinForImageClassification,
    DeiTForImageClassification,
    BeitForImageClassification,
    CLIPVisionModel,
    ASTForAudioClassification,
)
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

from data_utils import load_dataset_and_splits, MODEL_CONFIGS, get_class_labels, is_audio_model


class CLIPClassifier(nn.Module):
    def __init__(self, vision_model, num_labels):
        super().__init__()
        self.vision_model = vision_model
        self.classifier = nn.Linear(vision_model.config.hidden_size, num_labels)

    def forward(self, pixel_values):
        outputs = self.vision_model(pixel_values=pixel_values)
        return self.classifier(outputs.pooler_output)


def load_model(model_name, num_classes, device):
    """Load pretrained model for classification."""
    hf_name = MODEL_CONFIGS[model_name]

    if model_name == "ast":
        model = ASTForAudioClassification.from_pretrained(
            hf_name, num_labels=num_classes, ignore_mismatched_sizes=True
        )
    elif model_name == "swin":
        model = SwinForImageClassification.from_pretrained(
            hf_name, num_labels=num_classes, ignore_mismatched_sizes=True
        )
    elif model_name == "vit":
        model = ViTForImageClassification.from_pretrained(
            hf_name, num_labels=num_classes, ignore_mismatched_sizes=True
        )
    elif model_name == "deit":
        model = DeiTForImageClassification.from_pretrained(
            hf_name, num_labels=num_classes, ignore_mismatched_sizes=True
        )
    elif model_name == "beit":
        model = BeitForImageClassification.from_pretrained(
            hf_name, num_labels=num_classes, ignore_mismatched_sizes=True
        )
    elif model_name == "clip":
        vision_model = CLIPVisionModel.from_pretrained(hf_name)
        model = CLIPClassifier(vision_model, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model.to(device)


def get_outputs(model, inputs, model_name):
    """Extract logits from model output."""
    if model_name == "ast":
        return model(input_values=inputs).logits
    elif model_name == "clip":
        return model(pixel_values=inputs)
    outputs = model(inputs)
    if isinstance(outputs, torch.Tensor):
        return outputs
    return outputs.logits


def train_epoch(model, loader, optimizer, criterion, device, model_name):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        logits = get_outputs(model, inputs, model_name)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, model_name):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        logits = get_outputs(model, inputs, model_name)
        loss = criterion(logits, labels)

        total_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = correct / total
    return total_loss / total, acc, all_preds, all_labels


def run_baseline(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data with k-shot split
    support_set, val_set, test_set, num_classes = load_dataset_and_splits(
        args.dataset_dir, args.model, args.k_shot, seed=args.seed
    )

    support_loader = DataLoader(support_set, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)

    # Load backbone for feature extraction (no classification head)
    from transformers import ASTModel
    hf_name = MODEL_CONFIGS[args.model]

    if args.model == "ast":
        backbone = ASTModel.from_pretrained(hf_name).to(device)
    else:
        backbone = load_model(args.model, num_classes, device)

    backbone.eval()

    # Pre-extract features ONCE with frozen backbone
    print("  Pre-extracting features from frozen backbone...")

    @torch.no_grad()
    def extract_all(loader, desc="Extracting"):
        feats, labs = [], []
        for inputs, labels in tqdm(loader, desc=desc):
            inputs = inputs.to(device)
            if args.model == "ast":
                out = backbone(input_values=inputs)
            else:
                out = backbone(pixel_values=inputs)
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                feats.append(out.pooler_output.cpu())
            else:
                feats.append(out.last_hidden_state.mean(dim=1).cpu())
            labs.append(labels)
        return torch.cat(feats), torch.cat(labs)

    train_features, train_labels = extract_all(support_loader, "Support features")
    val_features, val_labels = extract_all(val_loader, "Val features")
    test_features, test_labels = extract_all(test_loader, "Test features")

    feat_dim = train_features.shape[1]
    print(f"  Feature dim: {feat_dim}, extracted {len(train_features)}+{len(val_features)}+{len(test_features)} samples")

    del backbone
    torch.cuda.empty_cache()

    # Simple linear classifier on pre-extracted features
    classifier = nn.Linear(feat_dim, num_classes).to(device)
    train_features = train_features.to(device)
    train_labels = train_labels.to(device)
    val_features = val_features.to(device)
    val_labels = val_labels.to(device)

    optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.1)

    # Training loop (on features, no backbone forward pass needed)
    best_val_acc = 0.0
    patience_counter = 0
    best_state = None

    for epoch in range(args.epochs):
        # Train
        classifier.train()
        # Shuffle
        perm = torch.randperm(len(train_features))
        logits = classifier(train_features[perm])
        loss = criterion(logits, train_labels[perm])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc = (logits.argmax(1) == train_labels[perm]).float().mean().item()

        # Val
        classifier.eval()
        with torch.no_grad():
            val_logits = classifier(val_features)
            val_loss = criterion(val_logits, val_labels).item()
            val_acc = (val_logits.argmax(1) == val_labels).float().mean().item()

        scheduler.step(val_acc)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1}/{args.epochs} | "
                f"Train Loss: {loss.item():.4f} Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in classifier.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Test
    if best_state is not None:
        classifier.load_state_dict(best_state)
        classifier.to(device)

    classifier.eval()
    test_features = test_features.to(device)
    test_labels_dev = test_labels.to(device)
    with torch.no_grad():
        test_logits = classifier(test_features)
        preds = test_logits.argmax(1).cpu().numpy()
        labels = test_labels.numpy()
        test_acc = (test_logits.argmax(1) == test_labels_dev).float().mean().item()

    class_labels = get_class_labels()
    target_names = [class_labels[i] for i in range(num_classes)]
    report = classification_report(labels, preds, target_names=target_names, digits=4)

    print(f"\n{'='*60}")
    print(f"BASELINE RESULT | model={args.model} k_shot={args.k_shot} seed={args.seed}")
    print(f"{'='*60}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Best Val Accuracy: {best_val_acc:.4f}")
    print(report)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    result = {
        "method": "baseline",
        "model": args.model,
        "dataset": args.dataset_dir,
        "k_shot": args.k_shot,
        "seed": args.seed,
        "test_accuracy": test_acc,
        "best_val_accuracy": best_val_acc,
    }
    result_path = os.path.join(
        args.output_dir,
        f"baseline_{args.model}_k{args.k_shot}_seed{args.seed}.json"
    )
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to {result_path}")

    return test_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline k-shot fine-tuning for ASCA")
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="img_dataset_phone (images) or new_dataset_phone (audio)")
    parser.add_argument("--model", type=str, default="swin",
                        choices=["swin", "vit", "deit", "beit", "clip", "ast"])
    parser.add_argument("--k_shot", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--output_dir", type=str, default="results")

    args = parser.parse_args()
    run_baseline(args)
