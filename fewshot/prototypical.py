"""
Prototypical Networks for few-shot ASCA keystroke classification.

Two modes:
  1. frozen   - Pretrained backbone as frozen feature extractor, no training
  2. finetune - Episodic training to adapt the backbone

Usage:
    # Image models
    python prototypical.py --dataset_dir ../img_dataset_phone --model swin --k_shot 5 --mode frozen

    # Audio model (AST)
    python prototypical.py --dataset_dir ../new_dataset_phone --model ast --k_shot 5 --mode frozen
"""

import os
import sys
import argparse
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from transformers import (
    ViTModel,
    SwinModel,
    DeiTModel,
    BeitModel,
    CLIPVisionModel,
    ASTModel,
)
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

from data_utils import (
    load_dataset_and_splits,
    MODEL_CONFIGS,
    get_class_labels,
    is_audio_model,
)


# ---------------------------------------------------------------------------
# Feature extractor (backbone without classification head)
# ---------------------------------------------------------------------------

class FeatureExtractor(nn.Module):
    """Wraps a pretrained model to output feature embeddings."""

    def __init__(self, model_name):
        super().__init__()
        hf_name = MODEL_CONFIGS[model_name]
        self.model_name = model_name
        self._is_audio = is_audio_model(model_name)

        if model_name == "ast":
            self.backbone = ASTModel.from_pretrained(hf_name)
            self.feat_dim = self.backbone.config.hidden_size
        elif model_name == "swin":
            self.backbone = SwinModel.from_pretrained(hf_name)
            self.feat_dim = self.backbone.config.hidden_size
        elif model_name == "vit":
            self.backbone = ViTModel.from_pretrained(hf_name)
            self.feat_dim = self.backbone.config.hidden_size
        elif model_name == "deit":
            self.backbone = DeiTModel.from_pretrained(hf_name)
            self.feat_dim = self.backbone.config.hidden_size
        elif model_name == "beit":
            self.backbone = BeitModel.from_pretrained(hf_name)
            self.feat_dim = self.backbone.config.hidden_size
        elif model_name == "clip":
            self.backbone = CLIPVisionModel.from_pretrained(hf_name)
            self.feat_dim = self.backbone.config.hidden_size
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def forward(self, x):
        if self._is_audio:
            outputs = self.backbone(input_values=x)
        else:
            outputs = self.backbone(pixel_values=x)

        # Use pooled output (CLS token) as the embedding
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output
        # Fallback: mean of last hidden state
        return outputs.last_hidden_state.mean(dim=1)


# ---------------------------------------------------------------------------
# Prototypical Network logic
# ---------------------------------------------------------------------------

def compute_prototypes(embeddings, labels, num_classes):
    """Compute class prototypes as mean embeddings."""
    prototypes = torch.zeros(num_classes, embeddings.size(1), device=embeddings.device)
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() > 0:
            prototypes[c] = embeddings[mask].mean(dim=0)
    return prototypes


def classify_by_prototype(query_embeddings, prototypes):
    """Classify queries by nearest prototype (Euclidean distance)."""
    dists = torch.cdist(query_embeddings, prototypes, p=2)
    predictions = dists.argmin(dim=1)
    return predictions, dists


# ---------------------------------------------------------------------------
# Frozen mode
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_features(model, loader, device):
    """Extract features from all samples in a DataLoader."""
    model.eval()
    all_features, all_labels = [], []

    for inputs, labels in tqdm(loader, desc="Extracting features"):
        inputs = inputs.to(device)
        features = model(inputs)
        all_features.append(features.cpu())
        all_labels.append(labels)

    return torch.cat(all_features), torch.cat(all_labels)


def run_frozen(args):
    """Frozen backbone: compute prototypes from support, classify test."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    support_set, val_set, test_set, num_classes = load_dataset_and_splits(
        args.dataset_dir, args.model, args.k_shot, seed=args.seed
    )

    support_loader = DataLoader(support_set, batch_size=args.batch_size)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)

    print(f"Loading pretrained {args.model} backbone...")
    model = FeatureExtractor(args.model).to(device)
    print(f"Feature dim: {model.feat_dim}")

    support_features, support_labels = extract_features(model, support_loader, device)
    test_features, test_labels = extract_features(model, test_loader, device)

    prototypes = compute_prototypes(support_features, support_labels, num_classes)
    predictions, _ = classify_by_prototype(test_features, prototypes)

    preds = predictions.numpy()
    labels = test_labels.numpy()
    test_acc = accuracy_score(labels, preds)

    class_label_map = get_class_labels()
    target_names = [class_label_map[i] for i in range(num_classes)]
    report = classification_report(labels, preds, target_names=target_names, digits=4)

    print(f"\n{'='*60}")
    print(f"PROTONET (FROZEN) | model={args.model} k_shot={args.k_shot} seed={args.seed}")
    print(f"{'='*60}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(report)

    os.makedirs(args.output_dir, exist_ok=True)
    result = {
        "method": "protonet_frozen",
        "model": args.model,
        "dataset": args.dataset_dir,
        "k_shot": args.k_shot,
        "seed": args.seed,
        "test_accuracy": test_acc,
    }
    result_path = os.path.join(
        args.output_dir,
        f"protonet_frozen_{args.model}_k{args.k_shot}_seed{args.seed}.json"
    )
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to {result_path}")

    return test_acc


# ---------------------------------------------------------------------------
# Episodic fine-tuning mode
# ---------------------------------------------------------------------------

def sample_episode(features, labels, num_classes, n_way, k_support, k_query):
    """Sample a single episode for episodic training."""
    classes = random.sample(range(num_classes), min(n_way, num_classes))

    support_f, support_l = [], []
    query_f, query_l = [], []

    for new_label, cls in enumerate(classes):
        mask = labels == cls
        cls_features = features[mask]
        n_available = cls_features.size(0)
        n_needed = k_support + k_query

        if n_available < n_needed:
            perm = torch.randperm(n_available)
            support_f.append(cls_features[perm[:min(k_support, n_available)]])
            support_l.extend([new_label] * min(k_support, n_available))
            query_idx = torch.randint(0, n_available, (k_query,))
            query_f.append(cls_features[query_idx])
            query_l.extend([new_label] * k_query)
        else:
            perm = torch.randperm(n_available)
            support_f.append(cls_features[perm[:k_support]])
            support_l.extend([new_label] * k_support)
            query_f.append(cls_features[perm[k_support:k_support + k_query]])
            query_l.extend([new_label] * k_query)

    return torch.cat(support_f), torch.tensor(support_l), torch.cat(query_f), torch.tensor(query_l)


def prototypical_loss(support_feat, support_lab, query_feat, query_lab, n_way):
    """Compute prototypical network loss on an episode."""
    prototypes = compute_prototypes(support_feat, support_lab, n_way)
    dists = torch.cdist(query_feat, prototypes, p=2)
    log_probs = F.log_softmax(-dists, dim=1)
    loss = F.nll_loss(log_probs, query_lab)
    preds = (-dists).argmax(dim=1)
    acc = (preds == query_lab).float().mean()
    return loss, acc


def run_finetune(args):
    """Episodic fine-tuning of the backbone (last N layers only for speed)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    support_set, val_set, test_set, num_classes = load_dataset_and_splits(
        args.dataset_dir, args.model, args.k_shot, seed=args.seed
    )

    support_loader = DataLoader(support_set, batch_size=args.batch_size)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)

    print(f"Loading pretrained {args.model} backbone for fine-tuning...")
    model = FeatureExtractor(args.model).to(device)

    # Freeze all layers first, then unfreeze last 2 transformer layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze last 2 encoder layers + layernorm
    if hasattr(model.backbone, 'encoder'):
        layers = model.backbone.encoder.layer
        for layer in layers[-2:]:
            for param in layer.parameters():
                param.requires_grad = True
    if hasattr(model.backbone, 'layernorm'):
        for param in model.backbone.layernorm.parameters():
            param.requires_grad = True

    trainable = [p for p in model.parameters() if p.requires_grad]
    total = sum(p.numel() for p in model.parameters())
    trainable_n = sum(p.numel() for p in trainable)
    print(f"  Trainable params: {trainable_n:,} / {total:,}")

    optimizer = torch.optim.Adam(trainable, lr=args.finetune_lr)

    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0

    # Pre-load all support data to GPU for episodic training
    all_support_inputs = []
    all_support_labels = []
    for inputs, labels in tqdm(support_loader, desc="Loading support to GPU"):
        all_support_inputs.append(inputs.to(device))
        all_support_labels.append(labels.to(device))
    all_support_inputs = torch.cat(all_support_inputs)
    all_support_labels = torch.cat(all_support_labels)

    total_epochs = args.finetune_episodes // args.episodes_per_epoch
    for epoch in tqdm(range(total_epochs), desc="Finetune epochs"):
        model.train()
        epoch_loss, epoch_acc = 0.0, 0.0

        # ONE forward pass per epoch, accumulate losses across episodes
        optimizer.zero_grad()
        support_features = model(all_support_inputs)

        total_loss = 0
        for ep in range(args.episodes_per_epoch):
            s_feat, s_lab, q_feat, q_lab = sample_episode(
                support_features, all_support_labels, num_classes,
                n_way=min(args.n_way, num_classes),
                k_support=max(1, args.k_shot // 2),
                k_query=max(1, args.k_shot - args.k_shot // 2),
            )
            s_lab, q_lab = s_lab.to(device), q_lab.to(device)

            loss, acc = prototypical_loss(
                s_feat, s_lab, q_feat, q_lab,
                n_way=min(args.n_way, num_classes)
            )
            total_loss = total_loss + loss
            epoch_acc += acc.item()

        # One backward + step per epoch
        (total_loss / args.episodes_per_epoch).backward()
        optimizer.step()

        epoch_loss = total_loss.item() / args.episodes_per_epoch
        epoch_acc /= args.episodes_per_epoch

        # Validation (no grad)
        val_features, val_labels = extract_features(model, val_loader, device)
        sup_feat_val, sup_lab_val = extract_features(model, support_loader, device)

        prototypes = compute_prototypes(sup_feat_val, sup_lab_val, num_classes)
        val_preds, _ = classify_by_prototype(val_features, prototypes)
        val_acc = (val_preds == val_labels).float().mean().item()

        ep_num = (epoch + 1) * args.episodes_per_epoch
        print(
            f"Episodes {ep_num}/{args.finetune_episodes} | "
            f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"Early stopping at episode {ep_num}")
            break

    # Test
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.to(device)

    test_features, test_labels = extract_features(model, test_loader, device)
    sup_features, sup_labels = extract_features(model, support_loader, device)

    prototypes = compute_prototypes(sup_features, sup_labels, num_classes)
    predictions, _ = classify_by_prototype(test_features, prototypes)

    preds = predictions.numpy()
    labels = test_labels.numpy()
    test_acc = accuracy_score(labels, preds)

    class_label_map = get_class_labels()
    target_names = [class_label_map[i] for i in range(num_classes)]
    report = classification_report(labels, preds, target_names=target_names, digits=4)

    print(f"\n{'='*60}")
    print(f"PROTONET (FINETUNED) | model={args.model} k_shot={args.k_shot} seed={args.seed}")
    print(f"{'='*60}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Best Val Accuracy: {best_val_acc:.4f}")
    print(report)

    os.makedirs(args.output_dir, exist_ok=True)
    result = {
        "method": "protonet_finetune",
        "model": args.model,
        "dataset": args.dataset_dir,
        "k_shot": args.k_shot,
        "seed": args.seed,
        "test_accuracy": test_acc,
        "best_val_accuracy": best_val_acc,
    }
    result_path = os.path.join(
        args.output_dir,
        f"protonet_finetune_{args.model}_k{args.k_shot}_seed{args.seed}.json"
    )
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to {result_path}")

    return test_acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prototypical Networks for few-shot ASCA")
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="swin",
                        choices=["swin", "vit", "deit", "beit", "clip", "ast"])
    parser.add_argument("--k_shot", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--mode", type=str, default="frozen",
                        choices=["frozen", "finetune"])

    # Fine-tuning specific
    parser.add_argument("--finetune_lr", type=float, default=1e-5)
    parser.add_argument("--finetune_episodes", type=int, default=2000)
    parser.add_argument("--episodes_per_epoch", type=int, default=50)
    parser.add_argument("--n_way", type=int, default=20)
    parser.add_argument("--patience", type=int, default=10)

    args = parser.parse_args()

    if args.mode == "frozen":
        run_frozen(args)
    else:
        run_finetune(args)
