import os
import torch
import numpy as np
import matplotlib.pyplot as plt 
import torchvision.transforms.functional as F
import argparse
import seaborn as sns

from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision import transforms
from transformers import ViTForImageClassification, SwinForImageClassification, Swinv2ForImageClassification, DeiTForImageClassification, BeitForImageClassification
from transformers import CLIPVisionModel, CLIPConfig
from transformers import AutoImageProcessor
from collections import defaultdict
from PIL import Image


# Define the dataset path
dataset_dir = '../../img_dataset_phone'
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


# Define transformations
if selected_model in ('vit', 'swin', 'deit', 'beit', 'clip'):
    print("vit/swin/deit/beit/clip activated")
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
        # transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomRotation(10),
        # Shifting (translation)
        # transforms.RandomAffine(
        #     degrees=0,               # No additional rotation
        #     translate=(0.1, 0.1),    # Horizontal and vertical shifts (10% of image size)
        # ),
        # Masking (Random Erasing)
        transforms.RandomErasing(
            p=0.5,
            scale=(0.05, 0.2),
            ratio=(0.01, 5)
        ),
    ])
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
    ])
elif selected_model == 'swinv2':
    print("swinv2 activated")
    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
        # transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomRotation(10),
        # Shifting (translation)
        # transforms.RandomAffine(
        #     degrees=0,
        #     translate=(0.1, 0.1),
        # ),
        # Masking (Random Erasing)
        transforms.RandomErasing(
            p=0.5,
            scale=(0.05, 0.2),
            ratio=(0.01, 5)
        ),
    ])
    test_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
    ])
else:
    raise ValueError('[ERROR] Define any transformations')


# Load the dataset
full_dataset = ImageFolder(root=dataset_dir, transform=train_transforms)

image_path, _ = full_dataset.samples[0] 
sample_img = Image.open(image_path).convert('RGB')
transformed_img = train_transforms(sample_img)


# Denormalize the tensor for visualization
inv_normalize = transforms.Normalize(
    mean=[-m / s for m, s in zip(processor.image_mean, processor.image_std)],
    std=[1 / s for s in processor.image_std]
)
transformed_img = inv_normalize(transformed_img)
transformed_img = transformed_img.permute(1, 2, 0).numpy()
transformed_img = transformed_img.clip(0, 1)

# plt.imshow(transformed_img)
# plt.axis('off')
# plt.show()

# Split the dataset per class into train and test sets
from collections import defaultdict
from sklearn.model_selection import train_test_split

# Extract labels (targets) from the dataset
targets = [sample[1] for sample in full_dataset.samples]  # Assuming ImageFolder's samples attribute

# First split: Train and Temp (Val + Test)
train_indices, temp_indices, y_train, y_temp = train_test_split(
    range(len(targets)),
    targets,
    test_size=0.3,  # 30% of the data will go to val+test
    stratify=targets,
    random_state=42
)

# Second split: Validation and Test
val_indices, test_indices, y_val, y_test = train_test_split(
    temp_indices,
    y_temp,
    test_size=0.33,  # 33% of the temp data goes to test, resulting in 20% test of the total data
    stratify=y_temp,
    random_state=42
)

# Create Subset datasets
train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
test_dataset = torch.utils.data.Subset(full_dataset, test_indices)

# Apply test transforms to validation and test datasets
val_dataset.dataset.transform = test_transforms
test_dataset.dataset.transform = test_transforms

# Print dataset sizes to verify
print(f"Train set size: {len(train_dataset)}, Validation set size: {len(val_dataset)}, Test set size: {len(test_dataset)}")

# Create DataLoaders
train_loader    = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader      = DataLoader(val_dataset, batch_size=16)
test_loader     = DataLoader(test_dataset, batch_size=16)

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
    
    for batch in tqdm(train_loader, desc="Training", leave=False):
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
        for batch in tqdm(val_loader, desc="Validation", leave=False):
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
        torch.save(model.state_dict(), f'best_model_phone_{selected_model}.pth')
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
model.load_state_dict(torch.load(f'best_model_phone_{selected_model}.pth'))
class_labels = [str(i) for i in range(10)] + [chr(i) for i in range(ord('a'), ord('z') + 1)]

# Collect all predictions and labels
all_preds = []
all_labels = []

model.eval()
with torch.no_grad():
    for batch in test_loader:
        inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = get_outputs(model, inputs)
        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Classification Report
with open(f"classification_report_phone_{selected_model}.txt", "w") as file:
    file.write(classification_report(all_labels, all_preds, digits=4))


