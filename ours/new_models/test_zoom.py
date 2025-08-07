import os
import torch
import numpy as np
import matplotlib.pyplot as plt 
import argparse
import seaborn as sns
from PIL import Image

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from transformers import ViTForImageClassification, SwinForImageClassification, Swinv2ForImageClassification, DeiTForImageClassification, BeitForImageClassification
from transformers import CLIPVisionModel, CLIPConfig
from transformers import AutoImageProcessor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tqdm import tqdm


def load_model(model_type, model_path, device):
    """Load the trained model from the specified path"""
    
    # Define model configurations based on type
    if model_type == 'swin':
        model_name = 'microsoft/swin-tiny-patch4-window7-224'
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = SwinForImageClassification.from_pretrained(
            model_name,
            num_labels=36,
            ignore_mismatched_sizes=True  
        )
    elif model_type == 'swinv2':
        model_name = 'microsoft/swinv2-tiny-patch4-window16-256'
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = Swinv2ForImageClassification.from_pretrained(
            model_name,
            num_labels=36,
            ignore_mismatched_sizes=True  
        )
    elif model_type == 'vit':
        model_name = 'google/vit-base-patch16-224-in21k'
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=36,
            ignore_mismatched_sizes=True,
        )
    elif model_type == 'deit':
        model_name = 'facebook/deit-base-distilled-patch16-224'
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = DeiTForImageClassification.from_pretrained(
            model_name,
            num_labels=36,
            ignore_mismatched_sizes=True
        )
    elif model_type == 'beit':
        model_name = 'microsoft/beit-base-patch16-224-pt22k-ft22k'  
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = BeitForImageClassification.from_pretrained(
            model_name,
            num_labels=36,
            ignore_mismatched_sizes=True
        )
    elif model_type == 'clip':
        model_name = 'openai/clip-vit-base-patch32'
        processor = AutoImageProcessor.from_pretrained(model_name)
        vision_model = CLIPVisionModel.from_pretrained(model_name)
        
        num_labels = 36
        
        class CLIPVisionForCustomImageClassification(torch.nn.Module):
            def __init__(self, vision_model, num_labels):
                super().__init__()
                self.vision_model = vision_model
                self.classifier = torch.nn.Linear(vision_model.config.hidden_size, num_labels)
            
            def forward(self, pixel_values):
                outputs = self.vision_model(pixel_values=pixel_values)
                pooled_output = outputs.pooler_output
                logits = self.classifier(pooled_output)
                return logits
        
        model = CLIPVisionForCustomImageClassification(vision_model, num_labels)
    else:
        raise ValueError(f'[ERROR] Unsupported model type: {model_type}')
    
    # Load the trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model, processor


def get_transforms(model_type, processor):
    """Get the appropriate transforms for the model type"""
    
    if model_type in ('vit', 'swin', 'deit', 'beit', 'clip'):
        transforms_list = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
        ])
    elif model_type == 'swinv2':
        transforms_list = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
        ])
    else:
        raise ValueError(f'[ERROR] Unsupported model type for transforms: {model_type}')
    
    return transforms_list


def get_outputs(model, inputs, model_type):
    """Get model outputs based on model type"""
    if model_type == 'clip':
        outputs = model(pixel_values=inputs)
    else:
        outputs = model(inputs)
    
    if isinstance(outputs, torch.Tensor):
        return outputs
    elif hasattr(outputs, 'logits'):
        return outputs.logits
    else:
        raise ValueError("Model output format not recognized.")


def test_model(model, test_loader, device, model_type):
    """Test the model on the dataset and return predictions and labels"""
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing", leave=False):
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = get_outputs(model, inputs, model_type)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())
    
    return all_preds, all_labels, all_probs


def save_results(all_preds, all_labels, output_dir, model_type, dataset_name):
    """Save classification results and metrics"""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Generate class labels
    class_labels = [str(i) for i in range(10)] + [chr(i) for i in range(ord('a'), ord('z') + 1)]
    
    # Classification report
    report = classification_report(all_labels, all_preds, target_names=class_labels, digits=4)
    
    # Save classification report
    report_path = os.path.join(output_dir, f"test_results_{model_type}_{dataset_name}.txt")
    with open(report_path, "w") as file:
        file.write(f"Test Dataset: {dataset_name}\n")
        file.write(f"Model Type: {model_type}\n")
        file.write(f"Test Accuracy: {accuracy:.4f}\n\n")
        file.write("Classification Report:\n")
        file.write(report)
    
    print(f"Results saved to: {report_path}")
    
    # Generate and save confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title(f'Confusion Matrix - {model_type} on {dataset_name}')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.tight_layout()
    
    cm_path = os.path.join(output_dir, f"confusion_matrix_{model_type}_{dataset_name}.png")
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to: {cm_path}")
    
    return accuracy, report


def main():
    parser = argparse.ArgumentParser(description='Test trained models on new datasets')
    parser.add_argument(
        "--model",
        type=str,
        choices=['vit', 'swin', 'swinv2', 'deit', 'beit', 'clip'],
        required=True,
        help="Choose the model: 'vit', 'swin', 'swinv2', 'deit', 'beit', or 'clip'"
    )
    parser.add_argument(
        "--seed_folder",
        type=str,
        required=True,
        help="Seed folder containing the trained model (e.g., 'test_run')"
    )
    parser.add_argument(
        "--test_dataset",
        type=str,
        default="../../img_noise_audio_dataset_zoom",
        help="Path to the test dataset directory"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for testing"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="test_results",
        help="Directory to save test results"
    )
    
    args = parser.parse_args()
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Construct model path
    model_path = os.path.join(args.seed_folder, f"best_model_zoom_{args.model}.pth")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Loading model from: {model_path}")
    
    # Load model
    model, processor = load_model(args.model, model_path, device)
    print(f"Model {args.model} loaded successfully")
    
    # Get transforms
    test_transforms = get_transforms(args.model, processor)
    
    # Load test dataset
    if not os.path.exists(args.test_dataset):
        raise FileNotFoundError(f"Test dataset not found: {args.test_dataset}")
    
    print(f"Loading test dataset from: {args.test_dataset}")
    test_dataset = ImageFolder(root=args.test_dataset, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Number of classes: {len(test_dataset.classes)}")
    
    # Test the model
    print("Starting inference...")
    all_preds, all_labels, all_probs = test_model(model, test_loader, device, args.model)
    
    # Extract dataset name for saving results
    dataset_name = os.path.basename(args.test_dataset.rstrip('/'))
    
    # Save results
    accuracy, report = save_results(all_preds, all_labels, args.output_dir, args.model, dataset_name)
    
    print("\nTesting completed!")
    print(f"Final Test Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()