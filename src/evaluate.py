import os
import sys
import json
from typing import Dict, Any
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset import create_dataloaders
from models.classifier import load_checkpoint

def evaluate_test_set(
    model_path: str = "models/best_model.pth",
    dataset_dir: str = "dataset",
    batch_size: int = 32,
    confusion_matrix_path: str = "reports/confusion_matrix.png",
    results_path: str = "reports/evaluation_results.json"
) -> Dict[str, Any]:
    """
    Evaluates the model on the held-out test dataset,
    computes classification metrics, and generates a confusion matrix heatmap.
    """
    os.makedirs(os.path.dirname(confusion_matrix_path), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating with device: {device}")

    model, checkpoint = load_checkpoint(model_path, device=device)
    class_names = checkpoint.get("class_names", [])

    _, _, test_loader, loader_classes = create_dataloaders(
        dataset_dir=dataset_dir,
        batch_size=batch_size,
        num_workers=2
    )

    if not class_names:
        class_names = loader_classes

    all_preds = []
    all_labels = []
    all_probs = []

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(probs, 1)

            total_loss += loss.item() * images.size(0)
            total_samples += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    test_loss = total_loss / total_samples
    test_accuracy = float(np.mean(all_preds == all_labels))

    print(f"\n" + "="*50)
    print(f"TEST SET EVALUATION SUMMARY")
    print("="*50)
    print(f"Total Test Samples: {total_samples}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    report_dict = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        output_dict=True,
        digits=4
    )
    report_text = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        digits=4
    )
    print("\nClassification Report:\n")
    print(report_text)

    # Compute and plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        annot_kws={"size": 13, "weight": "bold"}
    )
    plt.title(f"Confusion Matrix (Test Accuracy: {test_accuracy*100:.2f}%)", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("Predicted Class", fontsize=12, labelpad=8)
    plt.ylabel("True Class", fontsize=12, labelpad=8)
    plt.tight_layout()
    plt.savefig(confusion_matrix_path, dpi=300)
    plt.close()
    print(f"Confusion matrix plot saved to {confusion_matrix_path}")

    # Save results to JSON
    results = {
        'test_accuracy': test_accuracy,
        'test_loss': test_loss,
        'total_samples': total_samples,
        'class_names': class_names,
        'classification_report': report_dict,
        'confusion_matrix': cm.tolist()
    }
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Evaluation metrics saved to {results_path}")

    return results

if __name__ == "__main__":
    evaluate_test_set()
