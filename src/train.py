import os
import sys
import time
import json
from typing import Dict, List, Any
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset import create_dataloaders
from models.classifier import get_model, unfreeze_backbone, save_checkpoint

def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> tuple[float, float]:
    """Runs a single training epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> tuple[float, float]:
    """Runs validation on the validation set."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def plot_training_history(history: Dict[str, List[float]], save_path: str = "reports/training_curves.png") -> None:
    """Plots and saves loss and accuracy curves."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Loss plot
    axes[0].plot(epochs, history['train_loss'], 'o-', label='Train Loss', color='#1f77b4', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 's--', label='Val Loss', color='#ff7f0e', linewidth=2)
    axes[0].set_title('Cross-Entropy Loss vs. Epochs', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Loss', fontsize=11)
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend(fontsize=10)

    # Accuracy plot
    axes[1].plot(epochs, [a * 100 for a in history['train_acc']], 'o-', label='Train Accuracy', color='#2ca02c', linewidth=2)
    axes[1].plot(epochs, [a * 100 for a in history['val_acc']], 's--', label='Val Accuracy', color='#d62728', linewidth=2)
    axes[1].set_title('Classification Accuracy (%) vs. Epochs', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('Accuracy (%)', fontsize=11)
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Training curves saved to {save_path}")

def run_training(
    model_name: str = "mobilenet_v2",
    dataset_dir: str = "dataset",
    batch_size: int = 32,
    warmup_epochs: int = 4,
    finetune_epochs: int = 3,
    warmup_lr: float = 1e-3,
    finetune_lr: float = 1e-4,
    save_path: str = "models/best_model.pth",
    history_path: str = "reports/training_history.json"
) -> Dict[str, Any]:
    """
    Executes 2-stage transfer learning:
      Phase 1: Feature Extraction (frozen backbone)
      Phase 2: Fine-tuning (top layers unfrozen with reduced learning rate)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, class_names = create_dataloaders(
        dataset_dir=dataset_dir,
        batch_size=batch_size,
        num_workers=2
    )
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")

    # Build model with frozen backbone for initial stage
    model = get_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=True,
        freeze_backbone=True
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    
    # Phase 1: Train classification head only
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=warmup_lr, weight_decay=1e-4)

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    best_val_acc = 0.0
    start_time = time.time()

    print("\n" + "="*50)
    print(f"STAGE 1: Feature Extraction (Head Training - {warmup_epochs} epochs)")
    print("="*50)

    for epoch in range(1, warmup_epochs + 1):
        t0 = time.time()
        t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_acc = validate_epoch(model, val_loader, criterion, device)
        dur = time.time() - t0

        history['train_loss'].append(t_loss)
        history['train_acc'].append(t_acc)
        history['val_loss'].append(v_loss)
        history['val_acc'].append(v_acc)

        print(f"Epoch [{epoch}/{warmup_epochs}] ({dur:.1f}s) - "
              f"Train Loss: {t_loss:.4f}, Train Acc: {t_acc*100:.2f}% | "
              f"Val Loss: {v_loss:.4f}, Val Acc: {v_acc*100:.2f}%")

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            save_checkpoint({
                'model_name': model_name,
                'num_classes': num_classes,
                'class_names': class_names,
                'model_state_dict': model.state_dict(),
                'val_accuracy': v_acc,
                'epoch': epoch,
                'stage': 'feature_extraction'
            }, save_path)

    print("\n" + "="*50)
    print(f"STAGE 2: Fine-Tuning (Backbone Unfrozen - {finetune_epochs} epochs)")
    print("="*50)

    unfreeze_backbone(model, model_name=model_name, unfreeze_all=False)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=finetune_lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=finetune_epochs, eta_min=1e-6)

    for epoch in range(warmup_epochs + 1, warmup_epochs + finetune_epochs + 1):
        t0 = time.time()
        t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_acc = validate_epoch(model, val_loader, criterion, device)
        scheduler.step()
        dur = time.time() - t0

        history['train_loss'].append(t_loss)
        history['train_acc'].append(t_acc)
        history['val_loss'].append(v_loss)
        history['val_acc'].append(v_acc)

        print(f"Epoch [{epoch}/{warmup_epochs + finetune_epochs}] ({dur:.1f}s) - "
              f"Train Loss: {t_loss:.4f}, Train Acc: {t_acc*100:.2f}% | "
              f"Val Loss: {v_loss:.4f}, Val Acc: {v_acc*100:.2f}%")

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            save_checkpoint({
                'model_name': model_name,
                'num_classes': num_classes,
                'class_names': class_names,
                'model_state_dict': model.state_dict(),
                'val_accuracy': v_acc,
                'epoch': epoch,
                'stage': 'fine_tuning'
            }, save_path)

    total_time = time.time() - start_time
    print(f"\nTraining finished in {total_time/60:.2f} mins. Best Validation Accuracy: {best_val_acc*100:.2f}%")

    # Plot curves and save history
    plot_training_history(history)
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"Training history saved to {history_path}")

    return {
        'best_val_acc': best_val_acc,
        'history': history,
        'class_names': class_names
    }

if __name__ == "__main__":
    run_training()
