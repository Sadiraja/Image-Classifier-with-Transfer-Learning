import os
from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
from torchvision import models

def get_model(
    model_name: str = "mobilenet_v2",
    num_classes: int = 5,
    pretrained: bool = True,
    dropout: float = 0.3,
    freeze_backbone: bool = False
) -> nn.Module:
    """
    Constructs a transfer learning model (MobileNetV2 or ResNet18)
    with a customized classification head for the target number of classes.
    """
    model_name = model_name.lower()
    
    if model_name == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        
        if freeze_backbone:
            for param in model.features.parameters():
                param.requires_grad = False
                
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(256, num_classes)
        )
        
    elif model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        
        if freeze_backbone:
            for name, param in model.named_parameters():
                if not name.startswith("fc"):
                    param.requires_grad = False
                    
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(256, num_classes)
        )
    else:
        raise ValueError(f"Unsupported model architecture: {model_name}. Choose 'mobilenet_v2' or 'resnet18'.")
        
    return model

def unfreeze_backbone(model: nn.Module, model_name: str = "mobilenet_v2", unfreeze_all: bool = False) -> None:
    """
    Unfreezes parameters for fine-tuning.
    If unfreeze_all is False, only the top feature layers are unfrozen.
    """
    model_name = model_name.lower()
    if unfreeze_all:
        for param in model.parameters():
            param.requires_grad = True
    else:
        if model_name == "mobilenet_v2":
            # Unfreeze the last 4 inverted residual blocks in features
            total_blocks = len(model.features)
            for i, block in enumerate(model.features):
                if i >= total_blocks - 4:
                    for param in block.parameters():
                        param.requires_grad = True
        elif model_name == "resnet18":
            # Unfreeze layer4 and fc
            for param in model.layer4.parameters():
                param.requires_grad = True
            for param in model.fc.parameters():
                param.requires_grad = True

def save_checkpoint(
    state: Dict[str, Any],
    filepath: str = "models/best_model.pth"
) -> None:
    """Saves model checkpoint along with metadata."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)
    print(f"Checkpoint successfully saved to {filepath}")

def load_checkpoint(
    filepath: str = "models/best_model.pth",
    device: Optional[torch.device] = None
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Loads model architecture and checkpoint state."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found at: {filepath}")
        
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    model_name = checkpoint.get("model_name", "mobilenet_v2")
    num_classes = checkpoint.get("num_classes", 5)
    
    model = get_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=False,
        freeze_backbone=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    return model, checkpoint
