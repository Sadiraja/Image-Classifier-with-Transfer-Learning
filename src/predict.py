import sys
import os
import argparse
from pathlib import Path
from typing import List, Tuple
from PIL import Image
import torch
from torchvision import transforms

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.classifier import load_checkpoint
from src.dataset import IMAGENET_MEAN, IMAGENET_STD

def get_inference_transform(img_size: int = 224) -> transforms.Compose:
    """Prepares transformation pipeline for input images."""
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def predict_image(
    image_path: str,
    model_path: str = "models/best_model.pth",
    top_k: int = 3,
    device: torch.device = None
) -> List[Tuple[str, float]]:
    """
    Predicts class probabilities for a single image.
    Returns list of (class_name, confidence_percentage) tuples.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, checkpoint = load_checkpoint(model_path, device=device)
    class_names = checkpoint.get("class_names", [])

    img = Image.open(image_path).convert("RGB")
    transform = get_inference_transform()
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    top_probs, top_indices = torch.topk(probs, min(top_k, len(class_names)))
    
    results = []
    for prob, idx in zip(top_probs, top_indices):
        results.append((class_names[idx.item()], prob.item() * 100.0))

    return results

def main():
    parser = argparse.ArgumentParser(description="Classify an image using the trained transfer learning model.")
    parser.add_argument("image_path", type=str, help="Path to image file")
    parser.add_argument("--model", type=str, default="models/best_model.pth", help="Path to checkpoint")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top predictions to display")
    args = parser.parse_args()

    if not Path(args.image_path).exists():
        print(f"Error: Image file not found: {args.image_path}")
        sys.exit(1)

    print(f"\nAnalyzing image: {args.image_path}")
    results = predict_image(args.image_path, model_path=args.model, top_k=args.top_k)

    print("\nPredictions:")
    print("-" * 35)
    for rank, (cls, conf) in enumerate(results, 1):
        print(f" {rank}. {cls.capitalize():<12} : {conf:6.2f}%")
    print("-" * 35)

if __name__ == "__main__":
    main()
