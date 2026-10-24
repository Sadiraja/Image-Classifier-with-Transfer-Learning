import os
import shutil
import tarfile
import random
from pathlib import Path
from typing import Tuple, List, Dict
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Mean and std for ImageNet pretrained models
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_transforms(img_size: int = 224) -> Dict[str, transforms.Compose]:
    """
    Returns data transformations for training, validation, and testing.
    Includes data augmentation for training to prevent overfitting.
    """
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)), # 256 for 224
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    return {
        'train': train_transform,
        'val': eval_transform,
        'test': eval_transform
    }

def prepare_dataset_splits(
    raw_archive: str = "data/flower_photos.tgz",
    output_dir: str = "dataset",
    sample_dir: str = "sample_images",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> List[str]:
    """
    Extracts raw archive, splits classes into train/val/test directories,
    and extracts a clean sample image for each class.
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Ratios must sum to 1.0"
    random.seed(seed)

    raw_extract_dir = os.path.join("data", "flower_photos")
    if not os.path.exists(raw_extract_dir):
        print(f"Extracting {raw_archive}...")
        with tarfile.open(raw_archive, "r:gz") as tar:
            tar.extractall(path="data")
        print("Extraction complete.")

    output_path = Path(output_dir)
    for split in ['train', 'val', 'test']:
        (output_path / split).mkdir(parents=True, exist_ok=True)

    sample_path = Path(sample_dir)
    sample_path.mkdir(parents=True, exist_ok=True)

    # Class directories inside raw_extract_dir
    class_dirs = [d for d in os.listdir(raw_extract_dir) 
                  if os.path.isdir(os.path.join(raw_extract_dir, d)) and not d.startswith('.')]
    class_dirs.sort()

    print(f"Found {len(class_dirs)} classes: {class_dirs}")

    split_counts = {'train': 0, 'val': 0, 'test': 0}

    for cls_name in class_dirs:
        src_cls_dir = os.path.join(raw_extract_dir, cls_name)
        images = [f for f in os.listdir(src_cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        images.sort()
        random.shuffle(images)

        # Save one sample image for demonstration/testing
        if images and not (sample_path / f"{cls_name}.jpg").exists():
            shutil.copy2(os.path.join(src_cls_dir, images[0]), sample_path / f"{cls_name}.jpg")

        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train + n_val]
        test_imgs = images[n_train + n_val:]

        for split, split_imgs in [('train', train_imgs), ('val', val_imgs), ('test', test_imgs)]:
            cls_target_dir = output_path / split / cls_name
            cls_target_dir.mkdir(parents=True, exist_ok=True)
            for img_name in split_imgs:
                dest = cls_target_dir / img_name
                if not dest.exists():
                    shutil.copy2(os.path.join(src_cls_dir, img_name), dest)
            split_counts[split] += len(split_imgs)

        print(f"Class '{cls_name}': {len(train_imgs)} train, {len(val_imgs)} val, {len(test_imgs)} test (Total: {n_total})")

    print(f"Splits summary: Train={split_counts['train']}, Val={split_counts['val']}, Test={split_counts['test']}")
    return class_dirs

def create_dataloaders(
    dataset_dir: str = "dataset",
    batch_size: int = 32,
    num_workers: int = 2,
    img_size: int = 224
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Creates and returns train, val, and test DataLoaders along with class names.
    """
    transforms_dict = get_transforms(img_size)

    train_dataset = datasets.ImageFolder(
        root=os.path.join(dataset_dir, 'train'),
        transform=transforms_dict['train']
    )
    val_dataset = datasets.ImageFolder(
        root=os.path.join(dataset_dir, 'val'),
        transform=transforms_dict['val']
    )
    test_dataset = datasets.ImageFolder(
        root=os.path.join(dataset_dir, 'test'),
        transform=transforms_dict['test']
    )

    class_names = train_dataset.classes

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )

    return train_loader, val_loader, test_loader, class_names

if __name__ == "__main__":
    classes = prepare_dataset_splits()
    print("Dataset preparation completed successfully!")
