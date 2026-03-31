import os
import torch
from torch.utils.data import Subset, Dataset
from torchvision import datasets, transforms
import numpy as np
from collections import defaultdict
import random
from PIL import Image

class RGBDataset(Dataset):
    """Wrapper to convert grayscale images to RGB."""
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        # Convert to RGB if grayscale
        if isinstance(img, Image.Image):
            if img.mode != 'RGB':
                img = img.convert('RGB')
        elif isinstance(img, torch.Tensor):
            if img.shape[0] == 1:
                img = img.repeat(3, 1, 1)
        return img, label

def load_caltech101(data_root='./data', download=True):
    """Load Caltech-101 dataset with RGB conversion."""
    
    print("Loading Caltech-101 dataset...")
    
    # Simple transform for loading (no normalization yet)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    try:
        # Use torchvision's Caltech101
        dataset = datasets.Caltech101(
            root=data_root,
            download=download,
            transform=transform
        )
        print(f"Dataset loaded! Size: {len(dataset)}")
        
        # Wrap with RGB converter
        dataset = RGBDataset(dataset)
        return dataset
        
    except Exception as e:
        print(f"Error: {e}")
        raise

def stratified_split(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """Split dataset preserving class distribution."""
    
    # Get labels
    labels = []
    for idx in range(len(dataset)):
        _, label = dataset.dataset[idx] if hasattr(dataset, 'dataset') else dataset[idx]
        labels.append(label)
    labels = np.array(labels)
    
    # Group indices by class
    class_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        class_indices[label].append(idx)
    
    train_indices = []
    val_indices = []
    test_indices = []
    
    np.random.seed(seed)
    random.seed(seed)
    
    for label, indices in class_indices.items():
        shuffled = indices.copy()
        random.shuffle(shuffled)
        n = len(shuffled)
        
        n_train = int(train_ratio * n)
        n_val = int(val_ratio * n)
        
        train_indices.extend(shuffled[:n_train])
        val_indices.extend(shuffled[n_train:n_train + n_val])
        test_indices.extend(shuffled[n_train + n_val:])
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    
    print(f"\nDataset Split Statistics:")
    print(f"  Training:   {len(train_dataset)} images ({len(train_indices)/len(dataset)*100:.1f}%)")
    print(f"  Validation: {len(val_dataset)} images ({len(val_indices)/len(dataset)*100:.1f}%)")
    print(f"  Test:       {len(test_dataset)} images ({len(test_indices)/len(dataset)*100:.1f}%)")
    
    return train_dataset, val_dataset, test_dataset

def get_train_transform():
    """Get training transforms."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_val_test_transform():
    """Get validation/test transforms."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])