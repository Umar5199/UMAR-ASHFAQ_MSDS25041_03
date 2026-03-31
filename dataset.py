import torch
from torch.utils.data import Dataset
import random
from PIL import Image
import torchvision.transforms as transforms

class ContrastiveDataset(Dataset):
    """Returns pairs (img1, img2, label) where label=1 for same class, 0 for different."""
    
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        self.indices = list(range(len(dataset)))
        self.labels = []
        
        # Store labels
        for idx in range(len(dataset)):
            _, label = dataset[idx]
            self.labels.append(label)
        
        # Create class indices
        self.class_indices = {}
        for idx, label in enumerate(self.labels):
            if label not in self.class_indices:
                self.class_indices[label] = []
            self.class_indices[label].append(idx)
        self.classes = list(self.class_indices.keys())
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        # Get first image and label
        img1, label1 = self.dataset[self.indices[index]]
        
        # img1 might already be a tensor, if so we need to handle it
        if isinstance(img1, torch.Tensor):
            # Convert tensor back to PIL for transforms
            img1 = transforms.ToPILImage()(img1)
        
        # Decide positive or negative
        if random.random() < 0.5:
            # Positive pair - same class
            same_class = self.class_indices[label1]
            pos_idx = random.choice(same_class)
            while pos_idx == index and len(same_class) > 1:
                pos_idx = random.choice(same_class)
            img2, _ = self.dataset[self.indices[pos_idx]]
            if isinstance(img2, torch.Tensor):
                img2 = transforms.ToPILImage()(img2)
            target = 1
        else:
            # Negative pair - different class
            other_classes = [c for c in self.classes if c != label1]
            if other_classes:
                neg_class = random.choice(other_classes)
                neg_idx = random.choice(self.class_indices[neg_class])
                img2, _ = self.dataset[self.indices[neg_idx]]
                if isinstance(img2, torch.Tensor):
                    img2 = transforms.ToPILImage()(img2)
            else:
                # Fallback
                same_class = self.class_indices[label1]
                neg_idx = random.choice(same_class)
                while neg_idx == index and len(same_class) > 1:
                    neg_idx = random.choice(same_class)
                img2, _ = self.dataset[self.indices[neg_idx]]
                if isinstance(img2, torch.Tensor):
                    img2 = transforms.ToPILImage()(img2)
            target = 0
        
        # Apply transforms
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return img1, img2, torch.tensor(target, dtype=torch.float32)


class TripletDataset(Dataset):
    """Returns triplets (anchor, positive, negative)."""
    
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        self.indices = list(range(len(dataset)))
        self.labels = []
        
        for idx in range(len(dataset)):
            _, label = dataset[idx]
            self.labels.append(label)
        
        self.class_indices = {}
        for idx, label in enumerate(self.labels):
            if label not in self.class_indices:
                self.class_indices[label] = []
            self.class_indices[label].append(idx)
        self.classes = list(self.class_indices.keys())
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        # Anchor
        anchor, anchor_label = self.dataset[self.indices[index]]
        if isinstance(anchor, torch.Tensor):
            anchor = transforms.ToPILImage()(anchor)
        
        # Positive - same class
        same_class = self.class_indices[anchor_label]
        pos_idx = random.choice(same_class)
        while pos_idx == index and len(same_class) > 1:
            pos_idx = random.choice(same_class)
        positive, _ = self.dataset[self.indices[pos_idx]]
        if isinstance(positive, torch.Tensor):
            positive = transforms.ToPILImage()(positive)
        
        # Negative - different class
        other_classes = [c for c in self.classes if c != anchor_label]
        if other_classes:
            neg_class = random.choice(other_classes)
            neg_idx = random.choice(self.class_indices[neg_class])
            negative, _ = self.dataset[self.indices[neg_idx]]
            if isinstance(negative, torch.Tensor):
                negative = transforms.ToPILImage()(negative)
        else:
            same_class = self.class_indices[anchor_label]
            neg_idx = random.choice(same_class)
            while neg_idx == index:
                neg_idx = random.choice(same_class)
            negative, _ = self.dataset[self.indices[neg_idx]]
            if isinstance(negative, torch.Tensor):
                negative = transforms.ToPILImage()(negative)
        
        # Apply transforms
        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)
        
        return anchor, positive, negative