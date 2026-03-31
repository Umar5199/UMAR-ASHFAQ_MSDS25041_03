import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import time

from model import EmbeddingNet
from dataset import ContrastiveDataset, TripletDataset
from loss import ContrastiveLoss, TripletLoss, batch_hard_mining
from utils import (
    load_caltech101, stratified_split,
    get_train_transform, get_val_test_transform
)
from retrieval import recall_at_k


class Config:
    embedding_dim = 128
    batch_size = 32
    epochs = 10  # You can change this
    learning_rate = 0.001
    num_workers = 0  # Set to 0 for Windows to avoid multiprocessing issues
    margin_contrastive = 1.0
    margin_triplet = 0.2
    save_checkpoint_interval = 10
    data_root = './data'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_dataloaders():
    """Create dataloaders for training."""
    print("\n" + "="*60)
    print("Loading Caltech-101 Dataset")
    print("="*60)
    
    # Load dataset
    dataset = load_caltech101(data_root=Config.data_root)
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = stratified_split(dataset)
    
    # Get transforms
    train_transform = get_train_transform()
    val_transform = get_val_test_transform()
    
    # Create dataloaders with transforms applied
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        collate_fn=lambda x: x  # Simple collate to handle PIL images
    )
    
    # For contrastive/triplet, we need to apply transforms differently
    # We'll apply transforms in the dataset classes
    
    return train_dataset, val_dataset, test_dataset, train_transform, val_transform


def evaluate_model(model, dataset, transform, device, k_values=[1, 5, 10]):
    """Evaluate model on dataset."""
    model.eval()
    all_embeddings = []
    all_labels = []
    
    # Create dataloader with transform
    class EvalDataset(torch.utils.data.Dataset):
        def __init__(self, dataset, transform):
            self.dataset = dataset
            self.transform = transform
        
        def __len__(self):
            return len(self.dataset)
        
        def __getitem__(self, idx):
            img, label = self.dataset[idx]
            img = self.transform(img)
            return img, label
    
    eval_dataset = EvalDataset(dataset, transform)
    eval_loader = DataLoader(eval_dataset, batch_size=Config.batch_size, 
                             shuffle=False, num_workers=Config.num_workers)
    
    with torch.no_grad():
        for images, labels in tqdm(eval_loader, desc="Evaluating", leave=False):
            images = images.to(device)
            embeddings = model(images)
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels.numpy())
    
    embeddings = np.vstack(all_embeddings)
    labels = np.concatenate(all_labels)
    
    recalls = {}
    for k in k_values:
        recall = recall_at_k(embeddings, labels, k)
        recalls[f'recall@{k}'] = recall
    
    return recalls, embeddings, labels


def train_contrastive(train_dataset, val_dataset, train_transform, val_transform, device):
    """Experiment 1: Contrastive Loss."""
    print("\n" + "="*60)
    print("Experiment 1: Contrastive Loss with Random Pairs")
    print("="*60)
    
    # Create contrastive dataset
    contrastive_dataset = ContrastiveDataset(train_dataset, transform=train_transform)
    train_loader = DataLoader(contrastive_dataset, batch_size=Config.batch_size,
                              shuffle=True, num_workers=Config.num_workers)
    
    model = EmbeddingNet(embedding_dim=Config.embedding_dim).to(device)
    criterion = ContrastiveLoss(margin=Config.margin_contrastive)
    optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)
    
    train_losses = []
    val_recalls = {'recall@1': [], 'recall@5': [], 'recall@10': []}
    best_recall = 0.0
    
    for epoch in range(Config.epochs):
        model.train()
        epoch_loss = 0
        start_time = time.time()
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{Config.epochs}')
        for img1, img2, target in pbar:
            img1, img2, target = img1.to(device), img2.to(device), target.to(device)
            
            emb1 = model(img1)
            emb2 = model(img2)
            loss = criterion(emb1, emb2, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Evaluate
        recalls, _, _ = evaluate_model(model, val_dataset, val_transform, device)
        for k in val_recalls.keys():
            val_recalls[k].append(recalls[k])
        
        epoch_time = time.time() - start_time
        print(f'Epoch {epoch+1}: Loss={avg_loss:.4f}, '
              f'Recall@1={recalls["recall@1"]:.4f}, Time={epoch_time:.1f}s')
        
        if recalls['recall@1'] > best_recall:
            best_recall = recalls['recall@1']
            torch.save(model.state_dict(), 'weights/contrastive_best.pth')
    
    torch.save(model.state_dict(), 'weights/contrastive_final.pth')
    return model, train_losses, val_recalls


def train_triplet_random(train_dataset, val_dataset, train_transform, val_transform, device):
    """Experiment 2: Triplet Loss with Random Triplets."""
    print("\n" + "="*60)
    print("Experiment 2: Triplet Loss with Random Triplets")
    print("="*60)
    
    triplet_dataset = TripletDataset(train_dataset, transform=train_transform)
    train_loader = DataLoader(triplet_dataset, batch_size=Config.batch_size,
                              shuffle=True, num_workers=Config.num_workers)
    
    model = EmbeddingNet(embedding_dim=Config.embedding_dim).to(device)
    criterion = TripletLoss(margin=Config.margin_triplet)
    optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)
    
    train_losses = []
    val_recalls = {'recall@1': [], 'recall@5': [], 'recall@10': []}
    best_recall = 0.0
    
    for epoch in range(Config.epochs):
        model.train()
        epoch_loss = 0
        start_time = time.time()
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{Config.epochs}')
        for anchor, positive, negative in pbar:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            
            emb_anchor = model(anchor)
            emb_positive = model(positive)
            emb_negative = model(negative)
            loss = criterion(emb_anchor, emb_positive, emb_negative)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        recalls, _, _ = evaluate_model(model, val_dataset, val_transform, device)
        for k in val_recalls.keys():
            val_recalls[k].append(recalls[k])
        
        epoch_time = time.time() - start_time
        print(f'Epoch {epoch+1}: Loss={avg_loss:.4f}, '
              f'Recall@1={recalls["recall@1"]:.4f}, Time={epoch_time:.1f}s')
        
        if recalls['recall@1'] > best_recall:
            best_recall = recalls['recall@1']
            torch.save(model.state_dict(), 'weights/triplet_random_best.pth')
    
    torch.save(model.state_dict(), 'weights/triplet_random_final.pth')
    return model, train_losses, val_recalls


def train_triplet_hard(train_dataset, val_dataset, train_transform, val_transform, device):
    """Experiment 3: Triplet Loss with Hard Negative Mining."""
    print("\n" + "="*60)
    print("Experiment 3: Triplet Loss with Hard Negative Mining")
    print("="*60)
    
    # For hard mining, we need batches with labels
    class HardDataset(torch.utils.data.Dataset):
        def __init__(self, dataset, transform):
            self.dataset = dataset
            self.transform = transform
        
        def __len__(self):
            return len(self.dataset)
        
        def __getitem__(self, idx):
            img, label = self.dataset[idx]
            img = self.transform(img)
            return img, label
    
    hard_dataset = HardDataset(train_dataset, train_transform)
    train_loader = DataLoader(hard_dataset, batch_size=Config.batch_size,
                              shuffle=True, num_workers=Config.num_workers)
    
    model = EmbeddingNet(embedding_dim=Config.embedding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)
    
    train_losses = []
    val_recalls = {'recall@1': [], 'recall@5': [], 'recall@10': []}
    best_recall = 0.0
    
    for epoch in range(Config.epochs):
        model.train()
        epoch_loss = 0
        start_time = time.time()
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{Config.epochs}')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            embeddings = model(images)
            loss = batch_hard_mining(embeddings, labels, margin=Config.margin_triplet)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        recalls, _, _ = evaluate_model(model, val_dataset, val_transform, device)
        for k in val_recalls.keys():
            val_recalls[k].append(recalls[k])
        
        epoch_time = time.time() - start_time
        print(f'Epoch {epoch+1}: Loss={avg_loss:.4f}, '
              f'Recall@1={recalls["recall@1"]:.4f}, Time={epoch_time:.1f}s')
        
        if recalls['recall@1'] > best_recall:
            best_recall = recalls['recall@1']
            torch.save(model.state_dict(), 'weights/triplet_hard_best.pth')
    
    torch.save(model.state_dict(), 'weights/triplet_hard_final.pth')
    return model, train_losses, val_recalls


def main():
    print("="*60)
    print("Deep Learning Assignment - Metric Learning on Caltech-101")
    print("="*60)
    print(f"Device: {Config.device}")
    print(f"Epochs: {Config.epochs}")
    
    # Create directories
    os.makedirs('weights', exist_ok=True)
    os.makedirs('graphs', exist_ok=True)
    
    # Create dataloaders
    train_dataset, val_dataset, test_dataset, train_transform, val_transform = create_dataloaders()
    
    print("\n" + "="*60)
    print("Starting Experiments")
    print("="*60)
    
    # Experiment 1
    model1, losses1, recalls1 = train_contrastive(
        train_dataset, val_dataset, train_transform, val_transform, Config.device
    )
    
    # Experiment 2
    model2, losses2, recalls2 = train_triplet_random(
        train_dataset, val_dataset, train_transform, val_transform, Config.device
    )
    
    # Experiment 3
    model3, losses3, recalls3 = train_triplet_hard(
        train_dataset, val_dataset, train_transform, val_transform, Config.device
    )
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(losses1, label='Contrastive Loss')
    plt.plot(losses2, label='Triplet Random')
    plt.plot(losses3, label='Triplet Hard')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(recalls1['recall@1'], label='Contrastive')
    plt.plot(recalls2['recall@1'], label='Triplet Random')
    plt.plot(recalls3['recall@1'], label='Triplet Hard')
    plt.xlabel('Epoch')
    plt.ylabel('Recall@1')
    plt.title('Validation Recall@1')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('graphs/training_curves.png', dpi=150)
    plt.show()
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)


if __name__ == "__main__":
    main()