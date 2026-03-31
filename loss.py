"""
Loss Functions for Metric Learning
Implements Contrastive Loss, Triplet Loss, and Hard Negative Mining.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss for pairs.
    
    Formula:
        L = y * D² + (1 - y) * max(0, margin - D)²
    
    where D is the Euclidean distance between embeddings.
    
    Args:
        margin (float): Margin for negative pairs. Default: 1.0
    """
    
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, emb1, emb2, target):
        """
        Compute contrastive loss.
        
        Args:
            emb1 (torch.Tensor): Embeddings of first images (batch_size, embedding_dim)
            emb2 (torch.Tensor): Embeddings of second images (batch_size, embedding_dim)
            target (torch.Tensor): Binary labels (1 for positive, 0 for negative)
        
        Returns:
            torch.Tensor: Scalar loss value
        """
        # Compute Euclidean distance between embeddings
        D = torch.norm(emb1 - emb2, dim=1)  # Shape: (batch_size,)
        
        # Loss for positive pairs (target=1): minimize distance
        pos_loss = target * torch.pow(D, 2)
        
        # Loss for negative pairs (target=0): push apart with margin
        neg_loss = (1 - target) * torch.pow(torch.clamp(self.margin - D, min=0), 2)
        
        # Average loss over batch
        loss = torch.mean(pos_loss + neg_loss)
        
        return loss


class TripletLoss(nn.Module):
    """
    Triplet Loss.
    
    Formula:
        L = max(0, D(anchor, positive) - D(anchor, negative) + margin)
    
    Args:
        margin (float): Margin for triplet loss. Default: 0.2
    """
    
    def __init__(self, margin=0.2):
        super(TripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        """
        Compute triplet loss.
        
        Args:
            anchor (torch.Tensor): Anchor embeddings (batch_size, embedding_dim)
            positive (torch.Tensor): Positive embeddings (batch_size, embedding_dim)
            negative (torch.Tensor): Negative embeddings (batch_size, embedding_dim)
        
        Returns:
            torch.Tensor: Scalar loss value
        """
        # Compute distances
        pos_dist = torch.norm(anchor - positive, dim=1)  # Shape: (batch_size,)
        neg_dist = torch.norm(anchor - negative, dim=1)  # Shape: (batch_size,)
        
        # Compute triplet loss
        loss = torch.mean(torch.clamp(pos_dist - neg_dist + self.margin, min=0))
        
        return loss


def batch_hard_mining(embeddings, labels, margin=0.2):
    """
    Batch Hard Mining for triplet loss.
    
    For each anchor in the batch:
        - Hardest positive: same class with maximum distance
        - Hardest negative: different class with minimum distance
    
    Args:
        embeddings (torch.Tensor): Embeddings of shape (batch_size, embedding_dim)
        labels (torch.Tensor): Labels of shape (batch_size,)
        margin (float): Margin for triplet loss
    
    Returns:
        torch.Tensor: Triplet loss computed on hard triplets
    """
    batch_size = embeddings.size(0)
    device = embeddings.device
    
    # Compute pairwise distance matrix
    # Using Euclidean distance: ||a-b||² = ||a||² + ||b||² - 2*a·b
    # Since embeddings are normalized, ||a||² = ||b||² = 1
    # So Euclidean distance = sqrt(2 - 2*cosine_similarity)
    # But we can use torch.cdist directly
    distances = torch.cdist(embeddings, embeddings, p=2)  # Shape: (batch_size, batch_size)
    
    loss = 0
    valid_triplets = 0
    
    for i in range(batch_size):
        # Get mask for same class (excluding self)
        same_class_mask = (labels == labels[i]).float()
        same_class_mask[i] = 0
        
        # Get mask for different class
        diff_class_mask = (labels != labels[i]).float()
        
        # Hardest positive: same class, maximum distance
        if same_class_mask.sum() > 0:
            # Set distances of different class to -inf to ignore
            hardest_positive_dist = (distances[i] * same_class_mask).max()
            
            # Hardest negative: different class, minimum distance
            if diff_class_mask.sum() > 0:
                # Set distances of same class to inf to ignore
                diff_distances = distances[i].clone()
                diff_distances[diff_class_mask == 0] = float('inf')
                hardest_negative_dist = diff_distances.min()
                
                # Compute triplet loss for this anchor
                triplet_loss = torch.clamp(
                    hardest_positive_dist - hardest_negative_dist + margin, 
                    min=0
                )
                loss += triplet_loss
                valid_triplets += 1
    
    # Average over valid triplets
    if valid_triplets > 0:
        loss = loss / valid_triplets
    
    return loss


# Test the loss functions
if __name__ == "__main__":
    print("Testing loss functions...")
    
    batch_size = 8
    emb_dim = 128
    
    # Create dummy embeddings (normalized)
    embeddings = F.normalize(torch.randn(batch_size, emb_dim), dim=1)
    labels = torch.randint(0, 5, (batch_size,))
    
    # Test ContrastiveLoss
    contrastive_loss = ContrastiveLoss(margin=1.0)
    emb1 = embeddings[:4]
    emb2 = embeddings[4:]
    target = torch.tensor([1, 1, 0, 0], dtype=torch.float32)
    loss1 = contrastive_loss(emb1, emb2, target)
    print(f"Contrastive Loss: {loss1.item():.4f}")
    
    # Test TripletLoss
    triplet_loss = TripletLoss(margin=0.2)
    anchor = embeddings[:4]
    positive = embeddings[4:8]
    negative = embeddings[:4][[1, 0, 3, 2]]  # Random negative
    loss2 = triplet_loss(anchor, positive, negative)
    print(f"Triplet Loss: {loss2.item():.4f}")
    
    # Test Hard Negative Mining
    loss3 = batch_hard_mining(embeddings, labels, margin=0.2)
    print(f"Batch Hard Mining Loss: {loss3.item():.4f}")
    
    print("✓ Loss functions test passed!")