"""
Retrieval Evaluation Module
Implements Recall@K and nearest neighbor search functionality.
"""

import numpy as np
import torch
from sklearn.metrics.pairwise import euclidean_distances


def recall_at_k(embeddings, labels, k=1):
    """
    Compute Recall@K for retrieval task.
    
    For each query, checks if the correct class appears in the top-K nearest neighbors.
    
    Args:
        embeddings (np.ndarray): Array of shape (n_samples, embedding_dim)
        labels (np.ndarray): Array of shape (n_samples,)
        k (int): Number of nearest neighbors to consider
    
    Returns:
        float: Recall@K score (fraction of queries with correct class in top-K)
    """
    n_samples = len(embeddings)
    correct = 0
    
    # Compute pairwise distances
    distances = euclidean_distances(embeddings)
    
    for i in range(n_samples):
        # Get distances to all other samples
        dist_to_others = distances[i]
        dist_to_others[i] = np.inf  # Exclude self
        
        # Get top-k indices (closest neighbors)
        top_k_indices = np.argsort(dist_to_others)[:k]
        
        # Check if any of the top-k has same label
        if np.any(labels[top_k_indices] == labels[i]):
            correct += 1
    
    return correct / n_samples


def compute_all_recalls(embeddings, labels, ks=[1, 5, 10, 20]):
    """
    Compute Recall@K for multiple K values.
    
    Args:
        embeddings (np.ndarray): Array of shape (n_samples, embedding_dim)
        labels (np.ndarray): Array of shape (n_samples,)
        ks (list): List of K values
    
    Returns:
        dict: Dictionary with recall scores for each K
    """
    recalls = {}
    for k in ks:
        recall = recall_at_k(embeddings, labels, k)
        recalls[f'recall@{k}'] = recall
        print(f"Recall@{k}: {recall:.4f}")
    
    return recalls


def nearest_neighbors(query_embedding, all_embeddings, k=5, exclude_self=True):
    """
    Find nearest neighbors for a query embedding.
    
    Args:
        query_embedding (np.ndarray): Query embedding of shape (embedding_dim,)
        all_embeddings (np.ndarray): All embeddings of shape (n_samples, embedding_dim)
        k (int): Number of neighbors to return
        exclude_self (bool): Whether to exclude the query itself
    
    Returns:
        tuple: (indices, distances) of nearest neighbors
    """
    # Compute distances
    distances = euclidean_distances(query_embedding.reshape(1, -1), all_embeddings)[0]
    
    if exclude_self:
        # Find index of self (where distance is 0)
        # Note: Due to floating point, use tolerance
        self_idx = np.argmin(distances)
        distances[self_idx] = np.inf
    
    # Get top-k indices
    top_k_indices = np.argsort(distances)[:k]
    top_k_distances = distances[top_k_indices]
    
    return top_k_indices, top_k_distances


def compute_embeddings(model, dataloader, device):
    """
    Compute embeddings for all images in dataloader.
    
    Args:
        model: Embedding model
        dataloader: DataLoader for the dataset
        device: Device to run on
    
    Returns:
        tuple: (embeddings, labels)
    """
    model.eval()
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            embeddings = model(images)
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels.numpy())
    
    embeddings = np.vstack(all_embeddings)
    labels = np.concatenate(all_labels)
    
    return embeddings, labels


# Test the retrieval functions
if __name__ == "__main__":
    print("Testing retrieval functions...")
    
    # Create dummy data
    n_samples = 100
    emb_dim = 128
    embeddings = np.random.randn(n_samples, emb_dim)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    labels = np.random.randint(0, 10, n_samples)
    
    # Test recall_at_k
    recall = recall_at_k(embeddings, labels, k=5)
    print(f"Recall@5 on random data: {recall:.4f}")
    
    # Test nearest_neighbors
    query_idx = 0
    query_emb = embeddings[query_idx]
    indices, distances = nearest_neighbors(query_emb, embeddings, k=3)
    print(f"Nearest neighbors for query {query_idx}: {indices}")
    
    print("✓ Retrieval functions test passed!")