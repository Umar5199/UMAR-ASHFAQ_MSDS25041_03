"""
model_exact.py - Exactly matches your friend's model architecture
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class EmbeddingNet(nn.Module):
    """
    Exact model architecture that matches your friend's saved model.
    - Backbone: Sequential ResNet-50
    - Projection: Single Linear layer (no BatchNorm, no extra layers)
    """
    def __init__(self, embedding_dim=128):
        super(EmbeddingNet, self).__init__()
        
        # Load pretrained ResNet-50
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Extract all layers except the final FC
        layers = list(resnet.children())[:-1]
        
        # Create sequential backbone
        self.backbone = nn.Sequential(*layers)
        
        # Get the feature dimension (2048 for ResNet-50)
        self.feature_dim = 2048
        
        # Single linear projection (matches friend's format)
        self.projection = nn.Linear(self.feature_dim, embedding_dim)
        
    def forward(self, x):
        # Forward through backbone
        features = self.backbone(x)
        features = features.view(features.size(0), -1)  # Flatten to (batch, 2048)
        
        # Project to embedding space
        embeddings = self.projection(features)
        
        # L2 normalize
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings