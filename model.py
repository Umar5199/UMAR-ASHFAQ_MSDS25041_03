"""
Embedding Network Module
Builds embedding model using ResNet-50 pretrained on ImageNet as backbone.
Replaces classification head with projection layer and L2 normalization.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class EmbeddingNet(nn.Module):
    """
    Embedding network using pretrained ResNet-50 backbone.
    Projects image features to a normalized embedding space.
    
    Architecture:
        - Backbone: ResNet-50 (pretrained on ImageNet)
        - Remove final classification layer (fc)
        - Add projection head: Linear -> BatchNorm -> ReLU -> Linear
        - L2 normalize output embeddings
    
    Args:
        embedding_dim (int): Dimension of the output embedding space. Default: 128
        pretrained (bool): Whether to use pretrained weights. Default: True
    """
    
    def __init__(self, embedding_dim=128, pretrained=True):
        super(EmbeddingNet, self).__init__()
        
        # Load pretrained ResNet-50 backbone
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Get the number of input features to the final layer
        in_features = self.backbone.fc.in_features
        
        # Remove the classification head (replace with identity)
        self.backbone.fc = nn.Identity()
        
        # Projection head: maps 2048-dim features to embedding space
        self.projection = nn.Sequential(
            nn.Linear(in_features, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, H, W)
                              H and W should be at least 224x224.
        
        Returns:
            torch.Tensor: Normalized embeddings of shape (batch_size, embedding_dim)
                          Each embedding has L2 norm = 1.
        """
        # Extract features using backbone
        features = self.backbone(x)  # Shape: (batch_size, 2048)
        
        # Project to embedding space
        embeddings = self.projection(features)  # Shape: (batch_size, embedding_dim)
        
        # L2 normalize along the embedding dimension
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings


# Test the model
if __name__ == "__main__":
    print("Testing EmbeddingNet...")
    model = EmbeddingNet(embedding_dim=128)
    dummy_input = torch.randn(4, 3, 224, 224)
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output norms: {torch.norm(output, dim=1)}")  # Should be all 1.0
    print("✓ Model test passed!")