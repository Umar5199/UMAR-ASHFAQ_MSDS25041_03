"""
Inference Pipeline
Load trained model and generate embeddings for input images.
Works independently of training code.
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import argparse

from model import EmbeddingNet


class EmbeddingInference:
    """
    Inference pipeline for generating embeddings from images.
    
    Can accept single images or batches of images and returns
    normalized embedding vectors.
    """
    
    def __init__(self, model_path, embedding_dim=128, device=None):
        """
        Initialize inference pipeline.
        
        Args:
            model_path (str): Path to saved model weights
            embedding_dim (int): Dimension of embedding space
            device (torch.device): Device to run inference on
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Load model
        self.model = EmbeddingNet(embedding_dim=embedding_dim).to(self.device)
        
        # Load weights
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Model loaded from {model_path}")
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model.eval()
        
        # Define preprocessing transforms (must match training transforms)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def generate_embedding(self, image_path):
        """
        Generate embedding for a single image.
        
        Args:
            image_path (str): Path to image file
            
        Returns:
            np.ndarray: Embedding vector of shape (embedding_dim,)
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0)  # Add batch dimension
        image_tensor = image_tensor.to(self.device)
        
        # Generate embedding
        with torch.no_grad():
            embedding = self.model(image_tensor)
        
        return embedding.cpu().numpy().flatten()
    
    def generate_embeddings_batch(self, image_paths):
        """
        Generate embeddings for multiple images.
        
        Args:
            image_paths (list): List of image paths
            
        Returns:
            np.ndarray: Embeddings of shape (len(image_paths), embedding_dim)
        """
        embeddings = []
        for path in image_paths:
            emb = self.generate_embedding(path)
            embeddings.append(emb)
        return np.array(embeddings)
    
    def generate_embedding_from_tensor(self, image_tensor):
        """
        Generate embedding from preprocessed image tensor.
        
        Args:
            image_tensor (torch.Tensor): Image tensor of shape (C, H, W) or (B, C, H, W)
            
        Returns:
            np.ndarray: Embedding vector(s)
        """
        # Add batch dimension if needed
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        image_tensor = image_tensor.to(self.device)
        
        with torch.no_grad():
            embeddings = self.model(image_tensor)
        
        return embeddings.cpu().numpy()
    
    def get_model_info(self):
        """Print model information."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"\nModel Information:")
        print(f"  Device: {self.device}")
        print(f"  Embedding dimension: {self.model.projection[-1].out_features}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")


def main():
    """Example usage of inference pipeline."""
    parser = argparse.ArgumentParser(description='Generate embeddings for images')
    parser.add_argument('--model', type=str, required=True, 
                        help='Path to model weights')
    parser.add_argument('--image', type=str, nargs='+', 
                        help='Path(s) to image file(s)')
    parser.add_argument('--embedding_dim', type=int, default=128,
                        help='Embedding dimension')
    
    args = parser.parse_args()
    
    # Initialize inference
    inference = EmbeddingInference(
        model_path=args.model,
        embedding_dim=args.embedding_dim
    )
    
    inference.get_model_info()
    
    if args.image:
        if len(args.image) == 1:
            # Single image
            embedding = inference.generate_embedding(args.image[0])
            print(f"\nEmbedding for {args.image[0]}:")
            print(f"  Shape: {embedding.shape}")
            print(f"  Norm: {np.linalg.norm(embedding):.4f}")
            print(f"  First 10 values: {embedding[:10]}")
        else:
            # Multiple images
            embeddings = inference.generate_embeddings_batch(args.image)
            print(f"\nEmbeddings for {len(args.image)} images:")
            print(f"  Shape: {embeddings.shape}")
            print(f"  Norms: {np.linalg.norm(embeddings, axis=1)}")


if __name__ == "__main__":
    # Example usage
    print("Testing inference pipeline...")
    
    # Create dummy inference (will fail if model doesn't exist)
    try:
        inference = EmbeddingInference('weights/contrastive_final.pth')
        inference.get_model_info()
        
        # Generate dummy embedding
        dummy_image = torch.randn(3, 224, 224)
        embedding = inference.generate_embedding_from_tensor(dummy_image)
        print(f"\nDummy embedding shape: {embedding.shape}")
        print(f"Embedding norm: {np.linalg.norm(embedding):.4f}")
        
    except FileNotFoundError:
        print("Model not found. Train a model first using train.py")
    
    print("\nTo use inference:")
    print("  python inference.py --model weights/contrastive_final.pth --image image1.jpg")
    print("  python inference.py --model weights/contrastive_final.pth --image img1.jpg img2.jpg img3.jpg")