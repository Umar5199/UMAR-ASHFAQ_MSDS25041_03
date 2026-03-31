"""
save_embeddings.py - Fixed with proper resizing
"""

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Import the exact model that worked
from model_exact import EmbeddingNet

# ============================================
# CUSTOM DATASET WITH PROPER RESIZING
# ============================================
class FixedDataset(Dataset):
    """Fix image sizes and convert to RGB"""
    def __init__(self, dataset):
        self.dataset = dataset
        # Fixed transform: resize first, then convert to tensor
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Fixed size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        
        # Convert to PIL if it's a tensor
        if isinstance(img, torch.Tensor):
            img = transforms.ToPILImage()(img)
        
        # Convert to RGB if grayscale
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Apply transform (resize, to_tensor, normalize)
        img = self.transform(img)
        
        return img, label

# ============================================
# CONFIGURATION
# ============================================
BATCH_SIZE = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("="*60)
print("SAVING EMBEDDINGS FROM TRAINED MODELS")
print("="*60)
print(f"Device: {DEVICE}")

# Create embeddings folder
os.makedirs('embeddings', exist_ok=True)

# ============================================
# LOAD TEST DATASET
# ============================================
print("\n1. Loading Caltech-101 test dataset...")

# Load raw dataset (without transform)
print("Loading dataset...")
raw_dataset = torchvision.datasets.Caltech101(
    root='./data', 
    download=True,
    transform=None  # Load raw images
)

print(f"   Total dataset size: {len(raw_dataset)}")

# Apply fixes: resize to 224x224 + RGB conversion
dataset = FixedDataset(raw_dataset)
print(f"   All images resized to 224x224 and converted to RGB")

test_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
print(f"   Created dataloader with {len(test_loader)} batches")

# ============================================
# FUNCTION TO SAVE EMBEDDINGS
# ============================================
def save_embeddings(model_path, model_name, loader):
    """
    Generate and save embeddings for a model.
    """
    print(f"\n{'='*50}")
    print(f"Processing {model_name}...")
    print(f"{'='*50}")
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return False
    
    # Check file size
    size = os.path.getsize(model_path) / (1024 * 1024)
    print(f"   File size: {size:.1f} MB")
    
    try:
        # Load model
        print(f"   Loading model...")
        model = EmbeddingNet().to(DEVICE)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=DEVICE)
        
        # Extract model weights
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print(f"   Found model_state_dict in checkpoint")
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print(f"   Found state_dict in checkpoint")
        else:
            state_dict = checkpoint
            print(f"   Using checkpoint directly as weights")
        
        # Load into model
        model.load_state_dict(state_dict)
        model.eval()
        print(f"   ✅ Model loaded successfully!")
        
        # Generate embeddings
        print(f"   Generating embeddings for {len(loader.dataset)} images...")
        all_embeddings = []
        all_labels = []
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(loader):
                images = images.to(DEVICE)
                embeddings = model(images)
                
                all_embeddings.append(embeddings.cpu().numpy())
                all_labels.append(labels.numpy())
                
                # Progress indicator
                if (batch_idx + 1) % 50 == 0:
                    print(f"      Processed {batch_idx + 1}/{len(loader)} batches")
        
        # Combine all batches
        embeddings = np.vstack(all_embeddings)
        labels = np.concatenate(all_labels)
        
        print(f"   ✅ Embeddings generated!")
        print(f"      Shape: {embeddings.shape}")
        print(f"      Norm range: [{np.linalg.norm(embeddings[0]):.4f}, {np.linalg.norm(embeddings[-1]):.4f}]")
        
        # Save files
        emb_save_path = f'embeddings/{model_name}_test_embeddings.npy'
        label_save_path = f'embeddings/{model_name}_test_labels.npy'
        
        np.save(emb_save_path, embeddings)
        np.save(label_save_path, labels)
        
        print(f"   ✅ Saved embeddings to: {emb_save_path}")
        print(f"   ✅ Saved labels to: {label_save_path}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================
# PROCESS ALL MODELS
# ============================================
print("\n" + "="*60)
print("2. Processing Models")
print("="*60)

# Models to process
models = {
    'contrastive': 'weights/contrastive_final.pth',
    'triplet_random': 'weights/triplet_random_best.pth',
    'triplet_hard': 'weights/triplet_hard_final.pth'
}

# Check which files exist
print("\nChecking model files:")
for name, path in models.items():
    if os.path.exists(path):
        size = os.path.getsize(path) / (1024 * 1024)
        print(f"  ✅ {name}: {size:.1f} MB")
    else:
        print(f"  ❌ {name}: File not found at {path}")

# Process each model
results = {}
for name, path in models.items():
    if os.path.exists(path):
        results[name] = save_embeddings(path, name, test_loader)
    else:
        results[name] = False

# ============================================
# SUMMARY
# ============================================
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

success_count = sum(results.values())
total_count = len(results)

print(f"\n✅ Successfully processed: {success_count}/{total_count} models")

for name, success in results.items():
    if success:
        print(f"   ✅ {name}: embeddings saved")
    else:
        print(f"   ❌ {name}: failed")

if success_count == total_count:
    print("\n" + "🎉"*20)
    print("ALL EMBEDDINGS SAVED SUCCESSFULLY!")
    print("🎉"*20)
    print("\nNext steps:")
    print("   1. Run: python visualize.py")
    print("   2. Check graphs/ folder for visualizations")
    print("   3. Use these images in your report")
else:
    print("\n⚠️ Some models failed. Check the errors above.")

print("\n" + "="*60)