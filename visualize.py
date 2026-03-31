"""
visualize_complete.py - Complete visualization with retrieval images
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import euclidean_distances
import os
import torch
import torchvision
import torchvision.transforms as transforms

print("="*60)
print("COMPLETE VISUALIZATION SCRIPT")
print("="*60)

# Create graphs folder
os.makedirs('graphs', exist_ok=True)

# ============================================
# LOAD EMBEDDINGS
# ============================================
print("\n1. Loading embeddings...")

models = ['contrastive', 'triplet_random', 'triplet_hard']
embeddings_dict = {}
labels_dict = {}

for model in models:
    emb_path = f'embeddings/{model}_test_embeddings.npy'
    label_path = f'embeddings/{model}_test_labels.npy'
    
    if os.path.exists(emb_path):
        embeddings_dict[model] = np.load(emb_path)
        labels_dict[model] = np.load(label_path)
        print(f"   ✅ Loaded {model}: {embeddings_dict[model].shape}")
    else:
        print(f"   ❌ {model} not found")

if not embeddings_dict:
    print("No embeddings found!")
    exit()

# ============================================
# RECALL@K FUNCTION
# ============================================
def recall_at_k(embeddings, labels, k):
    distances = euclidean_distances(embeddings)
    correct = 0
    n = len(embeddings)
    
    for i in range(n):
        distances[i, i] = np.inf
        top_k = np.argsort(distances[i])[:k]
        if np.any(labels[top_k] == labels[i]):
            correct += 1
    
    return correct / n

# ============================================
# RESULTS TABLE
# ============================================
print("\n2. Creating results table...")

results = {}
for model in embeddings_dict:
    emb = embeddings_dict[model]
    lab = labels_dict[model]
    
    r1 = recall_at_k(emb, lab, 1)
    r5 = recall_at_k(emb, lab, 5)
    r10 = recall_at_k(emb, lab, 10)
    
    results[model] = [r1, r5, r10]
    print(f"   {model}: R@1={r1:.3f}, R@5={r5:.3f}, R@10={r10:.3f}")

# Create table
fig, ax = plt.subplots(figsize=(10, 3))
ax.axis('off')

table_data = [
    ['Contrastive Loss', f"{results['contrastive'][0]*100:.1f}%", 
     f"{results['contrastive'][1]*100:.1f}%", f"{results['contrastive'][2]*100:.1f}%"],
    ['Triplet Loss (Random)', f"{results['triplet_random'][0]*100:.1f}%", 
     f"{results['triplet_random'][1]*100:.1f}%", f"{results['triplet_random'][2]*100:.1f}%"],
    ['Triplet Loss (Hard)', f"{results['triplet_hard'][0]*100:.1f}%", 
     f"{results['triplet_hard'][1]*100:.1f}%", f"{results['triplet_hard'][2]*100:.1f}%"]
]

table = ax.table(cellText=table_data,
                 colLabels=['Model', 'Recall@1', 'Recall@5', 'Recall@10'],
                 cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 2)

plt.title('Retrieval Performance Comparison', fontsize=14, pad=20)
plt.savefig('graphs/results_table.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Saved: graphs/results_table.png")

# ============================================
# t-SNE PLOTS
# ============================================
print("\n3. Creating t-SNE plots...")

for model in embeddings_dict:
    emb = embeddings_dict[model]
    lab = labels_dict[model]
    
    # Sample 2000 points for faster t-SNE
    if len(emb) > 2000:
        idx = np.random.choice(len(emb), 2000, replace=False)
        emb = emb[idx]
        lab = lab[idx]
        print(f"   {model}: Sampled 2000 points")
    
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    emb_2d = tsne.fit_transform(emb)
    
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], 
                         c=lab, cmap='tab20', alpha=0.7, s=10)
    plt.colorbar(scatter, label='Class ID')
    plt.title(f't-SNE: {model.replace("_", " ").title()}', fontsize=14)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid(True, alpha=0.3)
    
    plt.savefig(f'graphs/tsne_{model}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: graphs/tsne_{model}.png")

# ============================================
# RETRIEVAL VISUALIZATIONS
# ============================================
print("\n4. Creating retrieval visualizations...")

# Load test images
print("   Loading test images...")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = torchvision.datasets.Caltech101(
    root='./data',
    download=True,
    transform=transform
)

# Denormalization for display
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

def denormalize(img):
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
    if img.shape[0] == 3:
        img = img.transpose(1, 2, 0)
    img = img * std + mean
    img = np.clip(img, 0, 1)
    return img

# For each model, create retrieval images
for model in embeddings_dict:
    print(f"\n   Creating retrieval for {model}...")
    
    # Create folder
    folder = f'graphs/{model}_retrieval'
    os.makedirs(folder, exist_ok=True)
    
    embeddings = embeddings_dict[model]
    labels = labels_dict[model]
    
    # Select 10 random queries
    np.random.seed(42)
    all_indices = list(range(len(embeddings)))
    query_indices = np.random.choice(all_indices, min(10, len(all_indices)), replace=False)
    
    for q_idx in query_indices:
        query_emb = embeddings[q_idx]
        
        # Compute distances
        distances = euclidean_distances(query_emb.reshape(1, -1), embeddings)[0]
        distances[q_idx] = np.inf
        top_k_indices = np.argsort(distances)[:5]
        
        # Create figure
        fig, axes = plt.subplots(1, 6, figsize=(18, 4))
        
        # Query image
        img, _ = dataset[q_idx]
        axes[0].imshow(denormalize(img))
        axes[0].set_title(f'QUERY\nClass {labels[q_idx]}', fontsize=10, fontweight='bold')
        axes[0].axis('off')
        
        # Retrieved images
        for i, idx in enumerate(top_k_indices):
            img, _ = dataset[idx]
            axes[i+1].imshow(denormalize(img))
            is_correct = labels[idx] == labels[q_idx]
            color = 'green' if is_correct else 'red'
            axes[i+1].set_title(f'Top-{i+1}\nClass {labels[idx]}', fontsize=9, color=color)
            axes[i+1].axis('off')
        
        plt.suptitle(f'{model.replace("_", " ").title()} - Retrieval Results', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'{folder}/query_{q_idx}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"   ✅ Saved {len(query_indices)} images to {folder}/")

# ============================================
# SUMMARY
# ============================================
print("\n" + "="*60)
print("VISUALIZATION COMPLETE!")
print("="*60)
print("\n📁 Files created in graphs/ folder:")
print("   ✅ results_table.png")
print("   ✅ tsne_contrastive.png")
print("   ✅ tsne_triplet_random.png")
print("   ✅ tsne_triplet_hard.png")
print("   ✅ contrastive_retrieval/ (10 images)")
print("   ✅ triplet_random_retrieval/ (10 images)")
print("   ✅ triplet_hard_retrieval/ (10 images)")

print("\n🎯 NEXT STEPS:")
print("   1. Open graphs/ folder")
print("   2. Use these images in your report")
print("   3. Create Report.pdf with analysis")
print("\n" + "="*60)