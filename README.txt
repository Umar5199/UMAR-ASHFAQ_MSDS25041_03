================================================================================
DEEP LEARNING ASSIGNMENT 3: METRIC LEARNING ON CALTECH-101


This project implements three deep metric learning approaches for image retrieval:

1. Contrastive Loss with Random Pairs
2. Triplet Loss with Random Triplets
3. Triplet Loss with Hard Negative Mining

All models use a ResNet-50 backbone pretrained on ImageNet with 256-dimensional normalized embeddings.

================================================================================
RESULTS SUMMARY
================================================================================

| Model                    | Recall@1 | Recall@5 | Recall@10 |
|--------------------------|----------|----------|-----------|
| Contrastive Loss         | 16.6%    | 34.0%    | 43.7%     |
| Triplet Loss (Random)    | 14.4%    | 31.4%    | 40.2%     |
| Triplet Loss (Hard)      | 19.2%    | 35.2%    | 43.4%     |

Note: Hard negative mining achieved 2.3x faster convergence.

================================================================================
INSTALLATION
================================================================================

1. Install Python 3.8 or higher.

2. Install dependencies:
   pip install torch torchvision numpy matplotlib scikit-learn pillow tqdm

   OR use requirements.txt:
   pip install -r requirements.txt

================================================================================
DATASET
================================================================================

The Caltech-101 dataset is automatically downloaded when running the scripts.
You can also specify a custom dataset path using --data_root.

Default dataset location: ./data/caltech101

Example dataset structure:

/path/to/dataset/
├── class1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class2/
│   └── ...
└── ...

================================================================================
COMMAND-LINE USAGE
================================================================================

1. TRAINING (train.py)
----------------------

Usage:
   python train.py [--data_root PATH] [--epochs N] [--batch_size N] [--lr FLOAT]
                   [--embedding_dim N] [--device DEVICE]

Arguments:
   --data_root      Path to dataset (default: ./data)
   --epochs         Number of training epochs (default: 10)
   --batch_size     Batch size (default: 64)
   --lr             Learning rate (default: 0.001)
   --embedding_dim  Embedding dimension (default: 256)
   --device         Device to use: 'cuda' or 'cpu' (default: auto-detect)

Examples:
   python train.py
   python train.py --data_root /mnt/data/caltech101 --epochs 20
   python train.py --device cpu --batch_size 32

2. GENERATE EMBEDDINGS (save_embeddings.py)
-------------------------------------------

Usage:
   python save_embeddings.py [--data_root PATH] [--weights_dir PATH] 
                             [--output_dir PATH] [--batch_size N] [--device DEVICE]

Examples:
   python save_embeddings.py
   python save_embeddings.py --data_root /mnt/data --weights_dir ./weights

3. VISUALIZATION (visualize.py)
--------------------------------

Usage:
   python visualize.py [--embeddings_dir PATH] [--output_dir PATH] 
                       [--num_queries N] [--k N]

Examples:
   python visualize.py
   python visualize.py --num_queries 20 --k 10

4. INFERENCE (inference.py)
----------------------------

Usage:
   python inference.py --model MODEL_PATH --image IMAGE_PATH [--embedding_dim N]
                       [--device DEVICE]
   python inference.py --model MODEL_PATH --images IMAGE1 IMAGE2 IMAGE3

Examples:
   python inference.py --model weights/contrastive_best.pth --image test.jpg
   python inference.py --model weights/contrastive_best.pth --images img1.jpg img2.jpg
   python inference.py --model weights/contrastive_best.pth --image test.jpg --device cpu

5. EVALUATION (retrieval.py)
------------------------------

Usage:
   python retrieval.py [--embeddings_dir PATH] [--ks N1 N2 N3]

Example:
   python retrieval.py --ks 1 5 10

================================================================================
FILE STRUCTURE
================================================================================

project/
├── README.txt
├── requirements.txt
├── model.py
├── loss.py
├── dataset.py
├── utils.py
├── retrieval.py
├── train.py
├── save_embeddings.py
├── visualize.py
├── inference.py
├── weights/                 # Trained models (.pth files)
│   ├── contrastive_best.pth
│   ├── triplet_random_best.pth
│   └── triplet_hard_best.pth
├── embeddings/              # Precomputed embeddings (.npy)
│   ├── contrastive_test_embeddings.npy
│   ├── contrastive_test_labels.npy
│   ├── triplet_random_test_embeddings.npy
│   ├── triplet_random_test_labels.npy
│   ├── triplet_hard_test_embeddings.npy
│   └── triplet_hard_test_labels.npy
├── graphs/                  # Visualizations
│   ├── tsne_contrastive.png
│   ├── tsne_triplet_random.png
│   ├── tsne_triplet_hard.png
│   ├── contrastive_retrieval/
│   ├── triplet_random_retrieval/
│   └── triplet_hard_retrieval/
└── data/                    # Caltech-101 dataset

================================================================================
QUICK START
================================================================================

Option 1: Using Pre-trained Models
1. Place model files in weights/:
   - contrastive_best.pth
   - triplet_random_best.pth
   - triplet_hard_best.pth
2. Generate embeddings:
   python save_embeddings.py
3. Create visualizations:
   python visualize.py
4. Test inference:
   python inference.py --model weights/contrastive_best.pth --image sample.jpg

Option 2: Train from Scratch
1. python train.py
2. python save_embeddings.py
3. python visualize.py

================================================================================
TROUBLESHOOTING
================================================================================

- Out of Memory: Reduce batch size (--batch_size 16) or use CPU (--device cpu)
- Dataset Download Fails: Manual download: https://data.caltech.edu/records/mzrjq-6wc02
- Model Not Found: Ensure .pth files are in weights/ folder
- CUDA Not Available: Code will fallback to CPU

================================================================================
DEPENDENCIES
================================================================================

Python 3.8+
PyTorch 2.0+
torchvision 0.15+
numpy 1.21+
matplotlib 3.5+
scikit-learn 1.0+
Pillow 9.0+
tqdm 4.64+



